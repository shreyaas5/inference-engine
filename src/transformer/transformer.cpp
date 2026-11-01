#include "inference_engine/transformer.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "inference_engine/kernels.hpp"
#include "inference_engine/quant.hpp"

namespace inference_engine {

TransformerEngine::TransformerEngine() = default;
TransformerEngine::~TransformerEngine() = default;
TransformerEngine::TransformerEngine(TransformerEngine&&) noexcept = default;
TransformerEngine& TransformerEngine::operator=(TransformerEngine&&) noexcept = default;

Status TransformerEngine::init(const ModelConfig& config,
                               const ModelLoader* loader,
                               SimdBackend backend,
                               uint32_t num_threads) {
    initialized_ = false;

    Status st = config.validate();
    if (!st.ok()) {
        return st;
    }
    if (config.weight_dtype != DataType::FP32 && config.weight_dtype != DataType::Q4_0) {
        return Status::Error(StatusCode::InvalidArgument,
                             "TransformerEngine requires FP32 or Q4_0 model weights");
    }
    if (loader == nullptr) {
        return Status::Error(StatusCode::InvalidArgument,
                             "ModelLoader must not be null for weight-bound TransformerEngine");
    }
    if (!loader->is_loaded()) {
        return Status::Error(StatusCode::InvalidArgument,
                             "ModelLoader is not loaded");
    }

    // Verify architecture hyperparameters between passed config and loader->config()
    const auto& lc = loader->config();
    if (config.vocab_size != lc.vocab_size ||
        config.hidden_dim != lc.hidden_dim ||
        config.intermediate_dim != lc.intermediate_dim ||
        config.num_layers != lc.num_layers ||
        config.num_query_heads != lc.num_query_heads ||
        config.num_kv_heads != lc.num_kv_heads ||
        config.head_dim != lc.head_dim ||
        config.weight_dtype != lc.weight_dtype ||
        std::fabs(config.rms_norm_eps - lc.rms_norm_eps) > 1e-9f ||
        std::fabs(config.rope_theta - lc.rope_theta) > 1e-3f) {
        return Status::Error(StatusCode::InvalidArgument,
                             "TransformerEngine::init: config does not match loader config");
    }

    Status bind_st = loader->bind_transformer_weights(&bindings_);
    if (!bind_st.ok()) {
        return bind_st;
    }

    Status kv_st = kv_cache_.init(config);
    if (!kv_st.ok()) {
        return kv_st;
    }

    // Set up threading
    num_threads_ = (num_threads > 0) ? num_threads : 1;
    if (num_threads_ > 1) {
        thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    } else {
        thread_pool_.reset();
    }

    // Pre-allocate intermediate activations and scratch buffers
    x_.assign(config.hidden_dim, 0.0f);
    norm_buf_.assign(config.hidden_dim, 0.0f);
    q_.assign(config.q_dim(), 0.0f);
    k_.assign(config.kv_dim(), 0.0f);
    v_.assign(config.kv_dim(), 0.0f);
    attn_out_.assign(config.q_dim(), 0.0f);
    proj_out_.assign(config.hidden_dim, 0.0f);
    gate_.assign(config.intermediate_dim, 0.0f);
    up_.assign(config.intermediate_dim, 0.0f);
    ffn_act_.assign(config.intermediate_dim, 0.0f);
    scores_.assign(config.max_seq_len, 0.0f);

    // Precompute RoPE cos/sin lookup tables across all valid sequence positions (KI-006)
    const size_t half_dim = config.head_dim / 2;
    const size_t total_rope_entries = static_cast<size_t>(config.max_seq_len) * half_dim;
    rope_cos_.resize(total_rope_entries);
    rope_sin_.resize(total_rope_entries);
    for (uint32_t p = 0; p < config.max_seq_len; ++p) {
        float* cos_row = rope_cos_.data() + static_cast<size_t>(p) * half_dim;
        float* sin_row = rope_sin_.data() + static_cast<size_t>(p) * half_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            const float freq = 1.0f / std::pow(config.rope_theta, static_cast<float>(2 * i) / static_cast<float>(config.head_dim));
            const float angle = static_cast<float>(p) * freq;
            cos_row[i] = std::cos(angle);
            sin_row[i] = std::sin(angle);
        }
    }

    config_ = config;
    loader_ = loader;
    backend_ = backend;
    initialized_ = true;
    return Status::Ok();
}

void TransformerEngine::matmul(float* out,
                               const TensorView& weight,
                               const float* x,
                               size_t out_dim,
                               size_t in_dim) const noexcept {
    ThreadPool* pool = thread_pool_ ? thread_pool_.get() : nullptr;
    if (weight.dtype() == DataType::FP32) {
        dispatch_matmul_f32(out, weight.data_f32(), x, out_dim, in_dim, backend_, pool);
    } else if (weight.dtype() == DataType::Q4_0) {
        dispatch_matmul_q4_0_f32(out, weight.data_q4_0(), x, out_dim, in_dim, backend_, pool);
    }
}

Status TransformerEngine::forward_token(int32_t token_id, uint32_t pos, float* out_logits) {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "TransformerEngine is not initialized");
    }
    if (loader_ == nullptr || bindings_.embed_tokens.empty()) {
        return Status::Error(StatusCode::InvalidArgument, "Model weights are not bound");
    }
    if (out_logits == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_logits must not be null");
    }
    if (token_id < 0 || static_cast<uint32_t>(token_id) >= config_.vocab_size) {
        return Status::Error(StatusCode::OutOfBounds, "token_id out of vocabulary range");
    }
    if (pos >= config_.max_seq_len) {
        return Status::Error(StatusCode::OutOfBounds, "pos exceeds max_seq_len");
    }
    if (pos != kv_cache_.seq_len()) {
        return Status::Error(StatusCode::InvalidArgument,
                             "pos (" + std::to_string(pos) +
                             ") does not match kv_cache committed seq_len (" +
                             std::to_string(kv_cache_.seq_len()) + ")");
    }

    const uint32_t hidden_dim = config_.hidden_dim;
    const uint32_t q_dim = config_.q_dim();
    const uint32_t kv_dim = config_.kv_dim();
    const uint32_t intermediate_dim = config_.intermediate_dim;
    const float eps = config_.rms_norm_eps;

    // Step 1: Token embedding lookup: copy row `token_id` (FP32) or dequantize single row (Q4_0)
    if (bindings_.embed_tokens.dtype() == DataType::FP32) {
        const float* embed_table = bindings_.embed_tokens.data_f32();
        if (embed_table == nullptr) {
            return Status::Error(StatusCode::InvalidArgument, "embed_tokens has null FP32 data");
        }
        const float* embed_row = embed_table + static_cast<size_t>(token_id) * hidden_dim;
        std::memcpy(x_.data(), embed_row, static_cast<size_t>(hidden_dim) * sizeof(float));
    } else if (bindings_.embed_tokens.dtype() == DataType::Q4_0) {
        const BlockQ4_0* embed_blocks = bindings_.embed_tokens.data_q4_0();
        if (embed_blocks == nullptr) {
            return Status::Error(StatusCode::InvalidArgument, "embed_tokens has null Q4_0 data");
        }
        const size_t blocks_per_row = hidden_dim / kQ4_0BlockElements;
        const BlockQ4_0* row_blocks = embed_blocks + static_cast<size_t>(token_id) * blocks_per_row;
        Status deq_st = dequantize_row_q4_0(row_blocks, x_.data(), hidden_dim);
        if (!deq_st.ok()) {
            return deq_st;
        }
    } else {
        return Status::Error(StatusCode::InvalidArgument,
                             std::string("Unsupported embed_tokens dtype: ") +
                                 dtype_name(bindings_.embed_tokens.dtype()));
    }

    // Step 2: Loop over all decoder layers
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        const LayerWeightBindings& lw = bindings_.layers[layer_idx];

        // 2a. Pre-Attention RMSNorm: u = RMSNorm(x, w_input_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.input_layernorm.data_f32(), hidden_dim, eps);

        // 2b. Q/K/V Projections: q = W_q u, k = W_k u, v = W_v u
        matmul(q_.data(), lw.q_proj, norm_buf_.data(), q_dim, hidden_dim);
        matmul(k_.data(), lw.k_proj, norm_buf_.data(), kv_dim, hidden_dim);
        matmul(v_.data(), lw.v_proj, norm_buf_.data(), kv_dim, hidden_dim);

        // 2c. RoPE for Q and K (using precomputed cos/sin tables to avoid per-token trig)
        const size_t half_dim = config_.head_dim / 2;
        const float* cos_row = rope_cos_.data() + static_cast<size_t>(pos) * half_dim;
        const float* sin_row = rope_sin_.data() + static_cast<size_t>(pos) * half_dim;
        scalar::rope_inplace_cached_f32(q_.data(), config_.num_query_heads, config_.head_dim, cos_row, sin_row);
        scalar::rope_inplace_cached_f32(k_.data(), config_.num_kv_heads, config_.head_dim, cos_row, sin_row);

        // 2d. KV-Cache Write (append without advancing committed sequence length)
        Status app_st = kv_cache_.append(layer_idx, pos, k_.data(), v_.data());
        if (!app_st.ok()) {
            return app_st;
        }

        // 2e. Causal Grouped Query Attention (GQA)
        scalar::gqa_attention_f32(attn_out_.data(),
                                  q_.data(),
                                  kv_cache_,
                                  layer_idx,
                                  pos,
                                  config_.num_query_heads,
                                  config_.num_kv_heads,
                                  config_.head_dim,
                                  scores_.data());

        // 2f. Output Projection and First Residual: x <- x + W_o a
        matmul(proj_out_.data(), lw.o_proj, attn_out_.data(), hidden_dim, q_dim);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            x_[i] += proj_out_[i];
        }

        // 2g. Post-Attention RMSNorm: m = RMSNorm(x, w_post_attention_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.post_attention_layernorm.data_f32(), hidden_dim, eps);

        // 2h. SwiGLU Feed-Forward Network and Second Residual:
        // g = W_gate m, p = W_up m, h = SiLU(g) * p, x <- x + W_down h
        matmul(gate_.data(), lw.gate_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
        matmul(up_.data(), lw.up_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
        scalar::swiglu_f32(ffn_act_.data(), gate_.data(), up_.data(), intermediate_dim);
        matmul(proj_out_.data(), lw.down_proj, ffn_act_.data(), hidden_dim, intermediate_dim);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            x_[i] += proj_out_[i];
        }
    }

    // Step 3: Advance committed sequence length only after all layers have finished
    Status commit_st = kv_cache_.commit_token();
    if (!commit_st.ok()) {
        return commit_st;
    }

    // Step 4: Final RMSNorm and LM-Head projection
    scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), bindings_.final_norm.data_f32(), hidden_dim, eps);
    matmul(out_logits, bindings_.lm_head, norm_buf_.data(), config_.vocab_size, hidden_dim);

    return Status::Ok();
}

Status TransformerEngine::forward_prompt(const int32_t* prompt_tokens,
                                         size_t prompt_len,
                                         float* out_final_logits) {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "TransformerEngine is not initialized");
    }
    if (prompt_tokens == nullptr && prompt_len > 0) {
        return Status::Error(StatusCode::InvalidArgument, "prompt_tokens is null");
    }
    if (prompt_len == 0) {
        return Status::Ok();
    }
    const uint32_t start_pos = kv_cache_.seq_len();
    if (start_pos + prompt_len > config_.max_seq_len) {
        return Status::Error(StatusCode::OutOfBounds, "Prompt length exceeds remaining max_seq_len");
    }

    std::vector<float> step_logits;
    if (out_final_logits == nullptr) {
        step_logits.resize(config_.vocab_size);
    }

    for (size_t i = 0; i < prompt_len; ++i) {
        const uint32_t pos = start_pos + static_cast<uint32_t>(i);
        float* logits_dst = (i + 1 == prompt_len && out_final_logits != nullptr)
                                ? out_final_logits
                                : (out_final_logits != nullptr ? out_final_logits : step_logits.data());
        Status st = forward_token(prompt_tokens[i], pos, logits_dst);
        if (!st.ok()) {
            return st;
        }
    }
    return Status::Ok();
}

}  // namespace inference_engine
