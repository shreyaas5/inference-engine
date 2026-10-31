#include "inference_engine/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "inference_engine/kernels.hpp"

namespace inference_engine {

Status TransformerEngine::init(const ModelConfig& config,
                               const ModelLoader* loader,
                               SimdBackend backend,
                               uint32_t /*num_threads*/) {
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

    config_ = config;
    loader_ = loader;
    backend_ = backend;
    initialized_ = true;
    return Status::Ok();
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
    if (config_.weight_dtype != DataType::FP32) {
        return Status::Error(StatusCode::NotImplemented,
                             "Q4_0 forward_token execution is not yet wired in TransformerEngine");
    }

    const uint32_t hidden_dim = config_.hidden_dim;
    const uint32_t q_dim = config_.q_dim();
    const uint32_t kv_dim = config_.kv_dim();
    const uint32_t intermediate_dim = config_.intermediate_dim;
    const float eps = config_.rms_norm_eps;

    const float* embed_table = bindings_.embed_tokens.data_f32();
    if (embed_table == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "embed_tokens is not FP32");
    }

    // Step 1: Token embedding lookup: copy row `token_id` from model.embed_tokens.weight
    const float* embed_row = embed_table + static_cast<size_t>(token_id) * hidden_dim;
    std::memcpy(x_.data(), embed_row, static_cast<size_t>(hidden_dim) * sizeof(float));

    // Step 2: Loop over all decoder layers
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        const LayerWeightBindings& lw = bindings_.layers[layer_idx];

        // 2a. Pre-Attention RMSNorm: u = RMSNorm(x, w_input_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.input_layernorm.data_f32(), hidden_dim, eps);

        // 2b. Q/K/V Projections: q = W_q u, k = W_k u, v = W_v u
        scalar::matmul_f32(q_.data(), lw.q_proj.data_f32(), norm_buf_.data(), q_dim, hidden_dim);
        scalar::matmul_f32(k_.data(), lw.k_proj.data_f32(), norm_buf_.data(), kv_dim, hidden_dim);
        scalar::matmul_f32(v_.data(), lw.v_proj.data_f32(), norm_buf_.data(), kv_dim, hidden_dim);

        // 2c. RoPE for Q and K
        scalar::rope_inplace_f32(q_.data(), config_.num_query_heads, config_.head_dim, pos, config_.rope_theta);
        scalar::rope_inplace_f32(k_.data(), config_.num_kv_heads, config_.head_dim, pos, config_.rope_theta);

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
        scalar::matmul_f32(proj_out_.data(), lw.o_proj.data_f32(), attn_out_.data(), hidden_dim, q_dim);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            x_[i] += proj_out_[i];
        }

        // 2g. Post-Attention RMSNorm: m = RMSNorm(x, w_post_attention_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.post_attention_layernorm.data_f32(), hidden_dim, eps);

        // 2h. SwiGLU Feed-Forward Network and Second Residual:
        // g = W_gate m, p = W_up m, h = SiLU(g) * p, x <- x + W_down h
        scalar::matmul_f32(gate_.data(), lw.gate_proj.data_f32(), norm_buf_.data(), intermediate_dim, hidden_dim);
        scalar::matmul_f32(up_.data(), lw.up_proj.data_f32(), norm_buf_.data(), intermediate_dim, hidden_dim);
        scalar::swiglu_f32(ffn_act_.data(), gate_.data(), up_.data(), intermediate_dim);
        scalar::matmul_f32(proj_out_.data(), lw.down_proj.data_f32(), ffn_act_.data(), hidden_dim, intermediate_dim);
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
    scalar::matmul_f32(out_logits, bindings_.lm_head.data_f32(), norm_buf_.data(), config_.vocab_size, hidden_dim);

    return Status::Ok();
}

}  // namespace inference_engine
