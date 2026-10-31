#include "inference_engine/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "inference_engine/kernels.hpp"

namespace inference_engine {

namespace {

Status bind_tensor(const ModelLoader* loader,
                   const std::string& name,
                   DataType expected_dtype,
                   uint32_t expected_ndim,
                   const std::vector<uint64_t>& expected_shape,
                   const float** out_ptr) {
    if (loader == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "ModelLoader is null");
    }
    const TensorView* view = loader->find_tensor(name);
    if (view == nullptr || view->empty()) {
        return Status::Error(StatusCode::InvalidArgument, "Required tensor not found in loader: " + name);
    }
    const TensorDesc& desc = view->desc();
    if (desc.dtype != expected_dtype) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tensor " + name + " dtype mismatch: expected " +
                             dtype_name(expected_dtype) + ", got " +
                             dtype_name(desc.dtype));
    }
    if (desc.ndim != expected_ndim) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tensor " + name + " ndim mismatch: expected " +
                             std::to_string(expected_ndim) + ", got " +
                             std::to_string(desc.ndim));
    }
    for (uint32_t d = 0; d < expected_ndim; ++d) {
        if (desc.shape[d] != expected_shape[d]) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "Tensor " + name + " shape[" + std::to_string(d) +
                                 "] mismatch: expected " + std::to_string(expected_shape[d]) +
                                 ", got " + std::to_string(desc.shape[d]));
        }
    }
    for (size_t d = expected_ndim; d < kMaxTensorDims; ++d) {
        if (desc.shape[d] != 0) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "Tensor " + name + " trailing dim " + std::to_string(d) +
                                 " must be 0");
        }
    }
    uint64_t expected_bytes = 0;
    if (!desc.compute_expected_bytes(&expected_bytes)) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Overflow computing expected bytes for " + name);
    }
    if (desc.payload_nbytes != expected_bytes) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tensor " + name + " payload_nbytes mismatch: expected " +
                             std::to_string(expected_bytes) + ", got " +
                             std::to_string(desc.payload_nbytes));
    }
    const float* ptr = view->data_f32();
    if (ptr == nullptr) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tensor " + name + " has null FP32 data pointer");
    }
    *out_ptr = ptr;
    return Status::Ok();
}

}  // namespace

Status TransformerEngine::init(const ModelConfig& config,
                               const ModelLoader* loader,
                               SimdBackend backend,
                               uint32_t /*num_threads*/) {
    initialized_ = false;

    Status st = config.validate();
    if (!st.ok()) {
        return st;
    }
    if (config.weight_dtype != DataType::FP32) {
        return Status::Error(StatusCode::InvalidArgument,
                             "M1 TransformerEngine requires FP32 weights");
    }
    if (loader == nullptr) {
        return Status::Error(StatusCode::InvalidArgument,
                             "ModelLoader must not be null for weight-bound TransformerEngine");
    }
    if (!loader->is_loaded()) {
        return Status::Error(StatusCode::InvalidArgument,
                             "ModelLoader is not loaded");
    }

    Status kv_st = kv_cache_.init(config);
    if (!kv_st.ok()) {
        return kv_st;
    }

    // 1. Bind model-level tensors: embed_tokens, norm, and lm_head
    Status bind_st = bind_tensor(loader,
                                 "model.embed_tokens.weight",
                                 DataType::FP32,
                                 2,
                                 {config.vocab_size, config.hidden_dim},
                                 &embed_tokens_);
    if (!bind_st.ok()) {
        return bind_st;
    }

    bind_st = bind_tensor(loader,
                          "model.norm.weight",
                          DataType::FP32,
                          1,
                          {config.hidden_dim},
                          &final_norm_);
    if (!bind_st.ok()) {
        return bind_st;
    }

    bind_st = bind_tensor(loader,
                          "lm_head.weight",
                          DataType::FP32,
                          2,
                          {config.vocab_size, config.hidden_dim},
                          &lm_head_);
    if (!bind_st.ok()) {
        return bind_st;
    }

    // 2. Bind per-layer decoder tensors across all config.num_layers
    layer_weights_.resize(config.num_layers);
    for (uint32_t l = 0; l < config.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        LayerWeights& lw = layer_weights_[l];

        bind_st = bind_tensor(loader,
                              prefix + "input_layernorm.weight",
                              DataType::FP32,
                              1,
                              {config.hidden_dim},
                              &lw.input_layernorm);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "self_attn.q_proj.weight",
                              DataType::FP32,
                              2,
                              {config.q_dim(), config.hidden_dim},
                              &lw.q_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "self_attn.k_proj.weight",
                              DataType::FP32,
                              2,
                              {config.kv_dim(), config.hidden_dim},
                              &lw.k_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "self_attn.v_proj.weight",
                              DataType::FP32,
                              2,
                              {config.kv_dim(), config.hidden_dim},
                              &lw.v_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "self_attn.o_proj.weight",
                              DataType::FP32,
                              2,
                              {config.hidden_dim, config.q_dim()},
                              &lw.o_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "post_attention_layernorm.weight",
                              DataType::FP32,
                              1,
                              {config.hidden_dim},
                              &lw.post_attention_layernorm);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "mlp.gate_proj.weight",
                              DataType::FP32,
                              2,
                              {config.intermediate_dim, config.hidden_dim},
                              &lw.gate_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "mlp.up_proj.weight",
                              DataType::FP32,
                              2,
                              {config.intermediate_dim, config.hidden_dim},
                              &lw.up_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }

        bind_st = bind_tensor(loader,
                              prefix + "mlp.down_proj.weight",
                              DataType::FP32,
                              2,
                              {config.hidden_dim, config.intermediate_dim},
                              &lw.down_proj);
        if (!bind_st.ok()) {
            return bind_st;
        }
    }

    // 3. Pre-allocate intermediate activations and scratch buffers
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
    if (loader_ == nullptr || embed_tokens_ == nullptr) {
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

    // Step 1: Token embedding lookup: copy row `token_id` from model.embed_tokens.weight
    const float* embed_row = embed_tokens_ + static_cast<size_t>(token_id) * hidden_dim;
    std::memcpy(x_.data(), embed_row, static_cast<size_t>(hidden_dim) * sizeof(float));

    // Step 2: Loop over all decoder layers
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        const LayerWeights& lw = layer_weights_[layer_idx];

        // 2a. Pre-Attention RMSNorm: u = RMSNorm(x, w_input_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.input_layernorm, hidden_dim, eps);

        // 2b. Q/K/V Projections: q = W_q u, k = W_k u, v = W_v u
        scalar::matmul_f32(q_.data(), lw.q_proj, norm_buf_.data(), q_dim, hidden_dim);
        scalar::matmul_f32(k_.data(), lw.k_proj, norm_buf_.data(), kv_dim, hidden_dim);
        scalar::matmul_f32(v_.data(), lw.v_proj, norm_buf_.data(), kv_dim, hidden_dim);

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
        scalar::matmul_f32(proj_out_.data(), lw.o_proj, attn_out_.data(), hidden_dim, q_dim);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            x_[i] += proj_out_[i];
        }

        // 2g. Post-Attention RMSNorm: m = RMSNorm(x, w_post_attention_layernorm, eps)
        scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.post_attention_layernorm, hidden_dim, eps);

        // 2h. SwiGLU Feed-Forward Network and Second Residual:
        // g = W_gate m, p = W_up m, h = SiLU(g) * p, x <- x + W_down h
        scalar::matmul_f32(gate_.data(), lw.gate_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
        scalar::matmul_f32(up_.data(), lw.up_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
        scalar::swiglu_f32(ffn_act_.data(), gate_.data(), up_.data(), intermediate_dim);
        scalar::matmul_f32(proj_out_.data(), lw.down_proj, ffn_act_.data(), hidden_dim, intermediate_dim);
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
    scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), final_norm_, hidden_dim, eps);
    scalar::matmul_f32(out_logits, lm_head_, norm_buf_.data(), config_.vocab_size, hidden_dim);

    return Status::Ok();
}

}  // namespace inference_engine
