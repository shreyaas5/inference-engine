#include "inference_engine/transformer.hpp"

#include <algorithm>

namespace inference_engine {

Status TransformerEngine::init(const ModelConfig& config,
                               const ModelLoader* loader,
                               SimdBackend backend,
                               uint32_t /*num_threads*/) {
    Status st = config.validate();
    if (!st.ok()) {
        return st;
    }
    Status kv_st = kv_cache_.init(config);
    if (!kv_st.ok()) {
        return kv_st;
    }
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
    if (out_logits == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_logits must not be null");
    }
    if (token_id < 0 || static_cast<uint32_t>(token_id) >= config_.vocab_size) {
        return Status::Error(StatusCode::OutOfBounds, "token_id out of vocabulary range");
    }
    if (pos >= config_.max_seq_len) {
        return Status::Error(StatusCode::OutOfBounds, "pos exceeds max_seq_len");
    }

    // M0 foundation scaffolding: record synthetic KV state across all layers for position `pos`
    // and commit the token position once all layers have been visited.
    std::vector<float> zero_kv(config_.kv_dim(), 0.0f);
    for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
        Status st = kv_cache_.append(layer, pos, zero_kv.data(), zero_kv.data());
        if (!st.ok()) {
            return st;
        }
    }

    if (pos == kv_cache_.seq_len()) {
        Status st = kv_cache_.commit_token();
        if (!st.ok()) {
            return st;
        }
    }

    std::fill(out_logits, out_logits + config_.vocab_size, 0.0f);
    return Status::Ok();
}

}  // namespace inference_engine
