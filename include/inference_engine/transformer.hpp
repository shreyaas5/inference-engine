#pragma once

#include <cstdint>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"
#include "inference_engine/kv_cache.hpp"
#include "inference_engine/loader.hpp"

namespace inference_engine {

// Autoregressive Decoding Position Indexing Contract:
// Suppose a prompt has N = 12 tokens:
// - Prefill processes prompt tokens at positions 0, 1, ..., 11.
// - After each position's forward pass across all 22 layers, `kv_cache.commit_token()`
//   advances the committed sequence length.
// - The logits produced by the final prompt token at position 11 predict the first
//   generated token (token #12, 0-indexed).
// - That first generated token is subsequently fed into `forward_token(token_id, 12, ...)`
//   at position 12, and subsequent generated tokens are processed at positions 13, 14, etc.
class TransformerEngine {
public:
    TransformerEngine() = default;

    Status init(const ModelConfig& config,
                const ModelLoader* loader = nullptr,
                SimdBackend backend = SimdBackend::Scalar,
                uint32_t num_threads = 1);

    bool initialized() const noexcept { return initialized_; }
    const ModelConfig& config() const noexcept { return config_; }
    KvCache& kv_cache() noexcept { return kv_cache_; }
    const KvCache& kv_cache() const noexcept { return kv_cache_; }
    const TransformerWeightBindings& bindings() const noexcept { return bindings_; }

    // Runs a single-token forward pass at explicit sequence position `pos`.
    // Writes vocabulary logits of size `config().vocab_size` into `out_logits`.
    Status forward_token(int32_t token_id, uint32_t pos, float* out_logits);

private:
    bool initialized_ = false;
    ModelConfig config_{};
    const ModelLoader* loader_ = nullptr;
    SimdBackend backend_ = SimdBackend::Scalar;
    KvCache kv_cache_{};
    TransformerWeightBindings bindings_{};

    // Pre-allocated activation and scratch buffers
    std::vector<float> x_{};
    std::vector<float> norm_buf_{};
    std::vector<float> q_{};
    std::vector<float> k_{};
    std::vector<float> v_{};
    std::vector<float> attn_out_{};
    std::vector<float> proj_out_{};
    std::vector<float> gate_{};
    std::vector<float> up_{};
    std::vector<float> ffn_act_{};
    std::vector<float> scores_{};
};

}  // namespace inference_engine
