#include "inference_engine/sampler.hpp"

namespace inference_engine {

int32_t Sampler::sample_greedy(const float* logits, size_t vocab_size) noexcept {
    if (logits == nullptr || vocab_size == 0) {
        return -1;
    }
    size_t best_idx = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return static_cast<int32_t>(best_idx);
}

int32_t Sampler::sample(const float* logits, size_t vocab_size) {
    if (config_.temperature <= 0.0f) {
        return sample_greedy(logits, vocab_size);
    }
    // M0 foundation: defaults to deterministic greedy argmax until stochastic top-p
    // sampling is enabled in M4.
    return sample_greedy(logits, vocab_size);
}

}  // namespace inference_engine
