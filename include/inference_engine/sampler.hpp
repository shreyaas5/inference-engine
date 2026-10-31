#pragma once

#include <cstddef>
#include <cstdint>

namespace inference_engine {

struct SamplerConfig {
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint64_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {}) noexcept : config_(config), rng_state_(config.seed) {}

    // Deterministic greedy argmax over `vocab_size` logits.
    // Ties are broken by selecting the lowest token index.
    static int32_t sample_greedy(const float* logits, size_t vocab_size) noexcept;

    // Samples next token according to configured temperature and top-p (falls back to greedy when temperature <= 0).
    int32_t sample(const float* logits, size_t vocab_size);

    uint64_t rng_state() const noexcept { return rng_state_; }

private:
    SamplerConfig config_{};
    uint64_t rng_state_ = 42;
};

}  // namespace inference_engine
