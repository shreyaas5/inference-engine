#include <iostream>
#include <limits>
#include <vector>

#include "inference_engine/sampler.hpp"

namespace inference_engine::testing {

bool run_sampler_tests() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {-1.0f, 3.5f, 3.5f, nan};

    if (Sampler::sample_greedy(logits.data(), logits.size()) != 1) {
        std::cerr << "Greedy sampling failed argmax or lowest-index tie-breaking\n";
        return false;
    }
    if (Sampler::sample_greedy(nullptr, logits.size()) != -1 ||
        Sampler::sample_greedy(&neg_inf, 1) != -1) {
        std::cerr << "Greedy sampling accepted an invalid logit vector\n";
        return false;
    }
    const float positive_infinities[] = {pos_inf, pos_inf, 1.0f};
    if (Sampler::sample_greedy(positive_infinities, 3) != 0) {
        std::cerr << "Greedy sampling did not handle positive infinity\n";
        return false;
    }

    const SamplerConfig stochastic_config{0.8f, 0.9f, 1234};
    Sampler first(stochastic_config);
    Sampler second(stochastic_config);
    for (size_t i = 0; i < 32; ++i) {
        if (first.sample(logits.data(), logits.size()) !=
            second.sample(logits.data(), logits.size())) {
            std::cerr << "Temperature/top-p sampling is not reproducible for a fixed seed\n";
            return false;
        }
    }

    const float peaked_logits[] = {10.0f, 0.0f, -1.0f};
    Sampler nucleus_only({0.8f, 0.5f, 7});
    for (size_t i = 0; i < 16; ++i) {
        if (nucleus_only.sample(peaked_logits, 3) != 0) {
            std::cerr << "Top-p sampling selected outside the 0.5 nucleus\n";
            return false;
        }
    }
    Sampler invalid_config({0.8f, 0.0f, 7});
    if (invalid_config.sample(peaked_logits, 3) != -1) {
        std::cerr << "Sampler accepted an invalid top-p value\n";
        return false;
    }
    return true;
}

}  // namespace inference_engine::testing
