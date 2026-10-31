#include "inference_engine/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace inference_engine {

int32_t Sampler::sample_greedy(const float* logits, size_t vocab_size) noexcept {
    if (logits == nullptr || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return -1;
    }

    size_t best_idx = vocab_size;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(logits[i]) || logits[i] == -std::numeric_limits<float>::infinity()) {
            continue;
        }
        if (best_idx == vocab_size || logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx == vocab_size ? -1 : static_cast<int32_t>(best_idx);
}

int32_t Sampler::sample(const float* logits, size_t vocab_size) {
    if (logits == nullptr || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        !std::isfinite(config_.temperature) || config_.temperature < 0.0f ||
        !std::isfinite(config_.top_p) || config_.top_p <= 0.0f || config_.top_p > 1.0f) {
        return -1;
    }
    if (config_.temperature == 0.0f) {
        return sample_greedy(logits, vocab_size);
    }

    size_t positive_infinities = 0;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (logits[i] == std::numeric_limits<float>::infinity()) {
            ++positive_infinities;
        }
    }
    if (positive_infinities != 0) {
        const size_t selected = static_cast<size_t>(
            uniform_random() * static_cast<double>(positive_infinities));
        size_t seen = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] == std::numeric_limits<float>::infinity() && seen++ == selected) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        if (!std::isnan(logits[i]) && logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    if (!std::isfinite(max_logit)) {
        return -1;
    }

    struct Candidate {
        int32_t id;
        double weight;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(logits[i])) {
            continue;
        }
        const double scaled =
            (static_cast<double>(logits[i]) - static_cast<double>(max_logit)) /
            static_cast<double>(config_.temperature);
        const double weight = std::exp(scaled);
        if (weight > 0.0) {
            candidates.push_back(Candidate{static_cast<int32_t>(i), weight});
        }
    }
    if (candidates.empty()) {
        return -1;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                  if (lhs.weight != rhs.weight) {
                      return lhs.weight > rhs.weight;
                  }
                  return lhs.id < rhs.id;
              });

    double total_weight = 0.0;
    for (const Candidate& candidate : candidates) {
        total_weight += candidate.weight;
    }
    const double nucleus_limit = static_cast<double>(config_.top_p) * total_weight;
    size_t nucleus_size = 0;
    double nucleus_weight = 0.0;
    do {
        nucleus_weight += candidates[nucleus_size].weight;
        ++nucleus_size;
    } while (nucleus_size < candidates.size() && nucleus_weight < nucleus_limit);

    const double target = uniform_random() * nucleus_weight;
    double cumulative = 0.0;
    for (size_t i = 0; i < nucleus_size; ++i) {
        cumulative += candidates[i].weight;
        if (target < cumulative) {
            return candidates[i].id;
        }
    }
    return candidates[nucleus_size - 1].id;
}

uint64_t Sampler::next_random() noexcept {
    uint64_t value = rng_state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    rng_state_ = value;
    return value * 0x2545F4914F6CDD1DULL;
}

double Sampler::uniform_random() noexcept {
    return static_cast<double>(next_random() >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace inference_engine
