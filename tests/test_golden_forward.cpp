#include <iostream>
#include <vector>

#include "inference_engine/sampler.hpp"
#include "inference_engine/transformer.hpp"

namespace inference_engine::testing {

bool run_golden_forward_tests() {
    // Verify the autoregressive position indexing contract on a 12-token prompt:
    // - Prompt tokens occupy positions 0..11.
    // - Final prompt logits (at pos 11) predict the first generated token.
    // - First generated token is processed at position 12, then 13, etc.
    ModelConfig cfg{};
    cfg.num_layers = 2;
    cfg.hidden_dim = 64;
    cfg.intermediate_dim = 128;
    cfg.num_query_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.vocab_size = 64;
    cfg.max_seq_len = 32;

    TransformerEngine engine;
    if (!engine.init(cfg).ok()) {
        std::cerr << "TransformerEngine::init failed\n";
        return false;
    }

    const std::vector<int32_t> prompt_tokens = {1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<float> logits(cfg.vocab_size, 0.0f);

    for (uint32_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        if (!engine.forward_token(prompt_tokens[pos], pos, logits.data()).ok()) {
            std::cerr << "Prefill forward_token failed at pos " << pos << "\n";
            return false;
        }
        if (engine.kv_cache().seq_len() != pos + 1) {
            std::cerr << "Expected committed seq_len to equal " << (pos + 1) << "\n";
            return false;
        }
    }

    // After 12-token prompt (positions 0..11), seq_len is 12.
    // First generated token is predicted from position 11's logits, then fed at position 12.
    const int32_t first_gen_token = Sampler::sample_greedy(logits.data(), logits.size());
    const uint32_t decode_pos_0 = static_cast<uint32_t>(prompt_tokens.size());  // 12
    if (decode_pos_0 != 12 || engine.kv_cache().seq_len() != 12) {
        std::cerr << "Autoregressive decode position contract mismatch\n";
        return false;
    }

    if (!engine.forward_token(first_gen_token, decode_pos_0, logits.data()).ok()) {
        std::cerr << "Decode forward_token failed at pos 12\n";
        return false;
    }
    if (engine.kv_cache().seq_len() != 13) {
        std::cerr << "Expected committed seq_len to equal 13 after first decode step\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
