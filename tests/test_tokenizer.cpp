#include <iostream>
#include <string>
#include <vector>

#include "inference_engine/sampler.hpp"
#include "inference_engine/tokenizer.hpp"

namespace inference_engine::testing {

bool run_tokenizer_tests() {
    // 1. Verify TinyLlama Zephyr chat template formatting contract.
    const std::vector<ChatMessage> messages = {
        {"system", "You are a helpful assistant."},
        {"user", "Hello!"},
    };
    const std::string formatted = Tokenizer::format_chat_prompt(messages, true);
    const std::string expected =
        "<|system|>\nYou are a helpful assistant.</s>\n"
        "<|user|>\nHello!</s>\n"
        "<|assistant|>\n";
    if (formatted != expected) {
        std::cerr << "TinyLlama chat template formatting mismatch\n";
        return false;
    }

    // 2. Verify BOS/EOS placement and round-trip scaffolding in Tokenizer.
    Tokenizer tok;
    std::vector<int32_t> ids;
    if (!tok.encode("Hi", true, true, &ids).ok()) {
        std::cerr << "Tokenizer::encode failed\n";
        return false;
    }
    if (ids.size() < 2 || ids.front() != kBosTokenId || ids.back() != kEosTokenId) {
        std::cerr << "Tokenizer BOS/EOS token placement failed\n";
        return false;
    }

    std::string decoded;
    if (!tok.decode(ids, &decoded).ok() || decoded != "Hi") {
        std::cerr << "Tokenizer::decode round-trip failed\n";
        return false;
    }

    // 3. Verify deterministic greedy sampler.
    const std::vector<float> logits = {-1.0f, 3.5f, 3.5f, 2.0f};
    if (Sampler::sample_greedy(logits.data(), logits.size()) != 1) {
        std::cerr << "Sampler::sample_greedy failed tie-breaking or argmax contract\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
