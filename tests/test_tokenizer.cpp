#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "inference_engine/tokenizer.hpp"

namespace inference_engine::testing {

bool run_tokenizer_tests() {
    // TinyLlama's tokenizer_config.json template renders only system/user/assistant roles.
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
    if (!Tokenizer::format_chat_prompt({}, true).empty()) {
        std::cerr << "Empty chat unexpectedly produced a generation prompt\n";
        return false;
    }

    // Encoding must fail explicitly until a real SentencePiece model is loaded.
    Tokenizer tok;
    std::vector<int32_t> ids;
    if (tok.encode("Hi", true, true, &ids).ok()) {
        std::cerr << "Tokenizer encoded input before loading SentencePiece\n";
        return false;
    }
    std::string model_path_str;
    const char* env_path = std::getenv("INFERENCE_ENGINE_TOKENIZER_MODEL");
    if (env_path != nullptr && env_path[0] != '\0') {
        model_path_str = env_path;
    } else {
        const std::string candidates[] = {
            "build/tokenizer/tokenizer.model",
            "tokenizer/tokenizer.model"
        };
        for (const auto& c : candidates) {
            FILE* f = std::fopen(c.c_str(), "rb");
            if (f != nullptr) {
                std::fclose(f);
                model_path_str = c;
                break;
            }
        }
    }
    if (model_path_str.empty()) {
        std::cerr << "INFERENCE_ENGINE_TOKENIZER_MODEL must point to TinyLlama tokenizer.model\n";
        return false;
    }
    const char* model_path = model_path_str.c_str();
    if (!tok.load(model_path).ok() || !tok.is_loaded()) {
        std::cerr << "Tokenizer failed to load TinyLlama SentencePiece model\n";
        return false;
    }

    if (!tok.encode("Hi", true, true, &ids).ok() ||
        ids.size() < 2 || ids.front() != kBosTokenId || ids.back() != kEosTokenId) {
        std::cerr << "Tokenizer BOS/EOS token placement failed\n";
        return false;
    }
    std::string decoded;
    if (!tok.decode(ids, &decoded).ok() || decoded != "<s>Hi</s>") {
        std::cerr << "SentencePiece encode/decode round-trip failed\n";
        return false;
    }

    if (!tok.encode("<s>hello</s>", false, false, &ids).ok() ||
        ids.size() < 3 || ids.front() != kBosTokenId || ids.back() != kEosTokenId) {
        std::cerr << "Tokenizer special-token mapping failed\n";
        return false;
    }
    if (tok.decode({-1}, &decoded).ok()) {
        std::cerr << "Tokenizer accepted an out-of-range token ID\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
