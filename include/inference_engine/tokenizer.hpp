#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "inference_engine/common.hpp"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace inference_engine {

inline constexpr int32_t kUnkTokenId = 0;
inline constexpr int32_t kBosTokenId = 1;
inline constexpr int32_t kEosTokenId = 2;

struct ChatMessage {
    std::string role{};
    std::string content{};
};

class Tokenizer {
public:
    Tokenizer() = default;

    Status load(const std::string& model_path);
    bool is_loaded() const noexcept { return loaded_; }

    Status encode(const std::string& text,
                  bool add_bos,
                  bool add_eos,
                  std::vector<int32_t>* out_ids) const;

    Status decode(const std::vector<int32_t>& token_ids, std::string* out_text) const;

    // Formats messages using the TinyLlama-1.1B-Chat-v1.0 Zephyr chat template:
    // <|system|>\n{system_message}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n
    static std::string format_chat_prompt(const std::vector<ChatMessage>& messages,
                                          bool add_generation_prompt = true);

private:
    bool loaded_ = false;
    std::string model_path_{};
    std::shared_ptr<sentencepiece::SentencePieceProcessor> processor_{};
};

}  // namespace inference_engine
