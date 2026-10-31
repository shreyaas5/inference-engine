#include "inference_engine/tokenizer.hpp"

#include <sstream>

namespace inference_engine {

Status Tokenizer::load(const std::string& model_path) {
    if (model_path.empty()) {
        return Status::Error(StatusCode::InvalidArgument, "Tokenizer model_path must not be empty");
    }
    model_path_ = model_path;
    loaded_ = true;
    return Status::Ok();
}

Status Tokenizer::encode(const std::string& text,
                         bool add_bos,
                         bool add_eos,
                         std::vector<int32_t>* out_ids) const {
    if (out_ids == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_ids must not be null");
    }
    out_ids->clear();
    if (add_bos) {
        out_ids->push_back(kBosTokenId);
    }
    // M0 foundation scaffolding: SentencePiece C++ processor integration is scheduled for M1.
    // For synthetic tests when no SentencePiece model is linked yet, byte-level fallback IDs
    // are offset into a safe range [3..258].
    for (unsigned char c : text) {
        out_ids->push_back(static_cast<int32_t>(c) + 3);
    }
    if (add_eos) {
        out_ids->push_back(kEosTokenId);
    }
    return Status::Ok();
}

Status Tokenizer::decode(const std::vector<int32_t>& token_ids, std::string* out_text) const {
    if (out_text == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_text must not be null");
    }
    out_text->clear();
    for (int32_t id : token_ids) {
        if (id == kBosTokenId || id == kEosTokenId || id == kUnkTokenId) {
            continue;
        }
        if (id >= 3 && id < 259) {
            out_text->push_back(static_cast<char>(id - 3));
        }
    }
    return Status::Ok();
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages,
                                          bool add_generation_prompt) {
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|" << msg.role << "|>\n" << msg.content << "</s>\n";
    }
    if (add_generation_prompt) {
        oss << "<|assistant|>\n";
    }
    return oss.str();
}

}  // namespace inference_engine
