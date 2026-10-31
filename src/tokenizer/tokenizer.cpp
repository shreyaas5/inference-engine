#include "inference_engine/tokenizer.hpp"

#include <array>
#include <string>
#include <sstream>
#include <vector>

#include "sentencepiece_processor.h"

namespace inference_engine {
namespace {

struct SpecialToken {
    const char* text;
    int32_t id;
};

constexpr std::array<SpecialToken, 3> kSpecialTokens{{
    {"<unk>", kUnkTokenId},
    {"<s>", kBosTokenId},
    {"</s>", kEosTokenId},
}};

Status encode_text(const sentencepiece::SentencePieceProcessor& processor,
                   const std::string& text,
                   std::vector<int32_t>* out_ids) {
    if (text.empty()) {
        return Status::Ok();
    }

    std::vector<int> encoded;
    const auto status = processor.Encode(text, &encoded);
    if (!status.ok()) {
        return Status::Error(StatusCode::InternalError,
                             "SentencePiece encode failed: " + status.ToString());
    }
    out_ids->insert(out_ids->end(), encoded.begin(), encoded.end());
    return Status::Ok();
}

}  // namespace

Status Tokenizer::load(const std::string& model_path) {
    if (model_path.empty()) {
        return Status::Error(StatusCode::InvalidArgument, "Tokenizer model_path must not be empty");
    }

    loaded_ = false;
    model_path_.clear();
    processor_.reset();

    auto processor = std::make_shared<sentencepiece::SentencePieceProcessor>();
    const auto status = processor->Load(model_path);
    if (!status.ok()) {
        return Status::Error(StatusCode::IoError,
                             "Failed to load SentencePiece model '" + model_path +
                                 "': " + status.ToString());
    }
    if (processor->GetPieceSize() != 32000 ||
        processor->unk_id() != kUnkTokenId ||
        processor->bos_id() != kBosTokenId ||
        processor->eos_id() != kEosTokenId) {
        return Status::Error(StatusCode::UnsupportedFormat,
                             "SentencePiece model does not match TinyLlama's "
                             "32000-piece vocabulary and <unk>/<s></s> IDs 0/1/2");
    }

    processor_ = std::move(processor);
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
    if (!loaded_ || processor_ == nullptr) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tokenizer must be loaded before encoding");
    }
    if (add_bos) {
        out_ids->push_back(kBosTokenId);
    }

    size_t segment_start = 0;
    size_t cursor = 0;
    while (cursor < text.size()) {
        const SpecialToken* special = nullptr;
        for (const SpecialToken& candidate : kSpecialTokens) {
            const size_t token_length = std::char_traits<char>::length(candidate.text);
            if (text.compare(cursor, token_length, candidate.text) == 0) {
                special = &candidate;
                break;
            }
        }
        if (special == nullptr) {
            ++cursor;
            continue;
        }

        Status status = encode_text(*processor_, text.substr(segment_start, cursor - segment_start),
                                    out_ids);
        if (!status.ok()) {
            out_ids->clear();
            return status;
        }
        out_ids->push_back(special->id);
        cursor += std::char_traits<char>::length(special->text);
        segment_start = cursor;
    }

    Status status = encode_text(*processor_, text.substr(segment_start), out_ids);
    if (!status.ok()) {
        out_ids->clear();
        return status;
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
    if (!loaded_ || processor_ == nullptr) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Tokenizer must be loaded before decoding");
    }

    std::vector<int> segment;
    segment.reserve(token_ids.size());
    const auto decode_segment = [&]() -> Status {
        if (segment.empty()) {
            return Status::Ok();
        }
        std::string decoded;
        const auto status = processor_->Decode(segment, &decoded);
        if (!status.ok()) {
            return Status::Error(StatusCode::InternalError,
                                 "SentencePiece decode failed: " + status.ToString());
        }
        out_text->append(decoded);
        segment.clear();
        return Status::Ok();
    };

    for (int32_t id : token_ids) {
        if (id < 0 || id >= processor_->GetPieceSize()) {
            out_text->clear();
            return Status::Error(StatusCode::OutOfBounds,
                                 "Token ID is outside the SentencePiece vocabulary");
        }
        const SpecialToken* special = nullptr;
        for (const SpecialToken& candidate : kSpecialTokens) {
            if (id == candidate.id) {
                special = &candidate;
                break;
            }
        }
        if (special == nullptr) {
            segment.push_back(id);
            continue;
        }

        const Status status = decode_segment();
        if (!status.ok()) {
            out_text->clear();
            return status;
        }
        out_text->append(special->text);
    }
    const Status status = decode_segment();
    if (!status.ok()) {
        out_text->clear();
        return status;
    }
    return Status::Ok();
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages,
                                          bool add_generation_prompt) {
    std::ostringstream oss;
    for (size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& msg = messages[i];
        if (msg.role == "system" || msg.role == "user" || msg.role == "assistant") {
            oss << "<|" << msg.role << "|>\n" << msg.content << "</s>\n";
        }
        if (i + 1 == messages.size() && add_generation_prompt) {
            oss << "<|assistant|>";
        }
    }
    if (add_generation_prompt && !messages.empty()) {
        oss << "\n";
    }
    return oss.str();
}

}  // namespace inference_engine
