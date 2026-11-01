#pragma once

#include <cstdint>
#include <string>

#include "inference_engine/common.hpp"

namespace inference_engine {

// Frozen TinyLlama-1.1B-Chat-v1.0 architectural constants.
inline constexpr uint32_t kTinyLlamaNumLayers = 22;
inline constexpr uint32_t kTinyLlamaHiddenDim = 2048;
inline constexpr uint32_t kTinyLlamaIntermediateDim = 5632;
inline constexpr uint32_t kTinyLlamaNumQueryHeads = 32;
inline constexpr uint32_t kTinyLlamaNumKvHeads = 4;
inline constexpr uint32_t kTinyLlamaHeadDim = 64;
inline constexpr uint32_t kTinyLlamaVocabSize = 32000;
inline constexpr uint32_t kTinyLlamaMaxSeqLen = 2048;
inline constexpr float kTinyLlamaRmsNormEps = 1e-5f;
inline constexpr float kTinyLlamaRopeTheta = 10000.0f;

struct ModelConfig {
    uint32_t num_layers = kTinyLlamaNumLayers;
    uint32_t hidden_dim = kTinyLlamaHiddenDim;
    uint32_t intermediate_dim = kTinyLlamaIntermediateDim;
    uint32_t num_query_heads = kTinyLlamaNumQueryHeads;
    uint32_t num_kv_heads = kTinyLlamaNumKvHeads;
    uint32_t head_dim = kTinyLlamaHeadDim;
    uint32_t vocab_size = kTinyLlamaVocabSize;
    uint32_t max_seq_len = kTinyLlamaMaxSeqLen;
    float rms_norm_eps = kTinyLlamaRmsNormEps;
    float rope_theta = kTinyLlamaRopeTheta;
    DataType weight_dtype = DataType::FP32;

    uint32_t kv_dim() const noexcept { return num_kv_heads * head_dim; }
    uint32_t q_dim() const noexcept { return num_query_heads * head_dim; }
    uint32_t gqa_group_size() const noexcept {
        return num_kv_heads > 0 ? (num_query_heads / num_kv_heads) : 0;
    }

    Status validate() const noexcept {
        if (num_layers == 0 || hidden_dim == 0 || intermediate_dim == 0 ||
            num_query_heads == 0 || num_kv_heads == 0 || head_dim == 0 ||
            vocab_size == 0 || max_seq_len == 0) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "ModelConfig dimensions must all be non-zero");
        }
        if (num_query_heads % num_kv_heads != 0) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "num_query_heads must be divisible by num_kv_heads for GQA");
        }
        if (num_query_heads * head_dim != hidden_dim) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "num_query_heads * head_dim must equal hidden_dim");
        }
        if (head_dim % 2 != 0) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "head_dim must be even for RoPE");
        }
        if (rms_norm_eps <= 0.0f || rope_theta <= 0.0f) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "rms_norm_eps and rope_theta must be positive");
        }
        return Status::Ok();
    }
};

struct RuntimeConfig {
    std::string model_path{};
    std::string tokenizer_path{};
    std::string prompt{};
    uint32_t max_new_tokens = 64;
    uint32_t num_threads = 1;
    SimdBackend backend = SimdBackend::Scalar;
    bool auto_detect_backend = true;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 42;
    bool benchmark_mode = false;
};

}  // namespace inference_engine
