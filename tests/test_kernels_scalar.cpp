#include <cmath>
#include <iostream>
#include <vector>

#include "inference_engine/kernels.hpp"
#include "inference_engine/kv_cache.hpp"

namespace inference_engine::testing {

bool run_kernels_scalar_tests() {
    // 1. RMSNorm test: constant vector [2, 2, 2, 2] with unit weights -> [1, 1, 1, 1].
    const std::vector<float> x = {2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> rms_out(4, 0.0f);
    scalar::rmsnorm_f32(rms_out.data(), x.data(), w.data(), 4, 1e-6f);
    for (float v : rms_out) {
        if (std::fabs(v - 1.0f) > 1e-5f) {
            std::cerr << "Scalar RMSNorm failed\n";
            return false;
        }
    }

    // 2. RoPE test at pos = 0 (identity rotation) and pos = 3 (norm preservation).
    std::vector<float> qk_pos0 = {1.0f, 2.0f, 3.0f, 4.0f};
    scalar::rope_inplace_f32(qk_pos0.data(), 1, 4, 0, 10000.0f);
    if (std::fabs(qk_pos0[0] - 1.0f) > 1e-6f || std::fabs(qk_pos0[2] - 3.0f) > 1e-6f) {
        std::cerr << "RoPE at pos=0 must be identity\n";
        return false;
    }

    std::vector<float> qk_pos3 = {1.5f, -0.5f, 2.0f, 1.0f};
    const float norm_before = scalar::dot_f32(qk_pos3.data(), qk_pos3.data(), 4);
    scalar::rope_inplace_f32(qk_pos3.data(), 1, 4, 3, 10000.0f);
    const float norm_after = scalar::dot_f32(qk_pos3.data(), qk_pos3.data(), 4);
    if (std::fabs(norm_before - norm_after) > 1e-5f) {
        std::cerr << "RoPE must preserve Euclidean norm\n";
        return false;
    }

    // 3. Softmax test: probabilities sum to 1 and preserve monotonicity.
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    scalar::softmax_f32(logits.data(), logits.size());
    const float sum = logits[0] + logits[1] + logits[2];
    if (std::fabs(sum - 1.0f) > 1e-6f || !(logits[0] < logits[1] && logits[1] < logits[2])) {
        std::cerr << "Scalar softmax failed\n";
        return false;
    }

    // 4. SwiGLU test: gate=0 -> output=0.
    const std::vector<float> gate = {0.0f, 1.0f};
    const std::vector<float> up = {5.0f, 2.0f};
    std::vector<float> swiglu_out(2, 0.0f);
    scalar::swiglu_f32(swiglu_out.data(), gate.data(), up.data(), 2);
    const float expected_1 = (1.0f / (1.0f + std::exp(-1.0f))) * 2.0f;
    if (std::fabs(swiglu_out[0]) > 1e-6f || std::fabs(swiglu_out[1] - expected_1) > 1e-5f) {
        std::cerr << "Scalar SwiGLU failed\n";
        return false;
    }

    // 5. GQA Attention test with 4 Q heads, 2 KV heads (group size 2), head dim 4.
    ModelConfig gqa_cfg{};
    gqa_cfg.num_layers = 1;
    gqa_cfg.hidden_dim = 16;
    gqa_cfg.intermediate_dim = 32;
    gqa_cfg.num_query_heads = 4;
    gqa_cfg.num_kv_heads = 2;
    gqa_cfg.head_dim = 4;
    gqa_cfg.vocab_size = 16;
    gqa_cfg.max_seq_len = 8;

    KvCache kv_cache;
    if (!kv_cache.init(gqa_cfg).ok()) {
        std::cerr << "Failed to init kv_cache for GQA test\n";
        return false;
    }

    // Position 0:
    // KV Head 0: K = [1, 0, 0, 0], V = [2, 3, 4, 5]
    // KV Head 1: K = [0, 1, 0, 0], V = [6, 7, 8, 9]
    const std::vector<float> k0 = {1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<float> v0 = {2.0f, 3.0f, 4.0f, 5.0f,  6.0f, 7.0f, 8.0f, 9.0f};
    if (!kv_cache.append(0, 0, k0.data(), v0.data()).ok() || !kv_cache.commit_token().ok()) {
        std::cerr << "Failed to append pos 0 in GQA test\n";
        return false;
    }

    // Position 1:
    // KV Head 0: K = [0, 1, 0, 0], V = [10, 11, 12, 13]
    // KV Head 1: K = [1, 0, 0, 0], V = [14, 15, 16, 17]
    const std::vector<float> k1 = {0.0f, 1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> v1 = {10.0f, 11.0f, 12.0f, 13.0f,  14.0f, 15.0f, 16.0f, 17.0f};
    if (!kv_cache.append(0, 1, k1.data(), v1.data()).ok() || !kv_cache.commit_token().ok()) {
        std::cerr << "Failed to append pos 1 in GQA test\n";
        return false;
    }

    // Query vectors at pos = 1 (attending over positions 0 and 1):
    // Q head 0: [1, 0, 0, 0] (group 0 -> KV head 0)
    // Q head 1: [0, 1, 0, 0] (group 0 -> KV head 0)
    // Q head 2: [0, 1, 0, 0] (group 1 -> KV head 1)
    // Q head 3: [1, 0, 0, 0] (group 1 -> KV head 1)
    const std::vector<float> q = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
    };
    std::vector<float> attn_out(16, 0.0f);
    std::vector<float> scratch(8, 0.0f);

    scalar::gqa_attention_f32(attn_out.data(), q.data(), kv_cache, 0, 1, 4, 2, 4, scratch.data());

    // For Q head 0 ([1, 0, 0, 0]) with KV head 0:
    // t=0: k0_h0=[1,0,0,0], dot = 1.0, scale = 1/sqrt(4) = 0.5 -> s0 = 0.5
    // t=1: k1_h0=[0,1,0,0], dot = 0.0, s1 = 0.0
    // w0 = exp(0.5) / (exp(0.5) + exp(0)) = 1.64872127 / 2.64872127 = 0.6224593f
    // w1 = 1 - w0 = 0.3775407f
    // out = w0 * [2, 3, 4, 5] + w1 * [10, 11, 12, 13]
    const float exp_0 = std::exp(0.5f);
    const float w0 = exp_0 / (exp_0 + 1.0f);
    const float w1 = 1.0f - w0;
    const float expected_h0_d0 = w0 * 2.0f + w1 * 10.0f;
    if (std::fabs(attn_out[0] - expected_h0_d0) > 1e-5f) {
        std::cerr << "GQA attention value mismatch for head 0, dim 0\n";
        return false;
    }

    // For Q head 2 ([0, 1, 0, 0]) with KV head 1:
    // t=0: k0_h1=[0,1,0,0], dot = 1.0 -> s0 = 0.5
    // t=1: k1_h1=[1,0,0,0], dot = 0.0 -> s1 = 0.0
    // out = w0 * [6, 7, 8, 9] + w1 * [14, 15, 16, 17]
    const float expected_h2_d0 = w0 * 6.0f + w1 * 14.0f;
    if (std::fabs(attn_out[8] - expected_h2_d0) > 1e-5f) {
        std::cerr << "GQA attention value mismatch for head 2, dim 0\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
