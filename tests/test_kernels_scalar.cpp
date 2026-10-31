#include <cmath>
#include <iostream>
#include <vector>

#include "inference_engine/kernels.hpp"

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

    return true;
}

}  // namespace inference_engine::testing
