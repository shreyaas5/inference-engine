#include <cmath>
#include <iostream>
#include <vector>

#include "inference_engine/kernels.hpp"

namespace inference_engine::testing {

bool run_kernels_simd_tests() {
    const size_t in_dim = 64;
    const size_t out_dim = 8;

    std::vector<float> weight(out_dim * in_dim);
    std::vector<float> x(in_dim);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    for (size_t i = 0; i < in_dim; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f;
    }

    std::vector<float> ref_out(out_dim, 0.0f);
    scalar::matmul_f32(ref_out.data(), weight.data(), x.data(), out_dim, in_dim);

    const SimdBackend best = detect_best_backend();
    ThreadPool pool(2);
    std::vector<float> simd_out(out_dim, 0.0f);
    dispatch_matmul_f32(simd_out.data(), weight.data(), x.data(), out_dim, in_dim, best, &pool);

    for (size_t r = 0; r < out_dim; ++r) {
        if (std::fabs(ref_out[r] - simd_out[r]) > 1e-5f) {
            std::cerr << "SIMD matmul_f32 mismatch at row " << r << "\n";
            return false;
        }
    }

    // Q4_0 matmul parity check across scalar and dispatched paths.
    std::vector<BlockQ4_0> q4_weights((out_dim * in_dim) / kQ4_0BlockElements);
    if (!quantize_row_q4_0(weight.data(), q4_weights.data(), weight.size()).ok()) {
        std::cerr << "Failed to quantize weights for SIMD test\n";
        return false;
    }

    std::vector<float> ref_q4_out(out_dim, 0.0f);
    std::vector<float> simd_q4_out(out_dim, 0.0f);
    scalar::matmul_q4_0_f32(ref_q4_out.data(), q4_weights.data(), x.data(), out_dim, in_dim);
    dispatch_matmul_q4_0_f32(
        simd_q4_out.data(), q4_weights.data(), x.data(), out_dim, in_dim, best, &pool);

    for (size_t r = 0; r < out_dim; ++r) {
        if (std::fabs(ref_q4_out[r] - simd_q4_out[r]) > 1e-5f) {
            std::cerr << "SIMD matmul_q4_0_f32 mismatch at row " << r << "\n";
            return false;
        }
    }

    return true;
}

}  // namespace inference_engine::testing
