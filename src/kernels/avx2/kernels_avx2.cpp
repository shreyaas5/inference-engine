#include "inference_engine/kernels.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace inference_engine::avx2 {

float dot_f32(const float* a, const float* b, size_t len) noexcept {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(va, vb, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float total = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                  lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < len; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
    return scalar::dot_f32(a, b, len);
#endif
}

float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept {
    // M0 foundation: delegates to scalar reference until M3 AVX2 Q4_0 kernel implementation.
    return scalar::dot_q4_0_f32(weight_row, x, len);
}

void matmul_f32(float* out,
                const float* weight,
                const float* x,
                size_t out_dim,
                size_t in_dim) noexcept {
    for (size_t r = 0; r < out_dim; ++r) {
        out[r] = dot_f32(weight + r * in_dim, x, in_dim);
    }
}

void matmul_q4_0_f32(float* out,
                     const BlockQ4_0* weight_blocks,
                     const float* x,
                     size_t out_dim,
                     size_t in_dim) noexcept {
    const size_t blocks_per_row = in_dim / kQ4_0BlockElements;
    for (size_t r = 0; r < out_dim; ++r) {
        out[r] = dot_q4_0_f32(weight_blocks + r * blocks_per_row, x, in_dim);
    }
}

}  // namespace inference_engine::avx2
