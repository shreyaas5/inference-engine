#include "inference_engine/kernels.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace inference_engine::avx2 {

namespace {

#if defined(__AVX2__)
// Fast horizontal reduction of 8-lane 256-bit float vector to a single float in registers.
inline float hsum256_ps(__m256 v) noexcept {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 sum128 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehl_ps(sum128, sum128);
    const __m128 sum64 = _mm_add_ps(sum128, shuf);
    const __m128 shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    const __m128 sum32 = _mm_add_ss(sum64, shuf2);
    return _mm_cvtss_f32(sum32);
}

// Computes the unscaled inner product sum_{i=0..31} (q_i - 8) * x_i for a single Q4_0 block.
inline float compute_block_unscaled_dot(const BlockQ4_0& block, const float* x_block) noexcept {
    // 1. Unaligned 16-byte load of the packed nibbles (BlockQ4_0 is 18 bytes, not 16-byte aligned).
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.qs));

    // 2. Separate low nibbles (weights 0..15) and high nibbles (weights 16..31).
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i eight_8 = _mm_set1_epi8(8);

    const __m128i q_low = _mm_and_si128(raw, mask_0f);
    const __m128i q_high = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

    // 3. Subtract 8 to obtain signed 8-bit integers in range [-8, +7].
    const __m128i low_s = _mm_sub_epi8(q_low, eight_8);
    const __m128i high_s = _mm_sub_epi8(q_high, eight_8);

    // 4. Sign-extend 8-bit signed integers to 32-bit integers, then convert to FP32.
    // Weights 0..7 (low half of low_s)
    const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_s));
    const __m256 x0 = _mm256_loadu_ps(x_block + 0);

    // Weights 8..15 (high half of low_s shifted to low 8 bytes)
    const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low_s, 8)));
    const __m256 x1 = _mm256_loadu_ps(x_block + 8);

    // Weights 16..23 (low half of high_s)
    const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_s));
    const __m256 x2 = _mm256_loadu_ps(x_block + 16);

    // Weights 24..31 (high half of high_s shifted to low 8 bytes)
    const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high_s, 8)));
    const __m256 x3 = _mm256_loadu_ps(x_block + 24);

    // 5. Multiply-accumulate across the four 8-wide float vectors.
#if defined(__FMA__)
    __m256 acc0 = _mm256_mul_ps(w0, x0);
    __m256 acc1 = _mm256_mul_ps(w1, x1);
    acc0 = _mm256_fmadd_ps(w2, x2, acc0);
    acc1 = _mm256_fmadd_ps(w3, x3, acc1);
    const __m256 block_acc = _mm256_add_ps(acc0, acc1);
#else
    const __m256 p0 = _mm256_mul_ps(w0, x0);
    const __m256 p1 = _mm256_mul_ps(w1, x1);
    const __m256 p2 = _mm256_mul_ps(w2, x2);
    const __m256 p3 = _mm256_mul_ps(w3, x3);
    const __m256 block_acc = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
#endif

    return hsum256_ps(block_acc);
}
#endif

}  // namespace

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
#if defined(__AVX2__)
    if (weight_row == nullptr || x == nullptr || len == 0) {
        return 0.0f;
    }

    const size_t num_blocks = len / kQ4_0BlockElements;
    float total = 0.0f;
    size_t b = 0;

    // 2-block unrolled loop to maximize instruction-level parallelism
    for (; b + 2 <= num_blocks; b += 2) {
        const float d0 = fp16_to_fp32(weight_row[b].d);
        const float sum0 = compute_block_unscaled_dot(weight_row[b], x + b * kQ4_0BlockElements);

        const float d1 = fp16_to_fp32(weight_row[b + 1].d);
        const float sum1 = compute_block_unscaled_dot(weight_row[b + 1], x + (b + 1) * kQ4_0BlockElements);

        total += (d0 * sum0) + (d1 * sum1);
    }

    // Process remaining single block if num_blocks is odd
    for (; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(weight_row[b].d);
        const float sum = compute_block_unscaled_dot(weight_row[b], x + b * kQ4_0BlockElements);
        total += d * sum;
    }

    return total;
#else
    return scalar::dot_q4_0_f32(weight_row, x, len);
#endif
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
