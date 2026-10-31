#include "inference_engine/kernels.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace inference_engine::neon {

namespace {

#if defined(__aarch64__) || defined(_M_ARM64)
// Computes unscaled inner product sum_{i=0..31} (q_i - 8) * x_i for a single Q4_0 block using ARM64 NEON.
inline float compute_block_unscaled_dot_neon(const BlockQ4_0& block, const float* x_block) noexcept {
    // 1. Unaligned 16-byte load of packed nibbles.
    const uint8x16_t raw = vld1q_u8(block.qs);

    // 2. Separate low nibbles (weights 0..15) and high nibbles (weights 16..31).
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const uint8x16_t q_low = vandq_u8(raw, mask_0f);
    const uint8x16_t q_high = vshrq_n_u8(raw, 4);

    // 3. Subtract 8 to obtain signed 8-bit integers in range [-8, +7].
    const int8x16_t eight_8 = vdupq_n_s8(8);
    const int8x16_t low_s = vsubq_s8(vreinterpretq_s8_u8(q_low), eight_8);
    const int8x16_t high_s = vsubq_s8(vreinterpretq_s8_u8(q_high), eight_8);

    // 4. Widen int8 -> int16.
    const int16x8_t s16_0 = vmovl_s8(vget_low_s8(low_s));    // weights 0..7
    const int16x8_t s16_1 = vmovl_s8(vget_high_s8(low_s));   // weights 8..15
    const int16x8_t s16_2 = vmovl_s8(vget_low_s8(high_s));   // weights 16..23
    const int16x8_t s16_3 = vmovl_s8(vget_high_s8(high_s));  // weights 24..31

    // 5. Widen int16 -> int32 and convert to float32, then FMA with activations.
    const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
    const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_0)));
    const float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
    const float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_1)));

    const float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
    const float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_2)));
    const float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
    const float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_3)));

    const float32x4_t x0 = vld1q_f32(x_block + 0);
    const float32x4_t x1 = vld1q_f32(x_block + 4);
    const float32x4_t x2 = vld1q_f32(x_block + 8);
    const float32x4_t x3 = vld1q_f32(x_block + 12);
    const float32x4_t x4 = vld1q_f32(x_block + 16);
    const float32x4_t x5 = vld1q_f32(x_block + 20);
    const float32x4_t x6 = vld1q_f32(x_block + 24);
    const float32x4_t x7 = vld1q_f32(x_block + 28);

    float32x4_t acc0 = vmulq_f32(f0, x0);
    float32x4_t acc1 = vmulq_f32(f1, x1);
    acc0 = vfmaq_f32(acc0, f2, x2);
    acc1 = vfmaq_f32(acc1, f3, x3);
    acc0 = vfmaq_f32(acc0, f4, x4);
    acc1 = vfmaq_f32(acc1, f5, x5);
    acc0 = vfmaq_f32(acc0, f6, x6);
    acc1 = vfmaq_f32(acc1, f7, x7);

    return vaddvq_f32(vaddq_f32(acc0, acc1));
}
#endif

}  // namespace

float dot_f32(const float* a, const float* b, size_t len) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }
    float total = vaddvq_f32(acc0);
    for (; i < len; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
    return scalar::dot_f32(a, b, len);
#endif
}

float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (weight_row == nullptr || x == nullptr || len == 0) {
        return 0.0f;
    }

    const size_t num_blocks = len / kQ4_0BlockElements;
    float total = 0.0f;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(weight_row[b].d);
        const float sum = compute_block_unscaled_dot_neon(weight_row[b], x + b * kQ4_0BlockElements);
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

}  // namespace inference_engine::neon
