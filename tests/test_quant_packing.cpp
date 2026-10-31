#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "inference_engine/quant.hpp"

namespace inference_engine::testing {

namespace {

bool test_fp16_edge_cases() {
    // 1. Zeros (+0.0f and -0.0f)
    const uint16_t pos_zero = fp32_to_fp16(0.0f);
    if (pos_zero != 0x0000u || fp16_to_fp32(pos_zero) != 0.0f) {
        std::cerr << "FP16 +0.0f roundtrip failed\n";
        return false;
    }
    const uint16_t neg_zero = fp32_to_fp16(-0.0f);
    if (neg_zero != 0x8000u || fp16_to_fp32(neg_zero) != -0.0f) {
        std::cerr << "FP16 -0.0f roundtrip failed\n";
        return false;
    }

    // 2. Exact standard numbers
    const float test_values[] = {1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.25f, 64.0f, -128.0f};
    for (float val : test_values) {
        const uint16_t h = fp32_to_fp16(val);
        const float rec = fp16_to_fp32(h);
        if (rec != val) {
            std::cerr << "FP16 exact value roundtrip failed for " << val << " (got " << rec << ")\n";
            return false;
        }
    }

    // 3. Maximum normal FP16 value (65504.0f)
    const uint16_t max_norm = fp32_to_fp16(65504.0f);
    if (max_norm != 0x7BFFu || fp16_to_fp32(max_norm) != 65504.0f) {
        std::cerr << "FP16 max normal roundtrip failed\n";
        return false;
    }

    // 4. Overflow to infinity (> 65504.0f)
    const uint16_t inf_pos = fp32_to_fp16(70000.0f);
    if (inf_pos != 0x7C00u || !std::isinf(fp16_to_fp32(inf_pos))) {
        std::cerr << "FP16 positive overflow to infinity failed\n";
        return false;
    }
    const uint16_t inf_neg = fp32_to_fp16(-70000.0f);
    if (inf_neg != 0xFC00u || !std::isinf(fp16_to_fp32(inf_neg))) {
        std::cerr << "FP16 negative overflow to infinity failed\n";
        return false;
    }

    // 5. Explicit Infinity
    const float pos_inf_f = std::numeric_limits<float>::infinity();
    const float neg_inf_f = -std::numeric_limits<float>::infinity();
    if (fp32_to_fp16(pos_inf_f) != 0x7C00u || fp32_to_fp16(neg_inf_f) != 0xFC00u) {
        std::cerr << "FP16 infinity bit encoding failed\n";
        return false;
    }

    // 6. NaN handling
    const float qnan_f = std::numeric_limits<float>::quiet_NaN();
    const uint16_t nan_h = fp32_to_fp16(qnan_f);
    if ((nan_h & 0x7C00u) != 0x7C00u || (nan_h & 0x03FFu) == 0) {
        std::cerr << "FP16 NaN encoding failed\n";
        return false;
    }
    if (!std::isnan(fp16_to_fp32(nan_h))) {
        std::cerr << "FP16 NaN decoding failed\n";
        return false;
    }

    // 7. Smallest normal value (2^-14 = 6.1035156e-5f)
    const float min_norm_f = 6.103515625e-5f;
    const uint16_t min_norm_h = fp32_to_fp16(min_norm_f);
    if (min_norm_h != 0x0400u || std::fabs(fp16_to_fp32(min_norm_h) - min_norm_f) > 1e-9f) {
        std::cerr << "FP16 min normal roundtrip failed\n";
        return false;
    }

    // 8. Subnormal values (e.g. 2^-15, 2^-24)
    const float subnorm_f = 3.0517578125e-5f;  // 2^-15
    const uint16_t subnorm_h = fp32_to_fp16(subnorm_f);
    if (subnorm_h != 0x0200u || std::fabs(fp16_to_fp32(subnorm_h) - subnorm_f) > 1e-9f) {
        std::cerr << "FP16 subnormal roundtrip failed\n";
        return false;
    }

    return true;
}

bool test_validation_errors() {
    std::vector<float> data(32, 1.0f);
    BlockQ4_0 block{};

    // Null pointers
    if (quantize_row_q4_0(nullptr, &block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 should fail on null src\n";
        return false;
    }
    if (quantize_row_q4_0(data.data(), nullptr, 32).ok()) {
        std::cerr << "quantize_row_q4_0 should fail on null dst\n";
        return false;
    }
    if (dequantize_row_q4_0(nullptr, data.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 should fail on null src\n";
        return false;
    }
    if (dequantize_row_q4_0(&block, nullptr, 32).ok()) {
        std::cerr << "dequantize_row_q4_0 should fail on null dst\n";
        return false;
    }

    // Non-multiple of 32 or zero elements
    const size_t bad_sizes[] = {0, 1, 16, 31, 33, 63, 65};
    for (size_t sz : bad_sizes) {
        std::vector<float> bad_buf(sz > 0 ? sz : 1, 0.0f);
        std::vector<BlockQ4_0> bad_blocks((sz + 31) / 32 + 1);
        if (quantize_row_q4_0(bad_buf.data(), bad_blocks.data(), sz).ok()) {
            std::cerr << "quantize_row_q4_0 should fail on size " << sz << "\n";
            return false;
        }
        if (dequantize_row_q4_0(bad_blocks.data(), bad_buf.data(), sz).ok()) {
            std::cerr << "dequantize_row_q4_0 should fail on size " << sz << "\n";
            return false;
        }
    }

    return true;
}

bool test_block_distributions() {
    // 1. Positive-only values
    std::vector<float> pos_src(32);
    for (size_t i = 0; i < 32; ++i) {
        pos_src[i] = 0.5f + static_cast<float>(i) * 0.25f;
    }
    BlockQ4_0 pos_block{};
    if (!quantize_row_q4_0(pos_src.data(), &pos_block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 failed on positive-only block\n";
        return false;
    }
    std::vector<float> pos_rec(32, 0.0f);
    if (!dequantize_row_q4_0(&pos_block, pos_rec.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on positive-only block\n";
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        const float step = std::fabs(pos_src[31]) / 8.0f;
        if (std::fabs(pos_src[i] - pos_rec[i]) > step) {
            std::cerr << "Positive-only reconstruction error exceeded bound at " << i << "\n";
            return false;
        }
    }

    // 2. Negative-only values
    std::vector<float> neg_src(32);
    for (size_t i = 0; i < 32; ++i) {
        neg_src[i] = -0.5f - static_cast<float>(i) * 0.25f;
    }
    BlockQ4_0 neg_block{};
    if (!quantize_row_q4_0(neg_src.data(), &neg_block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 failed on negative-only block\n";
        return false;
    }
    std::vector<float> neg_rec(32, 0.0f);
    if (!dequantize_row_q4_0(&neg_block, neg_rec.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on negative-only block\n";
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        const float step = std::fabs(neg_src[31]) / 8.0f;
        if (std::fabs(neg_src[i] - neg_rec[i]) > step) {
            std::cerr << "Negative-only reconstruction error exceeded bound at " << i << "\n";
            return false;
        }
    }

    // 3. Repeated constant values (e.g. all 3.5f)
    std::vector<float> const_src(32, 3.5f);
    BlockQ4_0 const_block{};
    if (!quantize_row_q4_0(const_src.data(), &const_block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 failed on constant block\n";
        return false;
    }
    std::vector<float> const_rec(32, 0.0f);
    if (!dequantize_row_q4_0(&const_block, const_rec.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on constant block\n";
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        if (std::fabs(const_src[i] - const_rec[i]) > 1e-4f) {
            std::cerr << "Constant block reconstruction error exceeded bound at " << i << "\n";
            return false;
        }
    }

    // 4. Large representable values
    std::vector<float> large_src(32, 1000.0f);
    large_src[0] = -4000.0f;
    BlockQ4_0 large_block{};
    if (!quantize_row_q4_0(large_src.data(), &large_block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 failed on large block\n";
        return false;
    }
    std::vector<float> large_rec(32, 0.0f);
    if (!dequantize_row_q4_0(&large_block, large_rec.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on large block\n";
        return false;
    }
    if (std::fabs(large_rec[0] - (-4000.0f)) > 1.0f) {
        std::cerr << "Large block extreme value reconstruction mismatch\n";
        return false;
    }

    // 5. Multi-block row (128 elements = 4 blocks) with pseudorandom numbers
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> multi_src(128);
    for (size_t i = 0; i < multi_src.size(); ++i) {
        multi_src[i] = dist(rng);
    }
    std::vector<BlockQ4_0> multi_blocks(4);
    if (!quantize_row_q4_0(multi_src.data(), multi_blocks.data(), multi_src.size()).ok()) {
        std::cerr << "quantize_row_q4_0 failed on multi-block row\n";
        return false;
    }
    std::vector<float> multi_rec(128, 0.0f);
    if (!dequantize_row_q4_0(multi_blocks.data(), multi_rec.data(), multi_src.size()).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on multi-block row\n";
        return false;
    }
    for (size_t b = 0; b < 4; ++b) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < 32; ++i) {
            max_abs = std::max(max_abs, std::fabs(multi_src[b * 32 + i]));
        }
        const float max_allowed_err = (max_abs / 8.0f) + 1e-4f;
        for (size_t i = 0; i < 32; ++i) {
            const float err = std::fabs(multi_src[b * 32 + i] - multi_rec[b * 32 + i]);
            if (err > max_allowed_err) {
                std::cerr << "Multi-block error exceeded bound at " << (b * 32 + i) << "\n";
                return false;
            }
        }
    }

    return true;
}

}  // namespace

bool run_quant_packing_tests() {
    if (sizeof(BlockQ4_0) != 18) {
        std::cerr << "BlockQ4_0 size must be 18 bytes\n";
        return false;
    }

    // Baseline Test 1: Zero block handling (all zeros -> d = 0, dequantizes to exact 0.0f).
    std::vector<float> zeros(32, 0.0f);
    BlockQ4_0 zero_block{};
    if (!quantize_row_q4_0(zeros.data(), &zero_block, 32).ok()) {
        std::cerr << "quantize_row_q4_0 failed on zero block\n";
        return false;
    }
    if (fp16_to_fp32(zero_block.d) != 0.0f) {
        std::cerr << "Expected zero block scale to be 0.0f\n";
        return false;
    }
    for (size_t j = 0; j < kQ4_0PackedBytes; ++j) {
        if (unpack_q4_0_low_nibble(zero_block.qs[j]) != 8 ||
            unpack_q4_0_high_nibble(zero_block.qs[j]) != 8) {
            std::cerr << "Expected zero block nibbles to equal 8\n";
            return false;
        }
    }

    std::vector<float> deq_zeros(32, 1.0f);
    if (!dequantize_row_q4_0(&zero_block, deq_zeros.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on zero block\n";
        return false;
    }
    for (float v : deq_zeros) {
        if (v != 0.0f) {
            std::cerr << "Expected dequantized zero block element to be 0.0f\n";
            return false;
        }
    }

    // Baseline Test 2: Nibble ordering contract (low nibble = indices 0..15, high nibble = indices 16..31).
    BlockQ4_0 manual_block{};
    manual_block.d = fp32_to_fp16(0.5f);
    for (uint8_t j = 0; j < 16; ++j) {
        // Low nibble gets j (0..15), high nibble gets 15 - j (15..0).
        manual_block.qs[j] = pack_q4_0_nibbles(j, static_cast<uint8_t>(15 - j));
    }
    std::vector<float> manual_out(32, 0.0f);
    if (!dequantize_row_q4_0(&manual_block, manual_out.data(), 32).ok()) {
        std::cerr << "dequantize_row_q4_0 failed on manual block\n";
        return false;
    }
    for (size_t j = 0; j < 16; ++j) {
        const float expected_low = (static_cast<int>(j) - 8) * 0.5f;
        const float expected_high = (static_cast<int>(15 - j) - 8) * 0.5f;
        if (std::fabs(manual_out[j] - expected_low) > 1e-6f ||
            std::fabs(manual_out[j + 16] - expected_high) > 1e-6f) {
            std::cerr << "Nibble ordering mismatch at index " << j << "\n";
            return false;
        }
    }

    // Baseline Test 3: Round-trip quantize/dequantize bounded reconstruction error.
    std::vector<float> ramp(32);
    for (size_t i = 0; i < 32; ++i) {
        ramp[i] = (static_cast<float>(i) - 16.0f) * 0.25f;
    }
    BlockQ4_0 ramp_block{};
    if (!quantize_row_q4_0(ramp.data(), &ramp_block, 32).ok()) {
        return false;
    }
    std::vector<float> ramp_rec(32, 0.0f);
    if (!dequantize_row_q4_0(&ramp_block, ramp_rec.data(), 32).ok()) {
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        if (std::fabs(ramp[i] - ramp_rec[i]) > 0.30f) {
            std::cerr << "Reconstruction error exceeded bound at index " << i << "\n";
            return false;
        }
    }

    // Extended Task A / Phase 2 validations
    if (!test_fp16_edge_cases()) {
        return false;
    }
    if (!test_validation_errors()) {
        return false;
    }
    if (!test_block_distributions()) {
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
