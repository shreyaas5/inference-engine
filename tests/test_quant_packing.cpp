#include <cmath>
#include <iostream>
#include <vector>

#include "inference_engine/quant.hpp"

namespace inference_engine::testing {

bool run_quant_packing_tests() {
    if (sizeof(BlockQ4_0) != 18) {
        std::cerr << "BlockQ4_0 size must be 18 bytes\n";
        return false;
    }

    // Test 1: Zero block handling (all zeros -> d = 0, dequantizes to exact 0.0f).
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

    // Test 2: Nibble ordering contract (low nibble = indices 0..15, high nibble = indices 16..31).
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

    // Test 3: Round-trip quantize/dequantize bounded reconstruction error.
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

    return true;
}

}  // namespace inference_engine::testing
