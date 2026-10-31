#pragma once

#include <cstddef>
#include <cstdint>

#include "inference_engine/common.hpp"

namespace inference_engine {

// Q4_0 block contract (upstream-compatible 32-weight block):
// - 32 weights per block
// - 18 bytes per block:
//   - bytes [0..1]: little-endian IEEE 754 binary16 scale `d`
//   - bytes [2..17]: 16 packed bytes `qs[0..15]`
// - Low nibble (qs[j] & 0x0F) stores weight index j (0..15)
// - High nibble ((qs[j] >> 4) & 0x0F) stores weight index j + 16 (16..31)
// - Dequantisation formula: w_i = (static_cast<int>(q_i) - 8) * fp16_to_fp32(d)
inline constexpr size_t kQ4_0BlockElements = 32;
inline constexpr size_t kQ4_0PackedBytes = 16;
inline constexpr size_t kQ4_0BlockBytes = 18;

#pragma pack(push, 1)
struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[kQ4_0PackedBytes];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4_0) == kQ4_0BlockBytes,
              "BlockQ4_0 must be exactly 18 bytes without padding");

uint16_t fp32_to_fp16(float value) noexcept;
float fp16_to_fp32(uint16_t bits) noexcept;

Status quantize_row_q4_0(const float* src, BlockQ4_0* dst_blocks, size_t num_elements) noexcept;
Status dequantize_row_q4_0(const BlockQ4_0* src_blocks, float* dst, size_t num_elements) noexcept;

inline constexpr uint8_t unpack_q4_0_low_nibble(uint8_t packed_byte) noexcept {
    return static_cast<uint8_t>(packed_byte & 0x0Fu);
}

inline constexpr uint8_t unpack_q4_0_high_nibble(uint8_t packed_byte) noexcept {
    return static_cast<uint8_t>((packed_byte >> 4) & 0x0Fu);
}

inline constexpr uint8_t pack_q4_0_nibbles(uint8_t low_q, uint8_t high_q) noexcept {
    return static_cast<uint8_t>((low_q & 0x0Fu) | ((high_q & 0x0Fu) << 4));
}

}  // namespace inference_engine
