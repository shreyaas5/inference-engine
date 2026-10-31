#include "inference_engine/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace inference_engine {

uint16_t fp32_to_fp16(float value) noexcept {
    uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(float));

    const uint32_t sign = (fbits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = fbits & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> (1 - exp);
        if (mantissa & 0x00001000u) {
            mantissa += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exp == 0xFF - 127 + 15) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        mantissa >>= 13;
        return static_cast<uint16_t>(sign | 0x7C00u | mantissa | (mantissa == 0 ? 1u : 0u));
    }
    if (mantissa & 0x00001000u) {
        mantissa += 0x00002000u;
        if (mantissa & 0x00800000u) {
            mantissa = 0;
            return static_cast<uint16_t>(
                (exp + 1 >= 31) ? (sign | 0x7C00u)
                                : (sign | (static_cast<uint32_t>(exp + 1) << 10)));
        }
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
}

float fp16_to_fp32(uint16_t bits) noexcept {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x03FFu;

    uint32_t out_bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            out_bits = sign;
        } else {
            uint32_t e = 127 - 14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03FFu;
            out_bits = sign | (e << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        out_bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        out_bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &out_bits, sizeof(float));
    return result;
}

Status quantize_row_q4_0(const float* src, BlockQ4_0* dst_blocks, size_t num_elements) noexcept {
    if (src == nullptr || dst_blocks == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "src and dst_blocks must not be null");
    }
    if (num_elements == 0 || (num_elements % kQ4_0BlockElements) != 0) {
        return Status::Error(StatusCode::InvalidArgument,
                             "num_elements must be a positive multiple of 32 for Q4_0");
    }

    const size_t num_blocks = num_elements / kQ4_0BlockElements;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float* block_src = src + b * kQ4_0BlockElements;
        float amax = 0.0f;
        float signed_max = 0.0f;

        for (size_t i = 0; i < kQ4_0BlockElements; ++i) {
            const float v = block_src[i];
            const float av = std::fabs(v);
            if (amax < av) {
                amax = av;
                signed_max = v;
            }
        }

        const float d = signed_max / -8.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        dst_blocks[b].d = fp32_to_fp16(d);

        for (size_t j = 0; j < kQ4_0PackedBytes; ++j) {
            const float x0 = block_src[j] * id;
            const float x1 = block_src[j + kQ4_0PackedBytes] * id;

            const int q0 = std::clamp(static_cast<int>(std::floor(x0 + 8.5f)), 0, 15);
            const int q1 = std::clamp(static_cast<int>(std::floor(x1 + 8.5f)), 0, 15);

            dst_blocks[b].qs[j] =
                pack_q4_0_nibbles(static_cast<uint8_t>(q0), static_cast<uint8_t>(q1));
        }
    }

    return Status::Ok();
}

Status dequantize_row_q4_0(const BlockQ4_0* src_blocks, float* dst, size_t num_elements) noexcept {
    if (src_blocks == nullptr || dst == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "src_blocks and dst must not be null");
    }
    if (num_elements == 0 || (num_elements % kQ4_0BlockElements) != 0) {
        return Status::Error(StatusCode::InvalidArgument,
                             "num_elements must be a positive multiple of 32 for Q4_0");
    }

    const size_t num_blocks = num_elements / kQ4_0BlockElements;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(src_blocks[b].d);
        float* block_dst = dst + b * kQ4_0BlockElements;

        for (size_t j = 0; j < kQ4_0PackedBytes; ++j) {
            const uint8_t packed = src_blocks[b].qs[j];
            const int q_low = static_cast<int>(unpack_q4_0_low_nibble(packed)) - 8;
            const int q_high = static_cast<int>(unpack_q4_0_high_nibble(packed)) - 8;

            block_dst[j] = static_cast<float>(q_low) * d;
            block_dst[j + kQ4_0PackedBytes] = static_cast<float>(q_high) * d;
        }
    }

    return Status::Ok();
}

}  // namespace inference_engine
