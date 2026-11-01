#include "inference_engine/kernels.hpp"
#include "inference_engine/kv_cache.hpp"

#include <algorithm>
#include <cmath>

namespace inference_engine::scalar {

void rmsnorm_f32(float* out, const float* x, const float* weight, size_t dim, float eps) noexcept {
    if (out == nullptr || x == nullptr || weight == nullptr || dim == 0) {
        return;
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    const float inv_rms =
        1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(dim)) + eps);
    for (size_t i = 0; i < dim; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

void rope_inplace_f32(float* qk,
                      size_t num_heads,
                      size_t head_dim,
                      uint32_t pos,
                      float rope_theta) noexcept {
    if (qk == nullptr || head_dim == 0 || (head_dim % 2) != 0) {
        return;
    }
    const size_t half_dim = head_dim / 2;
    for (size_t h = 0; h < num_heads; ++h) {
        float* head = qk + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            const float freq =
                1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(pos) * freq;
            const float cos_val = std::cos(angle);
            const float sin_val = std::sin(angle);

            const float x0 = head[i];
            const float x1 = head[i + half_dim];
            head[i] = x0 * cos_val - x1 * sin_val;
            head[i + half_dim] = x1 * cos_val + x0 * sin_val;
        }
    }
}

void rope_inplace_cached_f32(float* qk,
                             size_t num_heads,
                             size_t head_dim,
                             const float* cos_row,
                             const float* sin_row) noexcept {
    if (qk == nullptr || cos_row == nullptr || sin_row == nullptr ||
        num_heads == 0 || head_dim == 0 || (head_dim % 2 != 0)) {
        return;
    }
    const size_t half_dim = head_dim / 2;
    for (size_t h = 0; h < num_heads; ++h) {
        float* head = qk + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            const float cos_val = cos_row[i];
            const float sin_val = sin_row[i];
            const float x0 = head[i];
            const float x1 = head[i + half_dim];
            head[i] = x0 * cos_val - x1 * sin_val;
            head[i + half_dim] = x1 * cos_val + x0 * sin_val;
        }
    }
}

float dot_f32(const float* a, const float* b, size_t len) noexcept {
    float acc = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept {
    const size_t num_blocks = len / kQ4_0BlockElements;
    float total = 0.0f;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(weight_row[b].d);
        const float* x_block = x + b * kQ4_0BlockElements;
        float block_sum = 0.0f;
        for (size_t j = 0; j < kQ4_0PackedBytes; ++j) {
            const uint8_t packed = weight_row[b].qs[j];
            const int q0 = static_cast<int>(unpack_q4_0_low_nibble(packed)) - 8;
            const int q1 = static_cast<int>(unpack_q4_0_high_nibble(packed)) - 8;
            block_sum += static_cast<float>(q0) * x_block[j] +
                         static_cast<float>(q1) * x_block[j + kQ4_0PackedBytes];
        }
        total += d * block_sum;
    }
    return total;
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

void swiglu_f32(float* out, const float* gate, const float* up, size_t dim) noexcept {
    for (size_t i = 0; i < dim; ++i) {
        const float g = gate[i];
        const float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
}

void softmax_f32(float* x, size_t len) noexcept {
    if (x == nullptr || len == 0) {
        return;
    }
    float max_val = x[0];
    for (size_t i = 1; i < len; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += static_cast<double>(x[i]);
    }
    const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (size_t i = 0; i < len; ++i) {
        x[i] *= inv_sum;
    }
}

void gqa_attention_f32(float* out,
                       const float* q,
                       const KvCache& kv_cache,
                       uint32_t layer_idx,
                       uint32_t pos,
                       uint32_t num_query_heads,
                       uint32_t num_kv_heads,
                       uint32_t head_dim,
                       float* scores_scratch) noexcept {
    if (out == nullptr || q == nullptr || scores_scratch == nullptr ||
        num_query_heads == 0 || num_kv_heads == 0 || head_dim == 0) {
        return;
    }
    const uint32_t gqa_group_size = num_query_heads / num_kv_heads;
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t seq_count = pos + 1;

    for (uint32_t hq = 0; hq < num_query_heads; ++hq) {
        const uint32_t hkv = (gqa_group_size > 0) ? (hq / gqa_group_size) : 0;
        const float* q_head = q + static_cast<size_t>(hq) * head_dim;

        // 1. Compute dot-product attention scores across all cached positions 0..pos
        for (uint32_t t = 0; t < seq_count; ++t) {
            const float* k_vec = kv_cache.key_vector(layer_idx, t);
            if (k_vec == nullptr) {
                scores_scratch[t] = 0.0f;
                continue;
            }
            const float* k_head = k_vec + static_cast<size_t>(hkv) * head_dim;
            scores_scratch[t] = dot_f32(q_head, k_head, head_dim) * head_scale;
        }

        // 2. Numerically stable softmax over scores_scratch[0..pos]
        softmax_f32(scores_scratch, seq_count);

        // 3. Accumulate weighted values
        float* out_head = out + static_cast<size_t>(hq) * head_dim;
        std::fill(out_head, out_head + head_dim, 0.0f);

        for (uint32_t t = 0; t < seq_count; ++t) {
            const float* v_vec = kv_cache.value_vector(layer_idx, t);
            if (v_vec == nullptr) {
                continue;
            }
            const float* v_head = v_vec + static_cast<size_t>(hkv) * head_dim;
            const float weight = scores_scratch[t];
            for (uint32_t d = 0; d < head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
        }
    }
}

}  // namespace inference_engine::scalar
