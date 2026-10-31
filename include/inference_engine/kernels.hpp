#pragma once

#include <cstddef>
#include <cstdint>

#include "inference_engine/common.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/thread_pool.hpp"

namespace inference_engine {

class KvCache;

namespace scalar {

void rmsnorm_f32(float* out, const float* x, const float* weight, size_t dim, float eps) noexcept;
void rope_inplace_f32(float* qk,
                      size_t num_heads,
                      size_t head_dim,
                      uint32_t pos,
                      float rope_theta) noexcept;
void matmul_f32(float* out,
                const float* weight,
                const float* x,
                size_t out_dim,
                size_t in_dim) noexcept;
void matmul_q4_0_f32(float* out,
                     const BlockQ4_0* weight_blocks,
                     const float* x,
                     size_t out_dim,
                     size_t in_dim) noexcept;
void swiglu_f32(float* out, const float* gate, const float* up, size_t dim) noexcept;
void softmax_f32(float* x, size_t len) noexcept;
float dot_f32(const float* a, const float* b, size_t len) noexcept;
float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept;

void gqa_attention_f32(float* out,
                       const float* q,
                       const KvCache& kv_cache,
                       uint32_t layer_idx,
                       uint32_t pos,
                       uint32_t num_query_heads,
                       uint32_t num_kv_heads,
                       uint32_t head_dim,
                       float* scores_scratch) noexcept;

}  // namespace scalar

namespace neon {

float dot_f32(const float* a, const float* b, size_t len) noexcept;
float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept;
void matmul_f32(float* out,
                const float* weight,
                const float* x,
                size_t out_dim,
                size_t in_dim) noexcept;
void matmul_q4_0_f32(float* out,
                     const BlockQ4_0* weight_blocks,
                     const float* x,
                     size_t out_dim,
                     size_t in_dim) noexcept;

}  // namespace neon

namespace avx2 {

float dot_f32(const float* a, const float* b, size_t len) noexcept;
float dot_q4_0_f32(const BlockQ4_0* weight_row, const float* x, size_t len) noexcept;
void matmul_f32(float* out,
                const float* weight,
                const float* x,
                size_t out_dim,
                size_t in_dim) noexcept;
void matmul_q4_0_f32(float* out,
                     const BlockQ4_0* weight_blocks,
                     const float* x,
                     size_t out_dim,
                     size_t in_dim) noexcept;

}  // namespace avx2

// Runtime-dispatched kernel entry points.
float dispatch_dot_f32(const float* a, const float* b, size_t len, SimdBackend backend) noexcept;
float dispatch_dot_q4_0_f32(const BlockQ4_0* weight_row,
                            const float* x,
                            size_t len,
                            SimdBackend backend) noexcept;
void dispatch_matmul_f32(float* out,
                          const float* weight,
                          const float* x,
                          size_t out_dim,
                          size_t in_dim,
                          SimdBackend backend,
                          ThreadPool* pool = nullptr) noexcept;
void dispatch_matmul_q4_0_f32(float* out,
                               const BlockQ4_0* weight_blocks,
                               const float* x,
                               size_t out_dim,
                               size_t in_dim,
                               SimdBackend backend,
                               ThreadPool* pool = nullptr) noexcept;

}  // namespace inference_engine
