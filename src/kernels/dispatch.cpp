#include "inference_engine/kernels.hpp"

namespace inference_engine {

float dispatch_dot_f32(const float* a, const float* b, size_t len, SimdBackend backend) noexcept {
    if (backend == SimdBackend::Neon && is_backend_available(SimdBackend::Neon)) {
        return neon::dot_f32(a, b, len);
    }
    if (backend == SimdBackend::Avx2 && is_backend_available(SimdBackend::Avx2)) {
        return avx2::dot_f32(a, b, len);
    }
    return scalar::dot_f32(a, b, len);
}

float dispatch_dot_q4_0_f32(const BlockQ4_0* weight_row,
                            const float* x,
                            size_t len,
                            SimdBackend backend) noexcept {
    if (backend == SimdBackend::Neon && is_backend_available(SimdBackend::Neon)) {
        return neon::dot_q4_0_f32(weight_row, x, len);
    }
    if (backend == SimdBackend::Avx2 && is_backend_available(SimdBackend::Avx2)) {
        return avx2::dot_q4_0_f32(weight_row, x, len);
    }
    return scalar::dot_q4_0_f32(weight_row, x, len);
}

void dispatch_matmul_f32(float* out,
                         const float* weight,
                         const float* x,
                         size_t out_dim,
                         size_t in_dim,
                         SimdBackend backend,
                         ThreadPool* pool) noexcept {
    const bool use_neon = (backend == SimdBackend::Neon && is_backend_available(SimdBackend::Neon));
    const bool use_avx2 = (backend == SimdBackend::Avx2 && is_backend_available(SimdBackend::Avx2));

    if (pool != nullptr && pool->num_threads() > 1 && out_dim > 1) {
        auto dot_kernel = use_neon ? neon::dot_f32 : (use_avx2 ? avx2::dot_f32 : scalar::dot_f32);
        pool->parallel_for(out_dim, [&](size_t r, size_t /*tid*/) {
            out[r] = dot_kernel(weight + r * in_dim, x, in_dim);
        });
        return;
    }

    if (use_neon) {
        neon::matmul_f32(out, weight, x, out_dim, in_dim);
    } else if (use_avx2) {
        avx2::matmul_f32(out, weight, x, out_dim, in_dim);
    } else {
        scalar::matmul_f32(out, weight, x, out_dim, in_dim);
    }
}

void dispatch_matmul_q4_0_f32(float* out,
                              const BlockQ4_0* weight_blocks,
                              const float* x,
                              size_t out_dim,
                              size_t in_dim,
                              SimdBackend backend,
                              ThreadPool* pool) noexcept {
    const size_t blocks_per_row = in_dim / kQ4_0BlockElements;
    const bool use_neon = (backend == SimdBackend::Neon && is_backend_available(SimdBackend::Neon));
    const bool use_avx2 = (backend == SimdBackend::Avx2 && is_backend_available(SimdBackend::Avx2));

    if (pool != nullptr && pool->num_threads() > 1 && out_dim > 1) {
        auto dot_kernel = use_neon ? neon::dot_q4_0_f32 : (use_avx2 ? avx2::dot_q4_0_f32 : scalar::dot_q4_0_f32);
        pool->parallel_for(out_dim, [&](size_t r, size_t /*tid*/) {
            out[r] = dot_kernel(weight_blocks + r * blocks_per_row, x, in_dim);
        });
        return;
    }

    if (use_neon) {
        neon::matmul_q4_0_f32(out, weight_blocks, x, out_dim, in_dim);
    } else if (use_avx2) {
        avx2::matmul_q4_0_f32(out, weight_blocks, x, out_dim, in_dim);
    } else {
        scalar::matmul_q4_0_f32(out, weight_blocks, x, out_dim, in_dim);
    }
}

}  // namespace inference_engine
