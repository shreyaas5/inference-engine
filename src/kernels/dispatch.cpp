#include "inference_engine/kernels.hpp"

#include <algorithm>

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
        const size_t num_threads = pool->num_threads();
        const size_t target_tasks = std::min(out_dim, num_threads * 4);
        const size_t rows_per_task = std::max<size_t>(2, ((out_dim + target_tasks - 1) / target_tasks + 1) & ~size_t(1));
        const size_t actual_tasks = (out_dim + rows_per_task - 1) / rows_per_task;

        pool->parallel_for(actual_tasks, [&](size_t task_idx, size_t /*tid*/) {
            const size_t r_start = task_idx * rows_per_task;
            const size_t r_count = std::min(rows_per_task, out_dim - r_start);
            if (r_count > 0) {
                if (use_neon) {
                    neon::matmul_f32(out + r_start, weight + r_start * in_dim, x, r_count, in_dim);
                } else if (use_avx2) {
                    avx2::matmul_f32(out + r_start, weight + r_start * in_dim, x, r_count, in_dim);
                } else {
                    scalar::matmul_f32(out + r_start, weight + r_start * in_dim, x, r_count, in_dim);
                }
            }
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
        const size_t num_threads = pool->num_threads();
        const size_t target_tasks = std::min(out_dim, num_threads * 4);
        const size_t rows_per_task = std::max<size_t>(2, ((out_dim + target_tasks - 1) / target_tasks + 1) & ~size_t(1));
        const size_t actual_tasks = (out_dim + rows_per_task - 1) / rows_per_task;

        pool->parallel_for(actual_tasks, [&](size_t task_idx, size_t /*tid*/) {
            const size_t r_start = task_idx * rows_per_task;
            const size_t r_count = std::min(rows_per_task, out_dim - r_start);
            if (r_count > 0) {
                if (use_neon) {
                    neon::matmul_q4_0_f32(out + r_start, weight_blocks + r_start * blocks_per_row, x, r_count, in_dim);
                } else if (use_avx2) {
                    avx2::matmul_q4_0_f32(out + r_start, weight_blocks + r_start * blocks_per_row, x, r_count, in_dim);
                } else {
                    scalar::matmul_q4_0_f32(out + r_start, weight_blocks + r_start * blocks_per_row, x, r_count, in_dim);
                }
            }
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
