#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/kernels.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/thread_pool.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
static inline unsigned long long safe_xgetbv(unsigned int index) {
    return _xgetbv(index);
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static inline unsigned long long safe_xgetbv(unsigned int index) {
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
}
#endif
#endif

namespace inference_engine::testing {

namespace {

// Generates reproducible pseudorandom float vector in [low, high]
std::vector<float> generate_random_floats(size_t n, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

// Tolerance checker calibrated for FP32 SIMD horizontal reduction and FMA contraction
bool check_close(float ref, float val, float rel_tol = 1e-4f, float abs_tol = 1e-4f) {
    const float diff = std::fabs(ref - val);
    if (diff <= abs_tol) {
        return true;
    }
    const float max_mag = std::max(std::fabs(ref), std::fabs(val));
    return (diff / max_mag) <= rel_tol;
}

// ============================================================================
// Task A: Native AVX2 & CPUID Verification
// ============================================================================
bool test_native_cpu_features() {
    std::cout << "[TASK A] Verifying native CPU feature detection, CPUID, and fallback...\n";

    // 1. Scalar backend must always be reported as available
    if (!is_backend_available(SimdBackend::Scalar)) {
        std::cerr << "Scalar backend must always be available!\n";
        return false;
    }

#if defined(__x86_64__) || defined(_M_X64)
    bool has_cpuid_avx2 = false;
    bool has_cpuid_fma = false;
    bool has_cpuid_osxsave = false;
    bool os_ymm_enabled = false;

#if defined(_MSC_VER)
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    if (info[0] >= 1) {
        __cpuid(info, 1);
        has_cpuid_osxsave = (info[2] & (1 << 27)) != 0;
        has_cpuid_fma = (info[2] & (1 << 12)) != 0;
        if (has_cpuid_osxsave) {
            const unsigned long long xcr0 = safe_xgetbv(0);
            os_ymm_enabled = ((xcr0 & 0x6ULL) == 0x6ULL);
        }
    }
    if (info[0] >= 7) {
        int ext[4] = {0, 0, 0, 0};
        __cpuidex(ext, 7, 0);
        has_cpuid_avx2 = (ext[1] & (1 << 5)) != 0;
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        has_cpuid_osxsave = (ecx & (1 << 27)) != 0;
        has_cpuid_fma = (ecx & (1 << 12)) != 0;
        if (has_cpuid_osxsave) {
            const unsigned long long xcr0 = safe_xgetbv(0);
            os_ymm_enabled = ((xcr0 & 0x6ULL) == 0x6ULL);
        }
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        has_cpuid_avx2 = (ebx & (1 << 5)) != 0;
    }
#endif

    std::cout << "  - Hardware CPUID: AVX2=" << (has_cpuid_avx2 ? "yes" : "no")
              << ", FMA=" << (has_cpuid_fma ? "yes" : "no")
              << ", OSXSAVE=" << (has_cpuid_osxsave ? "yes" : "no")
              << ", OS XMM/YMM Enabled=" << (os_ymm_enabled ? "yes" : "no") << "\n";

    const bool avx2_detected = is_backend_available(SimdBackend::Avx2);
    std::cout << "  - Runtime is_backend_available(Avx2): " << (avx2_detected ? "true" : "false") << "\n";

    if (has_cpuid_avx2 && has_cpuid_fma && has_cpuid_osxsave && os_ymm_enabled) {
        if (!avx2_detected) {
            std::cerr << "is_backend_available(Avx2) returned false on AVX2-capable CPU!\n";
            return false;
        }
        if (detect_best_backend() != SimdBackend::Avx2) {
            std::cerr << "detect_best_backend() should be Avx2 on this machine!\n";
            return false;
        }
    }
#endif

    // 2. Test explicit scalar fallback through dispatch functions
    const size_t len = 64;
    auto w_f32 = generate_random_floats(len, -1.0f, 1.0f, 42);
    auto x_f32 = generate_random_floats(len, -1.0f, 1.0f, 43);
    std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
    quantize_row_q4_0(w_f32.data(), q4_row.data(), len);

    // Q4_0 dot fallback
    const float ref_dot_q4 = scalar::dot_q4_0_f32(q4_row.data(), x_f32.data(), len);
    const float fb_dot_q4 = dispatch_dot_q4_0_f32(q4_row.data(), x_f32.data(), len, SimdBackend::Scalar);
    if (ref_dot_q4 != fb_dot_q4) {
        std::cerr << "Scalar fallback for dot_q4_0_f32 did not match scalar reference!\n";
        return false;
    }

    // FP32 dot fallback
    const float ref_dot_f32 = scalar::dot_f32(w_f32.data(), x_f32.data(), len);
    const float fb_dot_f32 = dispatch_dot_f32(w_f32.data(), x_f32.data(), len, SimdBackend::Scalar);
    if (ref_dot_f32 != fb_dot_f32) {
        std::cerr << "Scalar fallback for dot_f32 did not match scalar reference!\n";
        return false;
    }

    // Q4_0 matmul fallback
    std::vector<float> ref_mat_q4(2, 0.0f);
    std::vector<float> fb_mat_q4(2, 0.0f);
    scalar::matmul_q4_0_f32(ref_mat_q4.data(), q4_row.data(), x_f32.data(), 2, len / 2);
    dispatch_matmul_q4_0_f32(fb_mat_q4.data(), q4_row.data(), x_f32.data(), 2, len / 2, SimdBackend::Scalar, nullptr);
    for (size_t i = 0; i < 2; ++i) {
        if (ref_mat_q4[i] != fb_mat_q4[i]) {
            std::cerr << "Scalar fallback for matmul_q4_0_f32 mismatch at row " << i << "\n";
            return false;
        }
    }

    // FP32 matmul fallback
    std::vector<float> ref_mat_f32(2, 0.0f);
    std::vector<float> fb_mat_f32(2, 0.0f);
    scalar::matmul_f32(ref_mat_f32.data(), w_f32.data(), x_f32.data(), 2, len / 2);
    dispatch_matmul_f32(fb_mat_f32.data(), w_f32.data(), x_f32.data(), 2, len / 2, SimdBackend::Scalar, nullptr);
    for (size_t i = 0; i < 2; ++i) {
        if (ref_mat_f32[i] != fb_mat_f32[i]) {
            std::cerr << "Scalar fallback for matmul_f32 mismatch at row " << i << "\n";
            return false;
        }
    }

    std::cout << "  - Scalar fallback verification: PASSED (exact bit-identity)\n";
    return true;
}

// ============================================================================
// Task B: Multi-length & Distribution Dot Product Tests
// ============================================================================
bool test_simd_dot_lengths() {
    const size_t test_lengths[] = {32, 64, 96, 128, 256, 512, 1024, 2048, 5632};

    for (size_t len : test_lengths) {
        auto weights = generate_random_floats(len, -2.0f, 2.0f, 1000 + static_cast<uint32_t>(len));
        auto x = generate_random_floats(len, -1.5f, 1.5f, 2000 + static_cast<uint32_t>(len));

        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        if (!quantize_row_q4_0(weights.data(), q4_row.data(), len).ok()) {
            std::cerr << "quantize_row_q4_0 failed for dot length " << len << "\n";
            return false;
        }

        const float scalar_dot = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float avx2_dot = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float dispatched_dot =
            dispatch_dot_q4_0_f32(q4_row.data(), x.data(), len, SimdBackend::Avx2);

        if (!check_close(scalar_dot, avx2_dot, 2e-4f, 2e-4f)) {
            std::cerr << "AVX2 dot_q4_0_f32 mismatch for length " << len
                      << " (scalar=" << scalar_dot << ", avx2=" << avx2_dot << ")\n";
            return false;
        }

        if (!check_close(avx2_dot, dispatched_dot, 1e-6f, 1e-6f)) {
            std::cerr << "Dispatch dot_q4_0_f32 mismatch for length " << len << "\n";
            return false;
        }

        // Also verify FP32 dot product across lengths
        const float scalar_f32 = scalar::dot_f32(weights.data(), x.data(), len);
        const float avx2_f32 = avx2::dot_f32(weights.data(), x.data(), len);
        if (!check_close(scalar_f32, avx2_f32, 1e-4f, 1e-4f)) {
            std::cerr << "AVX2 dot_f32 mismatch for length " << len << "\n";
            return false;
        }
    }
    return true;
}

bool test_simd_dot_distributions() {
    const size_t len = 256;

    // 1. Positive activations only
    {
        auto weights = generate_random_floats(len, -3.0f, 3.0f, 101);
        auto x = generate_random_floats(len, 0.1f, 5.0f, 102);
        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_row.data(), len);
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        if (!check_close(s, a, 2e-4f, 2e-4f)) {
            std::cerr << "Positive-only activations dot mismatch\n";
            return false;
        }
    }

    // 2. Negative activations only
    {
        auto weights = generate_random_floats(len, -3.0f, 3.0f, 201);
        auto x = generate_random_floats(len, -5.0f, -0.1f, 202);
        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_row.data(), len);
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        if (!check_close(s, a, 2e-4f, 2e-4f)) {
            std::cerr << "Negative-only activations dot mismatch\n";
            return false;
        }
    }

    // 3. All-zeros
    {
        std::vector<float> weights(len, 0.0f);
        std::vector<float> x(len, 1.25f);
        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_row.data(), len);
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        if (s != 0.0f || a != 0.0f) {
            std::cerr << "Zero-block dot product must be exactly 0.0f\n";
            return false;
        }
    }

    // 4. Large scale variation
    {
        auto weights = generate_random_floats(len, -1000.0f, 1000.0f, 301);
        auto x = generate_random_floats(len, -0.01f, 0.01f, 302);
        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_row.data(), len);
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        if (!check_close(s, a, 5e-4f, 5e-3f)) {
            std::cerr << "Large scale variation dot product mismatch\n";
            return false;
        }
    }

    // 5. Subnormal floats / near-zero activations
    {
        auto weights = generate_random_floats(len, -1.0f, 1.0f, 401);
        std::vector<float> x(len, 1e-35f);
        std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_row.data(), len);
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x.data(), len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x.data(), len);
        if (!check_close(s, a, 1e-4f, 1e-30f)) {
            std::cerr << "Subnormal float dot product mismatch\n";
            return false;
        }
    }

    return true;
}

// ============================================================================
// Task B: Misaligned Memory Buffer Loads
// ============================================================================
bool test_simd_dot_misaligned() {
    const size_t len = 128;
    // Allocate extra elements so we can test arbitrary pointer offsets
    std::vector<float> x_buf(len + 32, 0.5f);
    std::vector<float> w_buf(len, 1.0f);
    std::vector<BlockQ4_0> q4_row(len / kQ4_0BlockElements);
    quantize_row_q4_0(w_buf.data(), q4_row.data(), len);

    // Test non-32-byte aligned float pointers (offsets 1, 2, 3, 5, 7 floats)
    const size_t offsets[] = {0, 1, 2, 3, 5, 7};
    for (size_t off : offsets) {
        const float* x_ptr = x_buf.data() + off;
        const float s = scalar::dot_q4_0_f32(q4_row.data(), x_ptr, len);
        const float a = avx2::dot_q4_0_f32(q4_row.data(), x_ptr, len);
        if (!check_close(s, a, 2e-4f, 2e-4f)) {
            std::cerr << "Misaligned dot_q4_0_f32 mismatch at offset " << off << "\n";
            return false;
        }
    }
    return true;
}

// ============================================================================
// Task B: TinyLlama Projection Shapes & Numerical Error Metrics
// ============================================================================
bool test_simd_matmul_shapes_and_errors() {
    std::cout << "[TASK B] Testing numerical error metrics across TinyLlama projection shapes...\n";

    // Frozen TinyLlama-1.1B dimensions:
    // hidden_dim = 2048, intermediate_dim = 5632, kv_heads * head_dim = 4 * 64 = 256, vocab = 32000
    const struct ProjectionShape {
        const char* name;
        size_t out_dim;
        size_t in_dim;
    } projections[] = {
        {"Attention KV Slice", 256, 2048},
        {"Attention Q/K/V/O", 2048, 2048},
        {"SwiGLU Gate/Up", 5632, 2048},
        {"SwiGLU Down", 2048, 5632},
        {"LM Head Vocab", 32000, 2048},
    };

    std::cout << std::left << std::setw(22) << "Projection"
              << std::setw(15) << "Shape (MxK)"
              << std::setw(14) << "Max Abs Err"
              << std::setw(14) << "MAE"
              << std::setw(14) << "RMSE"
              << std::setw(14) << "Rel L2 Err"
              << "\n"
              << std::string(93, '-') << "\n";

    for (const auto& proj : projections) {
        const size_t total_weights = proj.out_dim * proj.in_dim;
        auto weights = generate_random_floats(total_weights, -1.0f, 1.0f, 700 + static_cast<uint32_t>(proj.out_dim));
        auto x = generate_random_floats(proj.in_dim, -1.0f, 1.0f, 800 + static_cast<uint32_t>(proj.in_dim));

        std::vector<BlockQ4_0> q4_weights(total_weights / kQ4_0BlockElements);
        if (!quantize_row_q4_0(weights.data(), q4_weights.data(), total_weights).ok()) {
            std::cerr << "quantize_row_q4_0 failed for " << proj.name << "\n";
            return false;
        }

        std::vector<float> ref_out(proj.out_dim, 0.0f);
        scalar::matmul_q4_0_f32(ref_out.data(), q4_weights.data(), x.data(), proj.out_dim, proj.in_dim);

        std::vector<float> avx2_out(proj.out_dim, 0.0f);
        avx2::matmul_q4_0_f32(avx2_out.data(), q4_weights.data(), x.data(), proj.out_dim, proj.in_dim);

        // Compute error metrics
        float max_abs_err = 0.0f;
        double sum_abs_err = 0.0;
        double sum_sq_err = 0.0;
        double sum_ref_sq = 0.0;

        for (size_t r = 0; r < proj.out_dim; ++r) {
            const float diff = std::fabs(ref_out[r] - avx2_out[r]);
            max_abs_err = std::max(max_abs_err, diff);
            sum_abs_err += diff;
            sum_sq_err += static_cast<double>(diff) * static_cast<double>(diff);
            sum_ref_sq += static_cast<double>(ref_out[r]) * static_cast<double>(ref_out[r]);
        }

        const double mae = sum_abs_err / static_cast<double>(proj.out_dim);
        const double rmse = std::sqrt(sum_sq_err / static_cast<double>(proj.out_dim));
        const double rel_err = std::sqrt(sum_sq_err) / std::sqrt(sum_ref_sq);

        const std::string shape_str = std::to_string(proj.out_dim) + "x" + std::to_string(proj.in_dim);
        std::cout << std::left << std::setw(22) << proj.name
                  << std::setw(15) << shape_str
                  << std::scientific << std::setprecision(3)
                  << std::setw(14) << max_abs_err
                  << std::setw(14) << mae
                  << std::setw(14) << rmse
                  << std::setw(14) << rel_err
                  << std::defaultfloat << "\n";

        // Verification bounds: FMA vs scalar mul+add contraction bound is well within 3e-4
        if (max_abs_err > 3e-4f) {
            std::cerr << "Max error exceeded 3e-4 on " << proj.name << "\n";
            return false;
        }
        if (rel_err > 1e-4) {
            std::cerr << "Relative error exceeded 1e-4 on " << proj.name << "\n";
            return false;
        }
    }
    std::cout << "\n";
    return true;
}

// ============================================================================
// Task B / C: Multithreaded Matmul Verification
// ============================================================================
bool test_simd_matmul_threading() {
    const struct MatShape {
        size_t out_dim;
        size_t in_dim;
    } shapes[] = {
        {8, 64},        // Small baseline
        {64, 2048},     // Attention QKV head slice
        {256, 2048},    // Multi-head projection
        {2048, 2048},   // Attention projection
    };

    const size_t thread_counts[] = {1, 2, 4, 8, 16};

    for (const auto& sh : shapes) {
        const size_t total_weights = sh.out_dim * sh.in_dim;
        auto weights = generate_random_floats(total_weights, -1.5f, 1.5f, 401);
        auto x = generate_random_floats(sh.in_dim, -1.0f, 1.0f, 402);

        std::vector<BlockQ4_0> q4_weights(total_weights / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_weights.data(), total_weights);

        std::vector<float> ref_out(sh.out_dim, 0.0f);
        scalar::matmul_q4_0_f32(ref_out.data(), q4_weights.data(), x.data(), sh.out_dim, sh.in_dim);

        for (size_t num_threads : thread_counts) {
            ThreadPool pool(num_threads);
            std::vector<float> pool_out(sh.out_dim, 0.0f);

            dispatch_matmul_q4_0_f32(
                pool_out.data(), q4_weights.data(), x.data(), sh.out_dim, sh.in_dim,
                SimdBackend::Avx2, &pool);

            for (size_t r = 0; r < sh.out_dim; ++r) {
                if (!check_close(ref_out[r], pool_out[r], 2e-4f, 2e-4f)) {
                    std::cerr << "Dispatched ThreadPool(" << num_threads << ") matmul mismatch at row "
                              << r << " for shape (" << sh.out_dim << "x" << sh.in_dim << ")\n";
                    return false;
                }
            }
        }
    }
    return true;
}

// ============================================================================
// Task B / SIMD: Odd Rows and Boundary Tails (2-Row Tiled Matmul Verification)
// ============================================================================
bool test_simd_matmul_odd_rows_and_tails() {
    std::cout << "[SIMD OPTS] Verifying 2-row tiled matmul on odd row counts and boundary tails...\n";

    const size_t odd_out_dims[] = {1, 3, 5, 7, 9, 13, 31, 65, 127, 255};
    const size_t in_dims[] = {32, 64, 128, 2048};

    for (size_t out_dim : odd_out_dims) {
        for (size_t in_dim : in_dims) {
            const size_t total_weights = out_dim * in_dim;
            auto weights = generate_random_floats(total_weights, -1.0f, 1.0f, 999 + static_cast<uint32_t>(out_dim + in_dim));
            auto x = generate_random_floats(in_dim, -1.0f, 1.0f, 888 + static_cast<uint32_t>(out_dim + in_dim));

            // 1. Q4_0 2-row tiled matmul
            std::vector<BlockQ4_0> q4_weights(total_weights / kQ4_0BlockElements);
            quantize_row_q4_0(weights.data(), q4_weights.data(), total_weights);

            std::vector<float> ref_q4(out_dim, 0.0f);
            scalar::matmul_q4_0_f32(ref_q4.data(), q4_weights.data(), x.data(), out_dim, in_dim);

            std::vector<float> avx2_q4(out_dim, 0.0f);
            avx2::matmul_q4_0_f32(avx2_q4.data(), q4_weights.data(), x.data(), out_dim, in_dim);

            for (size_t r = 0; r < out_dim; ++r) {
                if (!check_close(ref_q4[r], avx2_q4[r], 2e-4f, 2e-4f)) {
                    std::cerr << "AVX2 2-row tiled matmul_q4_0_f32 mismatch at row " << r
                              << " for odd shape (" << out_dim << "x" << in_dim << ")\n";
                    return false;
                }
            }

            // Also test dispatched multithreaded execution across odd rows
            ThreadPool pool(4);
            std::vector<float> dispatched_q4(out_dim, 0.0f);
            dispatch_matmul_q4_0_f32(dispatched_q4.data(), q4_weights.data(), x.data(),
                                     out_dim, in_dim, SimdBackend::Avx2, &pool);
            for (size_t r = 0; r < out_dim; ++r) {
                if (!check_close(ref_q4[r], dispatched_q4[r], 2e-4f, 2e-4f)) {
                    std::cerr << "Dispatched 2-row tiled matmul_q4_0_f32 mismatch at row " << r
                              << " for odd shape (" << out_dim << "x" << in_dim << ")\n";
                    return false;
                }
            }

            // 2. FP32 2-row tiled matmul
            std::vector<float> ref_f32(out_dim, 0.0f);
            scalar::matmul_f32(ref_f32.data(), weights.data(), x.data(), out_dim, in_dim);

            std::vector<float> avx2_f32(out_dim, 0.0f);
            avx2::matmul_f32(avx2_f32.data(), weights.data(), x.data(), out_dim, in_dim);

            for (size_t r = 0; r < out_dim; ++r) {
                if (!check_close(ref_f32[r], avx2_f32[r], 1e-4f, 1e-4f)) {
                    std::cerr << "AVX2 2-row tiled matmul_f32 mismatch at row " << r
                              << " for odd shape (" << out_dim << "x" << in_dim << ")\n";
                    return false;
                }
            }
        }
    }
    std::cout << "  - Odd rows and boundary tails (Q4_0 & FP32): PASSED\n";
    return true;
}

// ============================================================================
// Task C: Persistent ThreadPool Lifecycle, Distribution, Stress & Exceptions
// ============================================================================
bool test_thread_pool_verification() {
    std::cout << "[TASK C] Verifying ThreadPool lifecycle, task distribution, stress, and exceptions...\n";

    // 1. Worker Lifecycle (1, 2, 4, 8, 16 threads)
    const size_t thread_configs[] = {1, 2, 4, 8, 16};
    for (size_t tc : thread_configs) {
        ThreadPool pool(tc);
        if (pool.num_threads() != tc) {
            std::cerr << "ThreadPool num_threads mismatch: expected " << tc << ", got " << pool.num_threads() << "\n";
            return false;
        }
    }

    // 2. Work Distribution across various task counts: 0, 1, 2, 7, 16, 64, 1024, 65536
    const size_t task_counts[] = {0, 1, 2, 7, 16, 64, 1024, 65536};
    ThreadPool pool8(8);

    for (size_t N : task_counts) {
        if (N == 0) {
            bool called = false;
            pool8.parallel_for(0, [&](size_t, size_t) { called = true; });
            if (called) {
                std::cerr << "parallel_for with 0 tasks should not invoke task_fn!\n";
                return false;
            }
            continue;
        }

        std::vector<std::atomic<uint32_t>> visit_count(N);
        for (size_t i = 0; i < N; ++i) {
            visit_count[i].store(0, std::memory_order_relaxed);
        }

        pool8.parallel_for(N, [&](size_t task_idx, size_t thread_idx) {
            if (thread_idx >= 8) {
                std::cerr << "Invalid thread_idx " << thread_idx << " in pool of 8 threads\n";
            }
            visit_count[task_idx].fetch_add(1, std::memory_order_relaxed);
        });

        for (size_t i = 0; i < N; ++i) {
            if (visit_count[i].load(std::memory_order_relaxed) != 1) {
                std::cerr << "Task " << i << " of " << N << " executed "
                          << visit_count[i].load() << " times (expected exactly 1)\n";
                return false;
            }
        }
    }
    std::cout << "  - Task distribution exactness: PASSED (all task counts up to 65536)\n";

    // 3. Stress Test: 1,000 rapid successive parallel_for invocations
    {
        const size_t iters = 1000;
        const size_t n_tasks = 32;
        std::atomic<uint64_t> accumulator{0};

        for (size_t iter = 0; iter < iters; ++iter) {
            pool8.parallel_for(n_tasks, [&](size_t /*task_idx*/, size_t /*thread_idx*/) {
                accumulator.fetch_add(1, std::memory_order_relaxed);
            });
        }

        const uint64_t expected_total = static_cast<uint64_t>(iters) * n_tasks;
        if (accumulator.load(std::memory_order_relaxed) != expected_total) {
            std::cerr << "Stress test count mismatch: expected " << expected_total
                      << ", got " << accumulator.load() << "\n";
            return false;
        }
        std::cout << "  - Stress test (1,000 rapid cycles, 32,000 tasks): PASSED\n";
    }

    // 4. Exception Safety and Propagation
    {
        bool exception_caught = false;
        try {
            pool8.parallel_for(16, [&](size_t task_idx, size_t /*thread_idx*/) {
                if (task_idx == 5) {
                    throw std::runtime_error("simulated_pool_exception");
                }
            });
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) == "simulated_pool_exception") {
                exception_caught = true;
            }
        }

        if (!exception_caught) {
            std::cerr << "Exception thrown in task_fn was not caught by caller!\n";
            return false;
        }

        // Verify pool recovers and functions normally after an exception
        std::atomic<uint32_t> post_count{0};
        pool8.parallel_for(10, [&](size_t, size_t) {
            post_count.fetch_add(1, std::memory_order_relaxed);
        });
        if (post_count.load() != 10) {
            std::cerr << "ThreadPool failed to recover after exception!\n";
            return false;
        }
        std::cout << "  - Exception safety & recovery: PASSED\n";
    }

    return true;
}

// ============================================================================
// Task E: Transformer Integration & Mixed-Precision Verification
// ============================================================================
bool test_transformer_integration_contracts() {
    std::cout << "[TASK E] Verifying Transformer integration contracts & mixed-precision flow...\n";

    // 1. Zero-copy TensorView Q4_0 block accessor contract
    {
        const size_t total_elements = 2048 * 2048;
        const size_t total_blocks = total_elements / kQ4_0BlockElements;
        std::vector<BlockQ4_0> mock_weight_storage(total_blocks);
        mock_weight_storage[0].d = fp32_to_fp16(1.5f);
        mock_weight_storage[0].qs[0] = pack_q4_0_nibbles(2, 10);

        TensorDesc desc;
        desc.name = "model.layers.0.self_attn.q_proj.weight";
        desc.dtype = DataType::Q4_0;
        desc.ndim = 2;
        desc.shape = {2048, 2048, 0, 0};
        desc.payload_offset = 0;
        desc.payload_nbytes = total_blocks * sizeof(BlockQ4_0);

        TensorView view(desc, reinterpret_cast<const uint8_t*>(mock_weight_storage.data()));

        if (view.dtype() != DataType::Q4_0) {
            std::cerr << "TensorView dtype is not Q4_0\n";
            return false;
        }
        if (view.num_elements() != total_elements) {
            std::cerr << "TensorView num_elements mismatch\n";
            return false;
        }
        if (view.num_blocks() != total_blocks) {
            std::cerr << "TensorView num_blocks mismatch\n";
            return false;
        }
        if (view.data_q4_0() != mock_weight_storage.data()) {
            std::cerr << "TensorView::data_q4_0() did not return zero-copy pointer!\n";
            return false;
        }
        if (view.data_f32() != nullptr) {
            std::cerr << "TensorView::data_f32() must return nullptr for Q4_0 tensor!\n";
            return false;
        }
        // Verify reading through zero-copy accessor
        const BlockQ4_0* b0 = view.data_q4_0();
        if (fp16_to_fp32(b0->d) != 1.5f || unpack_q4_0_low_nibble(b0->qs[0]) != 2) {
            std::cerr << "Data accessed through TensorView::data_q4_0() was corrupted\n";
            return false;
        }
        std::cout << "  - Zero-copy TensorView::data_q4_0() access: PASSED\n";
    }

    // 2. Simulated Mixed-Precision SwiGLU MLP Projection:
    //    x (FP32 2048) -> Gate (Q4_0 5632x2048) -> Up (Q4_0 5632x2048) -> SwiGLU -> Down (Q4_0 2048x5632) -> out (FP32 2048)
    {
        const size_t hidden_dim = 2048;
        const size_t intermediate_dim = 5632;

        auto x = generate_random_floats(hidden_dim, -1.0f, 1.0f, 901);

        const size_t gate_up_elems = intermediate_dim * hidden_dim;
        const size_t down_elems = hidden_dim * intermediate_dim;

        auto gate_f32 = generate_random_floats(gate_up_elems, -0.5f, 0.5f, 902);
        auto up_f32 = generate_random_floats(gate_up_elems, -0.5f, 0.5f, 903);
        auto down_f32 = generate_random_floats(down_elems, -0.5f, 0.5f, 904);

        std::vector<BlockQ4_0> gate_q4(gate_up_elems / kQ4_0BlockElements);
        std::vector<BlockQ4_0> up_q4(gate_up_elems / kQ4_0BlockElements);
        std::vector<BlockQ4_0> down_q4(down_elems / kQ4_0BlockElements);

        quantize_row_q4_0(gate_f32.data(), gate_q4.data(), gate_up_elems);
        quantize_row_q4_0(up_f32.data(), up_q4.data(), gate_up_elems);
        quantize_row_q4_0(down_f32.data(), down_q4.data(), down_elems);

        // Reference Scalar Execution
        std::vector<float> ref_gate_act(intermediate_dim);
        std::vector<float> ref_up_act(intermediate_dim);
        std::vector<float> ref_swiglu_act(intermediate_dim);
        std::vector<float> ref_down_act(hidden_dim);

        scalar::matmul_q4_0_f32(ref_gate_act.data(), gate_q4.data(), x.data(), intermediate_dim, hidden_dim);
        scalar::matmul_q4_0_f32(ref_up_act.data(), up_q4.data(), x.data(), intermediate_dim, hidden_dim);
        scalar::swiglu_f32(ref_swiglu_act.data(), ref_gate_act.data(), ref_up_act.data(), intermediate_dim);
        scalar::matmul_q4_0_f32(ref_down_act.data(), down_q4.data(), ref_swiglu_act.data(), hidden_dim, intermediate_dim);

        // Dispatched SIMD + Multithreaded Execution (8 threads)
        ThreadPool pool(8);
        std::vector<float> sim_gate_act(intermediate_dim);
        std::vector<float> sim_up_act(intermediate_dim);
        std::vector<float> sim_swiglu_act(intermediate_dim);
        std::vector<float> sim_down_act(hidden_dim);

        dispatch_matmul_q4_0_f32(sim_gate_act.data(), gate_q4.data(), x.data(), intermediate_dim, hidden_dim, SimdBackend::Avx2, &pool);
        dispatch_matmul_q4_0_f32(sim_up_act.data(), up_q4.data(), x.data(), intermediate_dim, hidden_dim, SimdBackend::Avx2, &pool);
        scalar::swiglu_f32(sim_swiglu_act.data(), sim_gate_act.data(), sim_up_act.data(), intermediate_dim);
        dispatch_matmul_q4_0_f32(sim_down_act.data(), down_q4.data(), sim_swiglu_act.data(), hidden_dim, intermediate_dim, SimdBackend::Avx2, &pool);

        float max_diff = 0.0f;
        for (size_t i = 0; i < hidden_dim; ++i) {
            max_diff = std::max(max_diff, std::fabs(ref_down_act[i] - sim_down_act[i]));
            if (!check_close(ref_down_act[i], sim_down_act[i], 5e-3f, 2e-3f)) {
                std::cerr << "Mixed-precision SwiGLU MLP mismatch at index " << i
                          << " (ref=" << ref_down_act[i] << ", sim=" << sim_down_act[i] << ")\n";
                return false;
            }
        }
        std::cout << "  - Mixed-precision SwiGLU MLP flow: PASSED (max diff = "
                  << std::scientific << std::setprecision(3) << max_diff << std::defaultfloat << ")\n\n";
    }

    return true;
}

// ============================================================================
// Task D: Systematic Microbenchmarking Suite
// ============================================================================
void run_performance_benchmarks() {
    std::cout << "\n========================================================================================\n"
              << "            MODULE I — CPU SIMD & MULTITHREADING BENCHMARK REPORT                      \n"
              << "========================================================================================\n";

    // 1. Dot product microbenchmark: 2048 elements (TinyLlama hidden_dim)
    const size_t dot_len = 2048;
    const size_t dot_iters = 50000;
    auto dot_w = generate_random_floats(dot_len, -1.0f, 1.0f, 501);
    auto dot_x = generate_random_floats(dot_len, -1.0f, 1.0f, 502);
    std::vector<BlockQ4_0> dot_q4(dot_len / kQ4_0BlockElements);
    quantize_row_q4_0(dot_w.data(), dot_q4.data(), dot_len);

    // Warm-up
    volatile float dummy = 0.0f;
    for (size_t i = 0; i < 1000; ++i) {
        dummy += scalar::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
        dummy += avx2::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
    }

    // Benchmark scalar dot
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < dot_iters; ++i) {
        dummy += scalar::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double scalar_dot_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Benchmark AVX2 dot
    const auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < dot_iters; ++i) {
        dummy += avx2::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double avx2_dot_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const double dot_speedup = scalar_dot_ms / avx2_dot_ms;
    const double dot_gflops =
        (2.0 * static_cast<double>(dot_len) * static_cast<double>(dot_iters)) / (avx2_dot_ms * 1e6);

    std::cout << "[Dot Product Q4_0 (len = " << dot_len << ", " << dot_iters << " iterations)]\n"
              << "  Scalar Time : " << std::fixed << std::setprecision(2) << scalar_dot_ms << " ms\n"
              << "  AVX2 Time   : " << std::fixed << std::setprecision(2) << avx2_dot_ms << " ms\n"
              << "  Speedup     : " << std::fixed << std::setprecision(2) << dot_speedup << "x\n"
              << "  Throughput  : " << std::fixed << std::setprecision(2) << dot_gflops << " GFLOPS\n\n";

    // 2. FP32 Matrix-Vector Multiplication: 2048 x 2048
    {
        const size_t rows = 2048, cols = 2048, iters = 20;
        const size_t n_weights = rows * cols;
        auto w_f32 = generate_random_floats(n_weights, -0.5f, 0.5f, 601);
        auto x_f32 = generate_random_floats(cols, -0.5f, 0.5f, 602);
        std::vector<float> y(rows, 0.0f);

        // Warm-up
        scalar::matmul_f32(y.data(), w_f32.data(), x_f32.data(), rows, cols);
        avx2::matmul_f32(y.data(), w_f32.data(), x_f32.data(), rows, cols);

        // Scalar
        const auto fp_s0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            scalar::matmul_f32(y.data(), w_f32.data(), x_f32.data(), rows, cols);
        }
        const auto fp_s1 = std::chrono::steady_clock::now();
        const double scalar_fp32_ms = std::chrono::duration<double, std::milli>(fp_s1 - fp_s0).count() / iters;

        // AVX2 1T
        const auto fp_a0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            avx2::matmul_f32(y.data(), w_f32.data(), x_f32.data(), rows, cols);
        }
        const auto fp_a1 = std::chrono::steady_clock::now();
        const double avx2_fp32_1t_ms = std::chrono::duration<double, std::milli>(fp_a1 - fp_a0).count() / iters;

        // AVX2 8T
        ThreadPool pool(8);
        const auto fp_m0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            dispatch_matmul_f32(y.data(), w_f32.data(), x_f32.data(), rows, cols, SimdBackend::Avx2, &pool);
        }
        const auto fp_m1 = std::chrono::steady_clock::now();
        const double avx2_fp32_8t_ms = std::chrono::duration<double, std::milli>(fp_m1 - fp_m0).count() / iters;

        std::cout << "[Matmul FP32 (2048 x 2048, Attention Projection)]\n"
                  << "  Scalar (1T) : " << std::fixed << std::setprecision(2) << scalar_fp32_ms << " ms/iter\n"
                  << "  AVX2   (1T) : " << std::fixed << std::setprecision(2) << avx2_fp32_1t_ms << " ms/iter"
                  << "  (Speedup: " << (scalar_fp32_ms / avx2_fp32_1t_ms) << "x)\n"
                  << "  AVX2   (8T) : " << std::fixed << std::setprecision(2) << avx2_fp32_8t_ms << " ms/iter"
                  << "  (Speedup: " << (scalar_fp32_ms / avx2_fp32_8t_ms) << "x)\n\n";
    }

    // 3. Q4_0 Matrix-Vector Multiplications across all TinyLlama projections
    const struct BenchShape {
        const char* name;
        size_t rows;
        size_t cols;
        size_t iters;
    } bench_shapes[] = {
        {"Attention KV Slice", 256, 2048, 100},
        {"Attention Q/K/V/O", 2048, 2048, 50},
        {"SwiGLU Gate/Up", 5632, 2048, 25},
        {"SwiGLU Down", 2048, 5632, 25},
        {"LM Head Vocab", 32000, 2048, 10},
    };

    const size_t thread_counts[] = {1, 2, 4, 8, 16};

    for (const auto& bs : bench_shapes) {
        const size_t total_weights = bs.rows * bs.cols;
        auto w = generate_random_floats(total_weights, -1.0f, 1.0f, 801);
        auto x = generate_random_floats(bs.cols, -1.0f, 1.0f, 802);
        std::vector<BlockQ4_0> q4_w(total_weights / kQ4_0BlockElements);
        quantize_row_q4_0(w.data(), q4_w.data(), total_weights);

        std::vector<float> y(bs.rows, 0.0f);

        // Warm-up
        scalar::matmul_q4_0_f32(y.data(), q4_w.data(), x.data(), bs.rows, bs.cols);
        avx2::matmul_q4_0_f32(y.data(), q4_w.data(), x.data(), bs.rows, bs.cols);

        // Benchmark Scalar 1T
        const auto s_t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < bs.iters; ++i) {
            scalar::matmul_q4_0_f32(y.data(), q4_w.data(), x.data(), bs.rows, bs.cols);
        }
        const auto s_t1 = std::chrono::steady_clock::now();
        const double scalar_ms = std::chrono::duration<double, std::milli>(s_t1 - s_t0).count() / bs.iters;

        std::cout << "[Matmul Q4_0: " << bs.name << " (" << bs.rows << " x " << bs.cols << ")]\n"
                  << "  Scalar (1T) : " << std::fixed << std::setprecision(2) << scalar_ms << " ms/iter\n";

        // Benchmark AVX2 across thread counts
        for (size_t tc : thread_counts) {
            std::vector<double> sample_times;
            sample_times.reserve(bs.iters);

            if (tc == 1) {
                for (size_t i = 0; i < bs.iters; ++i) {
                    const auto a0 = std::chrono::steady_clock::now();
                    avx2::matmul_q4_0_f32(y.data(), q4_w.data(), x.data(), bs.rows, bs.cols);
                    const auto a1 = std::chrono::steady_clock::now();
                    sample_times.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
                }
            } else {
                ThreadPool pool(tc);
                for (size_t i = 0; i < bs.iters; ++i) {
                    const auto a0 = std::chrono::steady_clock::now();
                    dispatch_matmul_q4_0_f32(y.data(), q4_w.data(), x.data(), bs.rows, bs.cols, SimdBackend::Avx2, &pool);
                    const auto a1 = std::chrono::steady_clock::now();
                    sample_times.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
                }
            }

            // Compute median latency over iterations
            std::sort(sample_times.begin(), sample_times.end());
            const double median_ms = sample_times[sample_times.size() / 2];
            const double speedup = scalar_ms / median_ms;
            const double gflops = (2.0 * static_cast<double>(bs.rows) * static_cast<double>(bs.cols)) / (median_ms * 1e6);
            const double weight_mb = (q4_w.size() * sizeof(BlockQ4_0)) / (1024.0 * 1024.0);
            const double eff_bandwidth_gbps = (weight_mb / (median_ms / 1000.0)) / 1024.0;

            std::cout << "  AVX2 (" << std::setw(2) << tc << "T) : "
                      << std::fixed << std::setprecision(2) << median_ms << " ms"
                      << "  (Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x"
                      << ", GFLOPS: " << std::setw(6) << std::fixed << std::setprecision(2) << gflops
                      << ", Bandwidth: " << std::setw(5) << std::fixed << std::setprecision(2) << eff_bandwidth_gbps << " GB/s)\n";
        }
        std::cout << "\n";
    }

    std::cout << "========================================================================================\n\n";
    (void)dummy;
}

}  // namespace

bool run_kernels_simd_tests() {
    // 1. FP32 SIMD Matmul test (baseline parity check)
    const size_t in_dim = 64;
    const size_t out_dim = 8;

    std::vector<float> weight(out_dim * in_dim);
    std::vector<float> x(in_dim);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    for (size_t i = 0; i < in_dim; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f;
    }

    std::vector<float> ref_out(out_dim, 0.0f);
    scalar::matmul_f32(ref_out.data(), weight.data(), x.data(), out_dim, in_dim);

    const SimdBackend best = detect_best_backend();
    ThreadPool pool(2);
    std::vector<float> simd_out(out_dim, 0.0f);
    dispatch_matmul_f32(simd_out.data(), weight.data(), x.data(), out_dim, in_dim, best, &pool);

    for (size_t r = 0; r < out_dim; ++r) {
        if (std::fabs(ref_out[r] - simd_out[r]) > 1e-5f) {
            std::cerr << "SIMD matmul_f32 mismatch at row " << r << "\n";
            return false;
        }
    }

    // Task A: Native AVX2 & CPUID verification
    if (!test_native_cpu_features()) {
        return false;
    }

    // Task B: Multi-length and distribution dot product tests
    if (!test_simd_dot_lengths()) {
        return false;
    }
    if (!test_simd_dot_distributions()) {
        return false;
    }
    if (!test_simd_dot_misaligned()) {
        return false;
    }

    // Task B: TinyLlama projection shapes & numerical error metrics
    if (!test_simd_matmul_shapes_and_errors()) {
        return false;
    }

    // Task B / C: Multithreaded matmul verification
    if (!test_simd_matmul_threading()) {
        return false;
    }
    if (!test_simd_matmul_odd_rows_and_tails()) {
        return false;
    }

    // Task C: ThreadPool lifecycle, distribution, stress & exceptions
    if (!test_thread_pool_verification()) {
        return false;
    }

    // Task E: Transformer integration contracts & mixed-precision flow
    if (!test_transformer_integration_contracts()) {
        return false;
    }

    // Task D: Systematic microbenchmarking
    run_performance_benchmarks();

    return true;
}

}  // namespace inference_engine::testing
