#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "inference_engine/kernels.hpp"

namespace inference_engine::testing {

namespace {

// Helper to generate pseudorandom float vector in [low, high]
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

    return true;
}

bool test_simd_matmul_threading() {
    // Test dimensions representing TinyLlama projection shapes
    const struct MatShape {
        size_t out_dim;
        size_t in_dim;
    } shapes[] = {
        {8, 64},        // Small baseline
        {32, 256},      // Medium
        {64, 2048},     // Attention QKV head slice
        {256, 2048},    // Multi-head projection
    };

    const size_t thread_counts[] = {1, 2, 4, 8};

    for (const auto& sh : shapes) {
        const size_t total_weights = sh.out_dim * sh.in_dim;
        auto weights = generate_random_floats(total_weights, -1.5f, 1.5f, 401);
        auto x = generate_random_floats(sh.in_dim, -1.0f, 1.0f, 402);

        std::vector<BlockQ4_0> q4_weights(total_weights / kQ4_0BlockElements);
        quantize_row_q4_0(weights.data(), q4_weights.data(), total_weights);

        // Reference scalar matmul
        std::vector<float> ref_out(sh.out_dim, 0.0f);
        scalar::matmul_q4_0_f32(ref_out.data(), q4_weights.data(), x.data(), sh.out_dim, sh.in_dim);

        // AVX2 unthreaded matmul
        std::vector<float> avx2_out(sh.out_dim, 0.0f);
        avx2::matmul_q4_0_f32(avx2_out.data(), q4_weights.data(), x.data(), sh.out_dim, sh.in_dim);

        for (size_t r = 0; r < sh.out_dim; ++r) {
            if (!check_close(ref_out[r], avx2_out[r], 2e-4f, 2e-4f)) {
                std::cerr << "AVX2 matmul_q4_0_f32 mismatch at row " << r << " for shape ("
                          << sh.out_dim << "x" << sh.in_dim << ")\n";
                return false;
            }
        }

        // Test multithreaded dispatch across various thread counts
        for (size_t num_threads : thread_counts) {
            ThreadPool pool(num_threads);
            std::vector<float> pool_out(sh.out_dim, 0.0f);

            dispatch_matmul_q4_0_f32(
                pool_out.data(), q4_weights.data(), x.data(), sh.out_dim, sh.in_dim,
                SimdBackend::Avx2, &pool);

            for (size_t r = 0; r < sh.out_dim; ++r) {
                if (!check_close(ref_out[r], pool_out[r], 2e-4f, 2e-4f)) {
                    std::cerr << "Dispatched ThreadPool(" << num_threads << ") matmul mismatch at row "
                              << r << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

void run_performance_benchmarks() {
    std::cout << "\n========================================================================\n"
              << "          MODULE I — CPU SIMD & MULTITHREADING BENCHMARK REPORT         \n"
              << "========================================================================\n";

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

    // 2. Matrix-Vector Multiplication: 2048 x 2048 (TinyLlama attention projection)
    const size_t mat_rows = 2048;
    const size_t mat_cols = 2048;
    const size_t mat_iters = 50;
    const size_t total_mat_weights = mat_rows * mat_cols;
    auto mat_w = generate_random_floats(total_mat_weights, -1.0f, 1.0f, 601);
    auto mat_x = generate_random_floats(mat_cols, -1.0f, 1.0f, 602);
    std::vector<BlockQ4_0> mat_q4(total_mat_weights / kQ4_0BlockElements);
    quantize_row_q4_0(mat_w.data(), mat_q4.data(), total_mat_weights);

    std::vector<float> out_vec(mat_rows, 0.0f);

    // Warm-up
    scalar::matmul_q4_0_f32(out_vec.data(), mat_q4.data(), mat_x.data(), mat_rows, mat_cols);
    avx2::matmul_q4_0_f32(out_vec.data(), mat_q4.data(), mat_x.data(), mat_rows, mat_cols);

    // Scalar matmul
    const auto m_t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < mat_iters; ++i) {
        scalar::matmul_q4_0_f32(out_vec.data(), mat_q4.data(), mat_x.data(), mat_rows, mat_cols);
    }
    const auto m_t1 = std::chrono::steady_clock::now();
    const double scalar_mat_ms = std::chrono::duration<double, std::milli>(m_t1 - m_t0).count() / mat_iters;

    // AVX2 single-thread matmul
    const auto m_t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < mat_iters; ++i) {
        avx2::matmul_q4_0_f32(out_vec.data(), mat_q4.data(), mat_x.data(), mat_rows, mat_cols);
    }
    const auto m_t3 = std::chrono::steady_clock::now();
    const double avx2_st_ms = std::chrono::duration<double, std::milli>(m_t3 - m_t2).count() / mat_iters;

    std::cout << "[Matmul Q4_0 (" << mat_rows << " x " << mat_cols << ")]\n"
              << "  Scalar (1 thread) : " << std::fixed << std::setprecision(2) << scalar_mat_ms << " ms/iter\n"
              << "  AVX2   (1 thread) : " << std::fixed << std::setprecision(2) << avx2_st_ms << " ms/iter"
              << "  (Speedup: " << std::fixed << std::setprecision(2) << (scalar_mat_ms / avx2_st_ms) << "x)\n";

    // AVX2 multi-thread scaling
    const size_t threads_to_test[] = {2, 4, 8, 16};
    for (size_t nt : threads_to_test) {
        ThreadPool pool(nt);
        const auto mt_0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < mat_iters; ++i) {
            dispatch_matmul_q4_0_f32(
                out_vec.data(), mat_q4.data(), mat_x.data(), mat_rows, mat_cols,
                SimdBackend::Avx2, &pool);
        }
        const auto mt_1 = std::chrono::steady_clock::now();
        const double mt_ms = std::chrono::duration<double, std::milli>(mt_1 - mt_0).count() / mat_iters;
        const double mt_speedup = scalar_mat_ms / mt_ms;
        const double weight_mb = (mat_q4.size() * sizeof(BlockQ4_0)) / (1024.0 * 1024.0);
        const double eff_bandwidth_gbps = (weight_mb / (mt_ms / 1000.0)) / 1024.0;

        std::cout << "  AVX2   (" << std::setw(2) << nt << " threads): "
                  << std::fixed << std::setprecision(2) << mt_ms << " ms/iter"
                  << "  (Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << mt_speedup << "x"
                  << ", Effective Bandwidth: " << std::fixed << std::setprecision(2) << eff_bandwidth_gbps << " GB/s)\n";
    }

    std::cout << "========================================================================\n\n";
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

    // 2. Comprehensive SIMD tests for Module I Phase 7
    if (!test_simd_dot_lengths()) {
        return false;
    }
    if (!test_simd_dot_distributions()) {
        return false;
    }
    if (!test_simd_matmul_threading()) {
        return false;
    }

    // 3. Performance benchmarking
    run_performance_benchmarks();

    return true;
}

}  // namespace inference_engine::testing
