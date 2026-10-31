#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/kernels.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/thread_pool.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct BenchOptions {
    size_t iterations = 50;
    size_t warmup = 10;
    std::vector<size_t> thread_counts = {1, 2, 4, 8, 16};
    std::string csv_path = "";
};

struct TimingStats {
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
};

struct ErrorStats {
    float max_abs_err = 0.0f;
    double mae = 0.0;
    double rmse = 0.0;
    double rel_l2_err = 0.0;
};

std::vector<float> generate_random_floats(size_t n, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

TimingStats compute_stats(std::vector<double>& times_ms, double ops, double bytes, double baseline_median_ms = 0.0, size_t num_threads = 1) {
    TimingStats stats;
    if (times_ms.empty()) return stats;

    std::sort(times_ms.begin(), times_ms.end());
    stats.min_ms = times_ms.front();
    stats.max_ms = times_ms.back();
    stats.median_ms = times_ms[times_ms.size() / 2];

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.mean_ms = sum / static_cast<double>(times_ms.size());

    double sq_diff = 0.0;
    for (double t : times_ms) {
        sq_diff += (t - stats.mean_ms) * (t - stats.mean_ms);
    }
    stats.stddev_ms = std::sqrt(sq_diff / static_cast<double>(times_ms.size()));

    const double median_sec = stats.median_ms / 1000.0;
    stats.gflops = (ops / (median_sec * 1e9));
    stats.bandwidth_gbps = (bytes / (median_sec * 1e9));

    if (baseline_median_ms > 0.0) {
        stats.speedup = baseline_median_ms / stats.median_ms;
        if (num_threads > 0) {
            stats.efficiency = stats.speedup / static_cast<double>(num_threads);
        }
    } else {
        stats.speedup = 1.0;
        stats.efficiency = 1.0;
    }
    return stats;
}

ErrorStats compute_error(const std::vector<float>& ref, const std::vector<float>& act) {
    ErrorStats err;
    if (ref.size() != act.size() || ref.empty()) return err;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_ref_sq = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - act[i]);
        err.max_abs_err = std::max(err.max_abs_err, diff);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
        sum_ref_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }

    err.mae = sum_abs / static_cast<double>(ref.size());
    err.rmse = std::sqrt(sum_sq / static_cast<double>(ref.size()));
    err.rel_l2_err = std::sqrt(sum_sq) / std::sqrt(sum_ref_sq);
    return err;
}

struct ShapeDef {
    std::string name;
    size_t rows;
    size_t cols;
};

const std::vector<ShapeDef> kTinyLlamaShapes = {
    {"Attention Q projection", 2048, 2048},
    {"Attention K projection", 256, 2048},
    {"Attention V projection", 256, 2048},
    {"Attention O projection", 2048, 2048},
    {"SwiGLU Gate projection", 5632, 2048},
    {"SwiGLU Up projection", 5632, 2048},
    {"SwiGLU Down projection", 2048, 5632},
    {"LM Head projection", 32000, 2048}
};

void print_header() {
    std::cout << "\n========================================================================================================================\n"
              << "                      BIOS V2 — PS08: CPU KERNEL BENCHMARK & PROFILING HARNESS                                         \n"
              << "========================================================================================================================\n";
    std::cout << "Target: TinyLlama-1.1B-Chat-v1.0 (22 Layers, hidden=2048, intermediate=5632, vocab=32000)\n";
    std::cout << "Detected SIMD Backend: " << inference_engine::backend_name(inference_engine::detect_best_backend()) << "\n";
    std::cout << "AVX2 Available: " << (inference_engine::is_backend_available(inference_engine::SimdBackend::Avx2) ? "YES" : "NO") << "\n";
    std::cout << "NEON Available: " << (inference_engine::is_backend_available(inference_engine::SimdBackend::Neon) ? "YES" : "NO") << "\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    BenchOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            opts.iterations = std::stoul(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = std::stoul(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            opts.csv_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: inference_engine_bench [options]\n"
                      << "  --iterations <n>  Measurement iterations (default: 50)\n"
                      << "  --warmup <n>      Warmup iterations (default: 10)\n"
                      << "  --csv <path>      Output CSV file for reporting\n"
                      << "  --help, -h        Print this help message\n";
            return 0;
        }
    }

    print_header();

    std::ofstream csv_file;
    if (!opts.csv_path.empty()) {
        csv_file.open(opts.csv_path);
        csv_file << "Operation,Rows,Cols,Backend,Threads,MedianMs,MeanMs,StdDevMs,MinMs,MaxMs,GFLOPS,BandwidthGBps,Speedup,Efficiency,MaxAbsErr,RelL2Err\n";
    }

    // 1. Raw Dot Product Microbenchmark: len = 2048
    {
        const size_t dot_len = 2048;
        const size_t iters = 20000;
        auto dot_w = generate_random_floats(dot_len, -1.0f, 1.0f, 101);
        auto dot_x = generate_random_floats(dot_len, -1.0f, 1.0f, 102);
        std::vector<inference_engine::BlockQ4_0> dot_q4(dot_len / inference_engine::kQ4_0BlockElements);
        inference_engine::quantize_row_q4_0(dot_w.data(), dot_q4.data(), dot_len);

        volatile float dummy = 0.0f;
        for (size_t w = 0; w < 500; ++w) {
            dummy += inference_engine::scalar::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
            dummy += inference_engine::avx2::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
        }

        std::vector<double> s_times;
        for (size_t i = 0; i < 20; ++i) {
            auto t0 = Clock::now();
            for (size_t it = 0; it < iters; ++it) {
                dummy += inference_engine::scalar::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
            }
            auto t1 = Clock::now();
            s_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::vector<double> a_times;
        for (size_t i = 0; i < 20; ++i) {
            auto t0 = Clock::now();
            for (size_t it = 0; it < iters; ++it) {
                dummy += inference_engine::avx2::dot_q4_0_f32(dot_q4.data(), dot_x.data(), dot_len);
            }
            auto t1 = Clock::now();
            a_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        double total_ops = 2.0 * static_cast<double>(dot_len) * static_cast<double>(iters);
        double total_bytes = static_cast<double>(dot_q4.size() * sizeof(inference_engine::BlockQ4_0)) * static_cast<double>(iters);
        TimingStats s_stats = compute_stats(s_times, total_ops, total_bytes);
        TimingStats a_stats = compute_stats(a_times, total_ops, total_bytes, s_stats.median_ms);

        std::cout << "--- [MICROBENCHMARK: Q4_0 Dot Product (K = " << dot_len << ", " << iters << " iters/sample)] ---\n"
                  << "  Scalar (1T) : " << std::fixed << std::setprecision(2) << s_stats.median_ms << " ms  (" << s_stats.gflops << " GFLOPS)\n"
                  << "  AVX2   (1T) : " << std::fixed << std::setprecision(2) << a_stats.median_ms << " ms  (" << a_stats.gflops << " GFLOPS, "
                  << "Speedup: " << a_stats.speedup << "x)\n\n";
        (void)dummy;
    }

    // 2. Systematic Evaluation Across All 8 TinyLlama Projections
    for (const auto& shape : kTinyLlamaShapes) {
        std::cout << "========================================================================================================================\n"
                  << "Shape: " << shape.name << " (" << shape.rows << " x " << shape.cols << ")\n"
                  << "Weight Data: " << std::fixed << std::setprecision(2)
                  << (shape.rows * shape.cols * sizeof(inference_engine::BlockQ4_0) / 32) / (1024.0 * 1024.0) << " MB (Q4_0) | "
                  << (shape.rows * shape.cols * sizeof(float)) / (1024.0 * 1024.0) << " MB (FP32)\n"
                  << "------------------------------------------------------------------------------------------------------------------------\n";

        const size_t total_elements = shape.rows * shape.cols;
        auto weights_f32 = generate_random_floats(total_elements, -1.0f, 1.0f, 201 + static_cast<uint32_t>(shape.rows));
        auto x_f32 = generate_random_floats(shape.cols, -1.0f, 1.0f, 301 + static_cast<uint32_t>(shape.cols));

        std::vector<inference_engine::BlockQ4_0> q4_weights(total_elements / inference_engine::kQ4_0BlockElements);
        inference_engine::quantize_row_q4_0(weights_f32.data(), q4_weights.data(), total_elements);

        const double ops = 2.0 * static_cast<double>(shape.rows) * static_cast<double>(shape.cols);
        const double bytes = static_cast<double>(q4_weights.size() * sizeof(inference_engine::BlockQ4_0));

        // Scalar Reference Execution
        std::vector<float> ref_out(shape.rows, 0.0f);
        std::vector<double> scalar_times;
        for (size_t w = 0; w < opts.warmup; ++w) {
            inference_engine::scalar::matmul_q4_0_f32(ref_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols);
        }
        for (size_t i = 0; i < opts.iterations; ++i) {
            auto t0 = Clock::now();
            inference_engine::scalar::matmul_q4_0_f32(ref_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols);
            auto t1 = Clock::now();
            scalar_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        TimingStats scalar_stats = compute_stats(scalar_times, ops, bytes);

        std::cout << std::left << std::setw(18) << "Implementation"
                  << std::setw(12) << "Threads"
                  << std::setw(14) << "Median (ms)"
                  << std::setw(12) << "Min (ms)"
                  << std::setw(12) << "Max (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(16) << "Bandwidth (GB/s)"
                  << std::setw(12) << "Speedup"
                  << std::setw(12) << "Efficiency"
                  << "\n"
                  << std::string(115, '-') << "\n";

        std::cout << std::left << std::setw(18) << "Scalar Q4_0"
                  << std::setw(12) << 1
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << scalar_stats.median_ms
                  << std::setw(12) << scalar_stats.min_ms
                  << std::setw(12) << scalar_stats.max_ms
                  << std::setprecision(2)
                  << std::setw(12) << scalar_stats.gflops
                  << std::setw(16) << scalar_stats.bandwidth_gbps
                  << std::setw(12) << "1.00x"
                  << std::setw(12) << "100.0%"
                  << "\n";

        if (csv_file.is_open()) {
            csv_file << shape.name << "," << shape.rows << "," << shape.cols << ",Scalar,1,"
                     << scalar_stats.median_ms << "," << scalar_stats.mean_ms << "," << scalar_stats.stddev_ms << ","
                     << scalar_stats.min_ms << "," << scalar_stats.max_ms << ","
                     << scalar_stats.gflops << "," << scalar_stats.bandwidth_gbps << ",1.0,1.0,0.0,0.0\n";
        }

        // SIMD Benchmarks across thread counts
        for (size_t tc : opts.thread_counts) {
            std::vector<float> simd_out(shape.rows, 0.0f);
            std::vector<double> simd_times;

            if (tc == 1) {
                for (size_t w = 0; w < opts.warmup; ++w) {
                    inference_engine::avx2::matmul_q4_0_f32(simd_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols);
                }
                for (size_t i = 0; i < opts.iterations; ++i) {
                    auto t0 = Clock::now();
                    inference_engine::avx2::matmul_q4_0_f32(simd_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols);
                    auto t1 = Clock::now();
                    simd_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
            } else {
                inference_engine::ThreadPool pool(tc);
                for (size_t w = 0; w < opts.warmup; ++w) {
                    inference_engine::dispatch_matmul_q4_0_f32(simd_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols, inference_engine::SimdBackend::Avx2, &pool);
                }
                for (size_t i = 0; i < opts.iterations; ++i) {
                    auto t0 = Clock::now();
                    inference_engine::dispatch_matmul_q4_0_f32(simd_out.data(), q4_weights.data(), x_f32.data(), shape.rows, shape.cols, inference_engine::SimdBackend::Avx2, &pool);
                    auto t1 = Clock::now();
                    simd_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
            }

            TimingStats simd_stats = compute_stats(simd_times, ops, bytes, scalar_stats.median_ms, tc);
            ErrorStats err = compute_error(ref_out, simd_out);

            std::string label = "AVX2 Q4_0";
            std::cout << std::left << std::setw(18) << label
                      << std::setw(12) << tc
                      << std::fixed << std::setprecision(3)
                      << std::setw(14) << simd_stats.median_ms
                      << std::setw(12) << simd_stats.min_ms
                      << std::setw(12) << simd_stats.max_ms
                      << std::setprecision(2)
                      << std::setw(12) << simd_stats.gflops
                      << std::setw(16) << simd_stats.bandwidth_gbps
                      << std::setw(12) << (std::to_string(simd_stats.speedup).substr(0, 5) + "x")
                      << std::setw(12) << (std::to_string(simd_stats.efficiency * 100.0).substr(0, 5) + "%")
                      << "\n";

            if (csv_file.is_open()) {
                csv_file << shape.name << "," << shape.rows << "," << shape.cols << ",AVX2," << tc << ","
                         << simd_stats.median_ms << "," << simd_stats.mean_ms << "," << simd_stats.stddev_ms << ","
                         << simd_stats.min_ms << "," << simd_stats.max_ms << ","
                         << simd_stats.gflops << "," << simd_stats.bandwidth_gbps << ","
                         << simd_stats.speedup << "," << simd_stats.efficiency << ","
                         << err.max_abs_err << "," << err.rel_l2_err << "\n";
            }
        }
        std::cout << "\n";
    }

    if (csv_file.is_open()) {
        std::cout << "[INFO] Results exported to " << opts.csv_path << "\n";
    }
    std::cout << "========================================================================================================================\n\n";
    return 0;
}
