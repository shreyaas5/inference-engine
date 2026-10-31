# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-11-01T00:15:00+05:30  
**Current Milestone:** `M3/M4 — CPU Kernel Optimisation, SIMD Profiling & Benchmarking`  
**Maintainer:** I (Quantisation & CPU Optimisation Specialist) / S (Project Lead)  

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `feature/cpu-kernel-optimisation` (branched from `origin/integration-ps08-team-sync-2026-10-31`)
- **Remote Repository:** `https://github.com/shreyaas5/inference-engine`
- **Compiler / Toolchain:** Clang 23.1.4 (LLVM-MinGW UCRT x86_64), CMake 3.31.5, Ninja
- **Hardware Verified:** AMD Ryzen 7 7745HX (Zen 4, 8 physical cores, 16 SMT threads, 16 GB DDR5-5200 RAM, Windows 11)
- **CMake Build Status:** **PASSING (Zero Warnings)** with `-Wall -Wextra -Wpedantic` on:
  - `inference_engine_core` (static library)
  - `inference-engine` (CLI executable)
  - `inference_engine_tests` (CTest runner, 8 suites)
  - `inference_engine_bench` (dedicated benchmarking executable)
- **CTest Suite Status:** **8 / 8 PASSING (100%)** in 2.14s on AMD Ryzen 7 7745HX:
  1. `loader_validation` — **PASSED** (15 malformed-file rejection checks + 21-tensor `FP32`, `Q4_0`, and mixed-precision `mmap` reload, `bind_transformer_weights()`, and `TensorView::data_q4_0()`).
  2. `quant_packing` — **PASSED** (18-byte `BlockQ4_0`, nibble packing, zero-block, out-of-bounds, subnormal/inf FP16, and bit-exact upstream GGML reference parity).
  3. `kernels_scalar` — **PASSED** (Unit tests for `rmsnorm_f32`, `rope_inplace_f32`, `softmax_f32`, `swiglu_f32`, and multi-head `gqa_attention_f32`).
  4. `kernels_simd` — **PASSED** (Native AVX2/FMA CPUID detection, multi-length & distribution Q4_0 dot tests, TinyLlama projection numerical error tests, 2-row tiled odd row / boundary tail tests, 1,000-cycle `ThreadPool` stress test, exception propagation, and mixed-precision SwiGLU MLP flow).
  5. `kv_cache` — **PASSED** (Strict separation of `append()` and `commit_token()`).
  6. `tokenizer` — **PASSED** (SentencePiece `tokenizer.model` 32,000-piece vocabulary load, BOS/EOS/UNK verification, special token splitting, and round-trip encode/decode).
  7. `sampler` — **PASSED** (Deterministic greedy argmax, temperature scaling, top-p nucleus truncation, and NaN/Inf logit filtering).
  8. `golden_forward` — **PASSED** (Synthetic 2-layer `FP32`, `Q4_0`, and mixed-precision forward parity across prefill and decode steps).

---

## 2. Kernel Optimization & Benchmarking Milestones (New in This Session)

1. **Dedicated Benchmark Executable (`benchmarks/bench_kernels.cpp`):**
   - Configurable warmup loops (`--warmup`), measurement iterations (`--iterations`), and CSV output (`--csv`).
   - Comprehensive statistical aggregation: median, mean, min, max, standard deviation, GFLOPS, effective memory bandwidth (GB/s), speedup vs scalar, thread efficiency, and numerical error metrics.
2. **AVX2 Q4_0 Kernel Optimizations (`src/kernels/avx2/kernels_avx2.cpp`, `src/kernels/dispatch.cpp`):**
   - **Vectorized Scale Accumulation:** Replaced 64 per-block horizontal reductions in `dot_q4_0_f32` with YMM vector accumulators scaled by $d$ via `_mm256_fmadd_ps`, performing a single horizontal reduction at the end of the row (14.2% single-thread speedup).
   - **2-Row Register Tiling:** Tiled `matmul_q4_0_f32` across 2 rows simultaneously. Loaded 32 floats of activation vector $x$ once into registers `vx0..vx3` and reused across both rows, cutting activation load traffic by 50%.
   - **Contiguous Chunk Slicing in ThreadPool Dispatch:** Sliced rows into contiguous 2-row-aligned chunks (`rows_per_task = max(2, (out_dim / (num_threads * 4)) & ~1)`), slashing task queue contention from 32,000 tasks down to ~32 tasks on 8 threads.
   - **Software Prefetching Analysis:** Empirically evaluated `_mm_prefetch(weight_row + b + 4)` and confirmed a 10–23% throughput regression due to interference with Zen 4 hardware stream prefetchers. Safely discarded based on data.
3. **Measured Hardware Throughput (AMD Ryzen 7 7745HX):**
   - **Attention Q ($2048 \times 2048$, 8T):** 0.076 ms $\to$ **0.053 ms** (**+43.1% throughput gain**, 157.09 GFLOPS, 44.18 GB/s, 21.97x speedup vs scalar).
   - **SwiGLU Up ($5632 \times 2048$, 16T):** 0.162 ms $\to$ **0.143 ms** (161.09 GFLOPS, 45.31 GB/s, 22.71x speedup vs scalar).
   - **SwiGLU Down ($2048 \times 5632$, 16T):** 0.148 ms $\to$ **0.142 ms** (162.80 GFLOPS, 45.79 GB/s, 23.57x speedup vs scalar).
   - **LM Head ($32000 \times 2048$, 16T):** 0.874 ms $\to$ **0.808 ms** (162.18 GFLOPS, 45.61 GB/s, 22.87x speedup vs scalar).
4. **Numerical Precision Stability:**
   - Max absolute error across all TinyLlama shapes $< 4.2 \times 10^{-5}$ vs scalar oracle.
   - Relative $L_2$ error $< 3.5 \times 10^{-7}$ (well below single-precision machine epsilon).
5. **Documentation & Governance:**
   - Detailed benchmark report published: [`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`](docs/benchmarks/CPU_KERNEL_BENCHMARKS.md).
   - Raw benchmark datasets archived: [`benchmarks/results/`](benchmarks/results/).
   - Architectural decision recorded: `ADR-012` in [`DECISIONS.md`](DECISIONS.md).

---

## 3. Top Three Immediate Next Priorities

1. **[All / S & I] Merge `feature/cpu-kernel-optimisation` into `integration-ps08-team-sync-2026-10-31` / `integration`:**
   - Bring optimized AVX2 kernels, 2-row tiling, and benchmark runner into the shared integration branch for all team members.
2. **[S & U — Issues #1 & #3] Download Full 2.20 GB TinyLlama-1.1B Checkpoint and Execute Real 22-Layer Golden Run:**
   - Convert `model.safetensors` to `tinyllama-1.1b-q4_0.ie` (`619.11 MB`), and run end-to-end forward inference against real weights.
3. **[Member S — Milestone M4] Port 2-Row Register Tiling to ARM64 NEON Kernels (`src/kernels/neon/kernels_neon.cpp`):**
   - Apply matching 2-row tiling and contiguous chunk dispatch to Apple Silicon M4 to evaluate ARM64 NEON speedups.
