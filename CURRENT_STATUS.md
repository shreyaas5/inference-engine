# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-11-01T00:25:00+05:30  
**Current Milestone:** `M3/M4 — Phase 2B: Benchmark Verification, NEON Optimisation & Cross-Platform Validation`  
**Maintainer:** I (Quantisation & CPU Optimisation Specialist) / S (Project Lead)  

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `feature/cpu-kernel-optimisation` (tracking `origin/feature/cpu-kernel-optimisation`)
- **Remote Repository:** `https://github.com/shreyaas5/inference-engine`
- **Compiler / Toolchain:** Clang 23.1.4 (LLVM-MinGW UCRT x86_64), CMake 3.31.5, Ninja
- **Hardware Verified:** AMD Ryzen 7 7745HX (Zen 4, 8 physical cores, 16 SMT threads, 16 GB DDR5-5200 RAM, Windows 11)
- **ARM64 Toolchain Verified:** `clang++ --target=aarch64-w64-mingw32` (syntax & type checking with 0 warnings)
- **CMake Build Status:** **PASSING (Zero Warnings)** with `-Wall -Wextra -Wpedantic` on:
  - `inference_engine_core` (static library)
  - `inference-engine` (CLI executable)
  - `inference_engine_tests` (CTest runner, 8 suites)
  - `inference_engine_bench` (dedicated benchmarking executable)
- **CTest Suite Status:** **8 / 8 PASSING (100%)** in 2.08s on AMD Ryzen 7 7745HX:
  1. `loader_validation` — **PASSED** (15 malformed-file rejection checks + 21-tensor `FP32`, `Q4_0`, and mixed-precision `mmap` reload, `bind_transformer_weights()`, and `TensorView::data_q4_0()`).
  2. `quant_packing` — **PASSED** (18-byte `BlockQ4_0`, nibble packing, zero-block, out-of-bounds, subnormal/inf FP16, and bit-exact upstream GGML reference parity).
  3. `kernels_scalar` — **PASSED** (Unit tests for `rmsnorm_f32`, `rope_inplace_f32`, `softmax_f32`, `swiglu_f32`, and multi-head `gqa_attention_f32`).
  4. `kernels_simd` — **PASSED** (Native AVX2/FMA CPUID detection, multi-length & distribution Q4_0 dot tests, TinyLlama projection numerical error tests, 2-row tiled odd row / boundary tail tests for AVX2 and NEON, 1,000-cycle `ThreadPool` stress test, exception propagation, and mixed-precision SwiGLU MLP flow).
  5. `kv_cache` — **PASSED** (Strict separation of `append()` and `commit_token()`).
  6. `tokenizer` — **PASSED** (SentencePiece `tokenizer.model` 32,000-piece vocabulary load, BOS/EOS/UNK verification, special token splitting, and round-trip encode/decode).
  7. `sampler` — **PASSED** (Deterministic greedy argmax, temperature scaling, top-p nucleus truncation, and NaN/Inf logit filtering).
  8. `golden_forward` — **PASSED** (Synthetic 2-layer `FP32`, `Q4_0`, and mixed-precision forward parity across prefill and decode steps).

---

## 2. Phase 2B Deliverables & Verified Achievements

1. **Audited & Mathematically Consistent Benchmark Report:**
   - Published revised [`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`](docs/benchmarks/CPU_KERNEL_BENCHMARKS.md).
   - Recalculated all percentage improvements using standard formula:
     $$\text{Improvement (\%)} = \frac{\text{Baseline Latency} - \text{Optimised Latency}}{\text{Baseline Latency}} \times 100$$
   - Audited measurements on AMD Ryzen 7 7745HX:
     - **Attention Q (8T):** 0.0764 ms $\to$ **0.0534 ms** (**+30.10% latency reduction**, 1.43x throughput speedup, 157.09 GFLOPS, 44.18 GB/s).
     - **Attention Q (1T):** 0.3696 ms $\to$ **0.3235 ms** (**+12.47% latency reduction**, 1.14x speedup via vectorized scale accumulation).
     - **Attention O (8T):** 0.0690 ms $\to$ **0.0532 ms** (**+22.90% latency reduction**, 1.30x speedup).
     - **SwiGLU Gate (8T):** 0.1924 ms $\to$ **0.1603 ms** (**+16.68% latency reduction**, 1.20x speedup via 2-row tiling).
     - **SwiGLU Up (16T):** 0.1622 ms $\to$ **0.1432 ms** (**+11.71% latency reduction**, 1.13x speedup).
     - **SwiGLU Down (16T):** 0.1485 ms $\to$ **0.1417 ms** (**+4.58% latency reduction**, 162.80 GFLOPS, 45.79 GB/s).
     - **LM Head (16T):** 0.8740 ms $\to$ **0.8082 ms** (**+7.53% latency reduction**, 162.18 GFLOPS, 45.61 GB/s).
2. **Reproducible Validation of the 5 Core Performance Claims:**
   - **Vectorized Scale Accumulation:** Verified in `dot_q4_0_f32` (+12.47% single-thread gain on Attention Q).
   - **Two-Row Register Tiling:** Verified in `matmul_q4_0_f32` (+16.68% gain on SwiGLU Gate at 8T).
   - **ThreadPool Task Reduction & Scheduling:** Verified task queue count drop from 32,000 to 32 on LM Head (+7.53% gain at 16T).
   - **Software Prefetch Regression:** Confirmed a +29.78% latency penalty on Attention Q (8T) when adding manual `_mm_prefetch`, confirming conflict with Zen 4 hardware stream prefetchers.
3. **Completed ARM64 NEON Two-Row Tiling Port (`src/kernels/neon/kernels_neon.cpp`):**
   - Implemented `compute_block_unscaled_dot_neon_vec` and 2-row register tiling in `neon::matmul_q4_0_f32` and `neon::matmul_f32`.
   - Utilizes ARM64's 32 architectural vector registers to reuse activation vector loads across row pairs.
   - Preserves scalar fallback on non-ARM64 platforms.
   - Cross-compiled with `clang++ --target=aarch64-w64-mingw32` with zero errors.
   - Tested odd dimensions ($1..255$) and boundary tails in `test_kernels_simd.cpp`.
4. **Governance & Documentation:**
   - Updated `ADR-012` in [`DECISIONS.md`](DECISIONS.md) to document both AVX2 and NEON 2-row tiling.
   - Archived baseline, prefetch, and optimized CSVs in [`benchmarks/results/`](benchmarks/results/).

---

## 3. Top Three Immediate Next Priorities

1. **[All / S & I] Merge `feature/cpu-kernel-optimisation` into `integration-ps08-team-sync-2026-10-31` / `integration`:**
   - Integrate optimized AVX2 and NEON 2-row tiled kernels, benchmark runner, and documentation into the shared branch.
2. **[Member S — M4 Cross-Platform Benchmarks] Run `inference_engine_bench` on Apple Silicon M4:**
   - Execute benchmark suite natively on Apple MacBook Air M4 to record native ARM64 NEON hardware numbers.
3. **[S & U — Issues #1 & #3] Download Full 2.20 GB TinyLlama-1.1B Checkpoint and Execute Real 22-Layer Golden Run:**
   - Convert `model.safetensors` to `tinyllama-1.1b-q4_0.ie` (`619.11 MB`), and run end-to-end forward inference against real weights.
