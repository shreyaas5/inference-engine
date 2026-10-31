# `03_I_QUANTISATION_AND_CPU_KERNELS.md` — Member I Module Handoff

**Owner:** Member I (`Q4_0` Quantisation, ARM64 NEON Kernels, x86-64 AVX2/FMA Kernels, CPU Feature Detection, `ThreadPool`, & Kernel Dispatch)
**Consolidated Branch:** `integration-ps08-team-sync-2026-10-31`
**Merged Feature Branches:**
- `origin/feature/quantisation-simd` (`f76ab55`) — M2 `BlockQ4_0` upstream GGML parity (`KI-007`), ARM64 NEON & x86-64 AVX2/FMA `FP32` and `Q4_0` dot/matmul kernels (`KI-005`), CPUID detection, and `ThreadPool` task chunking (`ADR-009`)
- `origin/feature/quantisation-simd-m3` (`58c604d`) — M3 native x86-64 AVX2/FMA verification on AMD Ryzen 7 7745HX (`KI-013`), `ThreadPool` `std::exception_ptr` capture/rethrow (`ADR-010`), and TinyLlama shape accuracy/performance benchmarks

---

## 1. Owned Files & Responsibilities

| File Path | Role | Status |
|---|---|---|
| [`include/inference_engine/quant.hpp`](../../include/inference_engine/quant.hpp), [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp) | `BlockQ4_0` (`18` bytes / `32` weights), IEEE 754 `fp16_to_fp32` / `fp32_to_fp16`, `quantize_row_q4_0`, `dequantize_row_q4_0` | `REFERENCE VERIFIED (GGML)` |
| [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp) | ARM64 NEON vectorised `neon::dot_f32`, `neon::dot_q4_0_f32`, `neon::matmul_f32`, `neon::matmul_q4_0_f32` | `TESTED (Apple M4 ARM64)` |
| [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp) | x86-64 AVX2/FMA vectorised `avx2::dot_f32`, `avx2::dot_q4_0_f32`, `avx2::matmul_f32`, `avx2::matmul_q4_0_f32` | `TESTED (Ryzen 7 7745HX x86-64)` |
| [`include/inference_engine/thread_pool.hpp`](../../include/inference_engine/thread_pool.hpp), [`src/kernels/threading/thread_pool.cpp`](../../src/kernels/threading/thread_pool.cpp) | Persistent worker thread pool with dynamic task chunking (`target_chunks = num_workers * 4`) and `std::exception_ptr` propagation | `TESTED (Stress 32,000 tasks)` |
| [`src/kernels/dispatch.cpp`](../../src/kernels/dispatch.cpp) | Runtime backend selection (`Scalar`, `Neon`, `Avx2`) and row-parallel multi-threaded `dispatch_matmul_f32` / `dispatch_matmul_q4_0_f32` | `TESTED (8/8 CTest)` |
| [`tests/test_quant_packing.cpp`](../../tests/test_quant_packing.cpp), [`tests/test_kernels_simd.cpp`](../../tests/test_kernels_simd.cpp) | `Q4_0` packing/GGML parity tests, SIMD accuracy tests across all 5 TinyLlama matrix shapes, `ThreadPool` stress & exception tests | `TESTED (8/8 CTest)` |

---

## 2. Frozen Technical Contracts Owned by I

### 2.1 `BlockQ4_0` Memory & Nibble Layout (`18` Bytes per `32` Weights)
- `struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; }` (`static_assert(sizeof(BlockQ4_0) == 18)`).
- **Split-Half Nibble Ordering:**
  - Low nibble `qs[j] & 0x0F` stores unsigned quantised value $q_j \in [0, 15]$ for element index $j \in [0, 15]$.
  - High nibble `(qs[j] >> 4) & 0x0F` stores unsigned quantised value $q_{j+16} \in [0, 15]$ for element index $j + 16 \in [16, 31]$.
- **Dequantisation Formula:** $w_k = (q_k - 8) \times \text{fp16\_to\_fp32}(d)$.
- **Quantisation Scale Selection:** Finds element $w_{\text{amax}}$ with largest absolute value in the 32-element block, computes $d = w_{\text{amax}} / -8.0$, stores `d_fp16 = fp32_to_fp16(d)`, and quantises using `id = 1.0f / fp16_to_fp32(d_fp16)` and `static_cast<int>(x * id + 8.5f)` clamped to `[0, 15]` (bit-exact with upstream GGML `quantize_row_q4_0_ref`).

### 2.2 SIMD Kernel Architecture & Verification Evidence
- **ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`):**
  - Unpacks 16 packed bytes into low and high 4-bit vectors (`vandq_u8` / `vshrq_n_u8`), subtracts `8` in `int8x16_t`, widens to `int16` -> `int32` -> `float32x4_t`, and accumulates four 4-wide FMA lanes (`vfmaq_f32`) before scaling by `fp16_to_fp32(block.d)`.
- **x86-64 AVX2 + FMA (`src/kernels/avx2/kernels_avx2.cpp`):**
  - Loads 16 packed nibble bytes (`_mm_loadu_si128`), splits low/high halves into 8-lane `__m256i` (`_mm256_cvtepu8_epi32`), subtracts `8` (`_mm256_sub_epi32`), converts to `__m256` (`_mm256_cvtepi32_ps`), and accumulates into four `__m256` accumulators via `_mm256_fmadd_ps`.
  - **Verified Natively on AMD Ryzen 7 7745HX (`58c604d`):**
    - All 5 TinyLlama projection shapes (`256x2048`, `2048x2048`, `5632x2048`, `2048x5632`, `32000x2048`) match scalar reference with `MaxAbsErr < 4.6e-5` and `RelErr < 3.1e-7`.
    - Measured **16.79x speedup** on `2048x2048` attention projection (`110.38 GFLOPS`, `28.91 GB/s`) and **20.07x speedup** on `2048x5632` SwiGLU down projection (`125.51 GFLOPS`, `32.88 GB/s`) with 8 threads.

---

## 3. Ready-to-Paste Antigravity Continuation Prompt for Member I

```markdown
You are acting as Member I (Q4_0 Quantisation, ARM64 NEON & x86-64 AVX2/FMA Kernels, ThreadPool & Dispatch) for `inference-engine` (BIOS V2 — PS08).

Before making any changes, read:
1. `AGENTS.md`
2. `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`
3. `docs/team_handoffs/03_I_QUANTISATION_AND_CPU_KERNELS.md`
4. `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`

Current branch: `integration-ps08-team-sync-2026-10-31` (consolidates S `4515579`, U `ce1026a`, I `58c604d`, and P `ce769c8`; all 8/8 CTest suites pass).

Your next engineering tasks on your AMD Ryzen 7 7745HX x86-64 machine:
1. Fetch and check out `integration-ps08-team-sync-2026-10-31`:
   `git fetch origin && git checkout integration-ps08-team-sync-2026-10-31`
2. Configure and build all targets in Release mode and run all 8 CTest suites natively on x86-64 AVX2:
   `cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON && cmake --build build --config Release --parallel 8 && ctest --test-dir build --output-on-failure`
3. Once the converted `tinyllama-1.1b-fp32.ie` and `tinyllama-1.1b-q4_0.ie` models are available, run the `--benchmark` sweep across `--threads 1`, `2`, `4`, `8`, and `16` on Ryzen 7 7745HX and record the x86-64 AVX2 throughput scaling table in `docs/BENCHMARKING.md`.
```
