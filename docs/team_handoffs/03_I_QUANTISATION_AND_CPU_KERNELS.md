# `03_I_QUANTISATION_AND_CPU_KERNELS.md` — Member I Handoff: Quantisation, CPU Kernels, SIMD & Threading

> **Member:** **I** (`Q4_0` Quantisation, ARM64 NEON & x86-64 AVX2/FMA Kernels, Runtime CPU Dispatch, `ThreadPool`, and Kernel Benchmarking Owner)
> **Consolidated Baseline Branch:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Historical Feature Branches (Merged into `33d4bdc`):**
> - `feature/quantisation-simd` (`f76ab556cdaff7a59fb5232a347c32a8d38928c5`, parent `a948734`, merged into `integration` via `967f072`)
> - `origin/feature/quantisation-simd` / `origin/feature/quantisation-simd-m3` (`58c604d035df2a272cbffc682681493dcfdc418c`, merged into `33d4bdc` via `991a527`)
> **Latest Active Remote Feature Branch (Verified in Remote Repository, Pending Merge into Integration Baseline):**
> - `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1` — `"feat(kernels): optimize AVX2 Q4_0 matmul with 2-row register tiling and vectorized scale accumulation"`)

---

## A. Role and Ownership

### 1. Primary Responsibilities
- **`Q4_0` Block Quantisation & IEEE 754 FP16 Conversion (`VERIFIED`):** Own `BlockQ4_0`, software `fp16_to_fp32` / `fp32_to_fp16`, `quantize_row_q4_0`, `dequantize_row_q4_0`, and `scalar::dot_q4_0_f32` / `scalar::matmul_q4_0_f32` (`src/kernels/quant_q4_0.cpp`).
- **Architecture-Specific SIMD Kernels (`VERIFIED`):**
  - x86-64 AVX2 + FMA (`src/kernels/avx2/kernels_avx2.cpp`): `is_available()`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`.
  - ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`): `is_available()`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`.
- **Runtime CPU Dispatch & Multi-Threaded Work Partitioning (`VERIFIED`):** Own `detect_best_backend()`, `resolve_backend()`, `is_backend_available()`, `dispatch_matmul_f32()`, and `dispatch_matmul_q4_0_f32()` (`src/kernels/dispatch.cpp`).
- **Persistent Worker `ThreadPool` (`VERIFIED`):** Own `ThreadPool` (`include/inference_engine/thread_pool.hpp`, `src/kernels/threading/thread_pool.cpp`) with chunked work scheduling and `std::exception_ptr` propagation.
- **Kernel Microbenchmarking & Numerical Parity Suites (`VERIFIED`):** Own `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp`, `benchmarks/bench_kernels.cpp`, `benchmarks/results/*.csv`, and `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`.

### 2. Files Owned and Modified by I
- **Owned Public Headers:**
  - [`include/inference_engine/quant.hpp`](../../include/inference_engine/quant.hpp)
  - [`include/inference_engine/thread_pool.hpp`](../../include/inference_engine/thread_pool.hpp)
  - [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp) (co-owned with **U** for `scalar::*` signatures)
- **Owned C++ Implementation & Benchmarks:**
  - [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp)
  - [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp)
  - [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp)
  - [`src/kernels/threading/thread_pool.cpp`](../../src/kernels/threading/thread_pool.cpp)
  - [`src/kernels/dispatch.cpp`](../../src/kernels/dispatch.cpp)
  - `benchmarks/bench_kernels.cpp`, `benchmarks/results/benchmarks_{baseline,optimised,prefetch}.csv`, `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` (added on `origin/feature/cpu-kernel-optimisation` at `b05c4e4`)
- **Owned Unit Tests:**
  - [`tests/test_quant_packing.cpp`](../../tests/test_quant_packing.cpp)
  - [`tests/test_kernels_simd.cpp`](../../tests/test_kernels_simd.cpp)

### 3. Ownership Boundaries (What I Must NOT Independently Modify)
- Do **not** modify `.ie` binary format parsing or `bind_tinyllama_weights()` (`src/loader/`) without coordinating with **S**.
- Do **not** modify `TransformerEngine::forward_token` (`src/transformer/`) or `KvCache` (`src/core/`) without coordinating with **U**.
- Do **not** modify `src/tokenizer/`, `src/sampler/`, `tests/test_tokenizer.cpp`, or `src/app/main.cpp` without coordinating with **P**.

---

## B. Technical Background & Exact Implementation Details

### 1. `Q4_0` Split-Half Block Format & FP16 Conversion (`src/kernels/quant_q4_0.cpp`) (`VERIFIED`)
- **Memory Layout (`18` bytes per `32` weights):**
  - Bytes `0..1` (`uint16_t d`): IEEE 754 binary16 scale factor $d = \frac{\max_{0 \le i < 32} |w_i|}{8.0}$.
  - Bytes `2..17` (`uint8_t qs[16]`): Split-half nibble packing (GGML-compatible):
    - `qs[j] & 0x0F` stores quantised weight $q_j$ for index $j \in [0, 15]$.
    - `(qs[j] >> 4) & 0x0F` stores quantised weight $q_{j+16}$ for index $j + 16 \in [16, 31]$.
    - Unsigned nibble $q_i = \text{clamp}(\text{round}(w_i / d + 8.5\text{ or }\text{nearest}), 0, 15)$, dequantised as $(q_i - 8) \times \text{fp16\_to\_fp32}(d)$.
- **Bitwise IEEE 754 FP16 $\leftrightarrow$ FP32 (`fp16_to_fp32`, `fp32_to_fp16`):** Pure C++17 bitwise manipulation handling normal numbers, subnormals ($e = 0$), zero, $\pm\infty$, and `NaN` with round-to-nearest-even in `fp32_to_fp16`.

### 2. AVX2 + FMA Optimizations in `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) (`VERIFIED`)
Inspected directly from commit `b05c4e49bf135342e4f94a43a5e4c45763c802e1`:
1. **Runtime CPU + OS XSAVE Verification (`avx2::is_available()`):**
   - Queries CPUID leaf `1` for OSXSAVE (`bit 27`), AVX (`bit 28`), and FMA (`bit 12`), executes `_xgetbv(0)` to verify OS preservation of XMM/YMM state (`(xcr0 & 0x6) == 0x6`), and queries CPUID leaf `7` subleaf `0` (`EBX bit 5`) for AVX2.
2. **Vectorized Scale Accumulation (`compute_block_unscaled_dot_vec` in `src/kernels/avx2/kernels_avx2.cpp`):**
   - Loads `16` packed bytes via `_mm_loadu_si128`, masks low nibbles (`0..15`) and high nibbles (`16..31`) with `0x0F`, subtracts `8` (`_mm_sub_epi8(..., 8)`), widens 8-bit signed integers to 32-bit floats (`_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`), and accumulates the 4 sub-vectors (`0..7`, `8..15`, `16..23`, `24..31`) using `_mm256_fmadd_ps`.
   - Instead of calling `hsum256_ps` inside every 32-weight block, `avx2::dot_q4_0_f32` scales the 8-lane unscaled sum vector by `_mm256_set1_ps(d)` via `_mm256_fmadd_ps(block_dot_vec, _mm256_set1_ps(d), acc)` and performs **a single `hsum256_ps(acc)` horizontal reduction at the end of the row** (reducing horizontal reductions per `2048`-dim row from `64` to `1`).
3. **2-Row Register Tiling (`avx2::matmul_q4_0_f32` and `avx2::matmul_f32`):**
   - Processes output rows in pairs (`m` and `m + 1`, while `m + 1 < out_dim`): loads each 32-element slice of input vector `x` (`x0, x1, x2, x3`) once into YMM registers and reuses those 4 registers across both `row0[b]` and `row1[b]`, halving activation L1 cache load traffic.
   - Odd tail row (`m < out_dim` when `out_dim` is odd) cleanly falls back to `dot_q4_0_f32` / `dot_f32`, verified by `test_simd_matmul_odd_rows_and_tails()` (`M = 1, 3, 7, 15`, `K = 32, 64, 96, 128`).
4. **Contiguous 2-Row-Aligned Chunking in `src/kernels/dispatch.cpp` (`b05c4e4`):**
   - Computes `target_tasks = pool->num_threads() * 4` and `rows_per_task = std::max<size_t>(2, (((out_dim + target_tasks - 1) / target_tasks) + 1) & ~size_t(1))`.
   - Each worker task receives a contiguous multi-row slice `row_count = std::min(rows_per_task, out_dim - start_row)` and calls `avx2::matmul_q4_0_f32(out + start_row, ..., row_count, in_dim)` directly so the 2-row register-tiled loop executes inside every worker thread.

### 3. ARM64 NEON Implementation Status (`src/kernels/neon/kernels_neon.cpp`) (`PARTIALLY IMPLEMENTED` / `TESTED`)
- **What Is Implemented & Tested (`TESTED` on Apple Silicon):**
  - `neon::dot_f32` (4x4 `float32x4_t` unrolled FMA loop) and `neon::dot_q4_0_f32` (`vld1q_u8`, nibble unpack, `vreinterpretq_s8_u8(vsubq_u8(..., 8))`, `vmovl_s8` $\rightarrow$ `vmovl_s16` $\rightarrow$ `vcvtq_f32_s32`, hardware FP16 scale decode `vget_lane_f32(vcvt_f32_f16(...), 0)`, and 4-way `vfmaq_f32`).
  - Passes all `kernels_simd` unit tests on macOS ARM64 (`MaxAbsDiff <= 9.54e-6` vs scalar).
- **What Is NOT Yet Ported to NEON (`PARTIALLY IMPLEMENTED`):**
  - `src/kernels/neon/kernels_neon.cpp` was **not** modified in `b05c4e4`. Specifically:
    1. `neon::dot_q4_0_f32` still calls `vaddvq_f32(sum)` inside every 32-weight block (`64` horizontal reductions per `2048`-element row) instead of accumulating scaled vectors across blocks.
    2. `neon::matmul_q4_0_f32` and `neon::matmul_f32` still iterate one row at a time (`for (size_t row = 0; row < out_dim; ++row)`) rather than using 2-row register tiling.

---

## C. Completed Work, Commits, and Verified Benchmark Results

| Deliverable | Commit(s) | Classification | Evidence |
| :--- | :--- | :--- | :--- |
| `Q4_0` pack/unpack, FP16$\leftrightarrow$FP32, NEON & baseline AVX2 kernels, `ThreadPool`, `dispatch.cpp` | `a948734`, `f76ab55` (`feature/quantisation-simd`) | `VERIFIED` / `TESTED` | `quant_packing` (`4/4`) & `kernels_simd` (`8/8`) CTests |
| Native Ryzen 7 7745HX CPUID/XGETBV verification, `ThreadPool` `std::exception_ptr` propagation, 1000-cycle stress test | `58c604d` (`origin/feature/quantisation-simd-m3`) | `VERIFIED` / `TESTED` | `kernels_simd` CTest on AMD Ryzen 7 7745HX (Windows 11, Clang 23.1.4) |
| AVX2 2-row register tiling, vectorized scale accumulation, 2-row `dispatch.cpp` chunking, `inference_engine_bench` + 3 CSVs | `b05c4e4` (`origin/feature/cpu-kernel-optimisation`) | `VERIFIED` / `TESTED` | Verified against `benchmarks/results/benchmarks_{baseline,optimised,prefetch}.csv` |

### Verified Ryzen 7 7745HX Benchmark Measurements (`VERIFIED` from `benchmarks/results/benchmarks_optimised.csv` & `benchmarks_baseline.csv`)
Hardware: **AMD Ryzen 7 7745HX** (Zen 4, 8 physical cores / 16 SMT threads, AVX2 + FMA, 32 MB L3 cache), 50 iterations + 10 warmup runs per configuration:

| TinyLlama Operator | Shape ($M \times K$) | Scalar 1T (ms / GFLOPS) | AVX2 Baseline 1T (ms / GFLOPS) | AVX2 Opt 1T (ms / GFLOPS / Speedup) | AVX2 Baseline 8T (ms / GFLOPS) | AVX2 Opt 8T (ms / GFLOPS / GB/s / Speedup) | AVX2 Opt 16T (ms / GFLOPS / GB/s / Speedup) | MaxAbsErr vs Scalar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention K** | `256 x 2048` | `0.1458` / `7.19` | `0.0445` / `23.56` | `0.0385` / `27.24` (`3.79x`) | `0.0109` / `96.20` | `0.0112` / `93.62` / `26.33` (`13.02x`) | `0.0336` / `31.21` / `8.78` (`4.34x`) | `1.144e-05` |
| **Attention V** | `256 x 2048` | `0.1459` / `7.19` | `0.0434` / `24.16` | `0.0385` / `27.24` (`3.79x`) | `0.0111` / `94.47` | `0.0103` / `101.80` / `28.63` (`14.17x`) | `0.0211` / `49.70` / `13.98` (`6.91x`) | `1.144e-05` |
| **Attention Q** | `2048 x 2048` | `1.1734` / `7.15` | `0.3696` / `22.70` | `0.3235` / `25.93` (`3.63x`) | `0.0764` / `109.80` | `0.0534` / `157.09` / `44.18` (`21.97x`) | `0.0676` / `124.09` / `34.90` (`17.36x`) | `1.335e-05` |
| **Attention O** | `2048 x 2048` | `1.1769` / `7.13` | `0.3490` / `24.04` | `0.3269` / `25.66` (`3.60x`) | `0.0690` / `121.57` | `0.0532` / `157.68` / `44.35` (`22.12x`) | `0.0715` / `117.33` / `33.00` (`16.46x`) | `1.335e-05` |
| **SwiGLU Gate** | `5632 x 2048` | `3.2764` / `7.04` | `0.9946` / `23.19` | `0.9248` / `24.94` (`3.54x`) | `0.1924` / `119.90` | `0.1603` / `143.91` / `40.47` (`20.44x`) | `0.1482` / `155.66` / `43.78` (`22.11x`) | `2.289e-05` |
| **SwiGLU Up** | `5632 x 2048` | `3.2530` / `7.09` | `0.9552` / `24.15` | `0.9130` / `25.27` (`3.56x`) | `0.1726` / `133.65` | `0.1755` / `131.45` / `36.97` (`18.54x`) | `0.1432` / `161.09` / `45.31` (`22.72x`) | `2.289e-05` |
| **SwiGLU Down** | `2048 x 5632` | `3.3402` / `6.91` | `0.9355` / `24.66` | `0.9101` / `25.35` (`3.67x`) | `0.1393` / `165.60` | `0.1499` / `153.89` / `43.28` (`22.28x`) | `0.1417` / `162.80` / `45.79` (`23.57x`) | `4.196e-05` |
| **LM Head** | `32000 x 2048` | `18.4875` / `7.09` | `5.4436` / `24.08` | `6.1583` / `21.28` (`3.00x`) | `0.9141` / `143.39` | `1.0356` / `126.57` / `35.60` (`17.85x`) | `0.8082` / `162.18` / `45.61` (`22.87x`) | `2.670e-05` |

### Audit Corrections to `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` (`VERIFIED` Against Actual CSVs & `bench_kernels.cpp`)
1. **Prefetching CSV Figures (`benchmarks/results/benchmarks_prefetch.csv`):**
   - In `CPU_KERNEL_BENCHMARKS.md` Section 2.5, the narrative text slightly misquoted two numbers from `benchmarks_prefetch.csv`. The actual measured values in `benchmarks_prefetch.csv` are:
     - **Attention Q (`2048x2048`, 8T):** `0.0693 ms` (`121.05 GFLOPS`, `34.04 GB/s`) with `_mm_prefetch` vs `0.0534 ms` (`157.09 GFLOPS`, `44.18 GB/s`) without software prefetch (**+29.8% higher latency / -22.9% throughput regression**).
     - **SwiGLU Down (`2048x5632`, 8T):** `0.1676 ms` (`137.64 GFLOPS`, `38.71 GB/s`) with `_mm_prefetch` vs `0.1499 ms` (`153.89 GFLOPS`, `43.28 GB/s`) without software prefetch (**+11.8% higher latency / -10.6% throughput regression**).
     - **LM Head (`32000x2048`, 16T):** `0.8753 ms` (`149.75 GFLOPS`, `42.12 GB/s`) with `_mm_prefetch` vs `0.8082 ms` (`162.18 GFLOPS`, `45.61 GB/s`) without software prefetch (**+8.3% higher latency**).
   - The engineering conclusion in `ADR-012` is **100% confirmed by the CSVs**: omitting manual `_mm_prefetch` and relying on Zen 4 hardware L1/L2 stream prefetchers is strictly superior.
2. **`benchmarks/bench_kernels.cpp` Implementation Details vs Doc Narrative:**
   - `bench_kernels.cpp` uses `std::chrono::steady_clock` (not `high_resolution_clock`), defaults to `--iterations 50 --warmup 10`, computes bandwidth from compressed weight bytes `M * (K / 32) * 18` (excluding activation bytes), does not implement `--filter`, and hardcodes `avx2::matmul_q4_0_f32` / `SimdBackend::Avx2` (so on ARM64 macOS it falls back to scalar).

---

## D. Current State: What Works vs. What Remains Pending

- **What Works Now (`VERIFIED` / `TESTED`):**
  - `quant_q4_0.cpp`, `kernels_avx2.cpp` (with 2-row register tiling and vectorized scale accumulation in `b05c4e4`), `dispatch.cpp` (with contiguous 2-row-aligned chunk scheduling), and `thread_pool.cpp` (with `std::exception_ptr` safety) are implemented, verified on native Ryzen 7 7745HX hardware, and backed by CSV artifacts.
- **What Remains Incomplete (`PARTIALLY IMPLEMENTED`):**
  1. `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) is pushed to remote (`origin`) but not yet merged into the consolidated integration baseline (`integration-ps08-team-sync-2026-10-31`).
  2. `src/kernels/neon/kernels_neon.cpp` does **not** yet have the 2-row register tiling or deferred end-of-row horizontal reduction (`vaddvq_f32`) that `src/kernels/avx2/kernels_avx2.cpp` received in `b05c4e4`.
  3. `benchmarks/bench_kernels.cpp` hardcodes `avx2::matmul_q4_0_f32` and `SimdBackend::Avx2`, preventing native NEON microbenchmarking on Apple Silicon laptops.
  4. For small output dimensions (`M = 256` in Attention K/V), 16-thread SMT execution (`0.0336 ms`) is slower than 8-thread physical-core execution (`0.0112 ms`) because wake-up and synchronization overhead across 16 threads dominates a `144 KB` weight matrix.

---

## E. Dependencies on Other Team Members

- **On Member S:** Merging `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) into the consolidated integration branch and providing converted 22-layer `.ie` files (`models/tinyllama-q4_0.ie`) for end-to-end tok/s benchmarking.
- **Downstream Consumers (U and P):** `TransformerEngine::project_linear` (`src/transformer/transformer.cpp`) calls `dispatch_matmul_f32` and `dispatch_matmul_q4_0_f32` with `&thread_pool_`, so any kernel or dispatch improvements in `src/kernels/` immediately accelerate full-model prefill and decode.

---

## F. Remaining Tasks (Priority Order)

### `P0` — Blocks the Entire Project
1. **Coordinate Clean Merge of `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) into the Integration Baseline:**
   - Ensure `src/kernels/avx2/kernels_avx2.cpp`, `src/kernels/dispatch.cpp`, `tests/test_kernels_simd.cpp`, `CMakeLists.txt` (`inference_engine_bench`), `benchmarks/`, `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`, and `ADR-012` are merged cleanly without overwriting P's `tests/test_tokenizer.cpp`.

### `P1` — Required for the Next Major Milestone
2. **Port Vectorized Scale Accumulation and 2-Row Register Tiling to ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`):**
   - In `neon::dot_q4_0_f32`: replace per-block `vaddvq_f32(sum)` with a vector accumulator `acc = vfmaq_n_f32(acc, block_sum, d)` across all blocks in the row, calling `vaddvq_f32(acc)` once at the end of the row.
   - In `neon::matmul_q4_0_f32` and `neon::matmul_f32`: implement 2-row register tiling (`m` and `m + 1`) matching `avx2::matmul_q4_0_f32` so Apple Silicon M-series laptops achieve the same activation reuse and contiguous 2-row `dispatch.cpp` benefit.
3. **Make `benchmarks/bench_kernels.cpp` Dispatch Backend-Agnostic (`detect_best_backend()`):**
   - Replace hardcoded `avx2::matmul_q4_0_f32` and `SimdBackend::Avx2` in `benchmarks/bench_kernels.cpp` with `detect_best_backend()` (calling `neon::matmul_q4_0_f32` on ARM64 and `avx2::matmul_q4_0_f32` on x86-64) and record Apple Silicon NEON CSV results alongside Ryzen 7 7745HX AVX2 results.

### `P2` — Important Improvements
4. **Adaptive Thread-Count Cap for Small Output Dimensions (`out_dim <= 256`) in `src/kernels/dispatch.cpp`:**
   - As proven by `benchmarks_optimised.csv`, `256x2048` K/V projections run in `0.0103–0.0112 ms` at 8T (`93.6–101.8 GFLOPS`) but regress to `0.0211–0.0336 ms` at 16T (`31.2–49.7 GFLOPS`). Capping `target_tasks` or falling back to fewer chunks when `out_dim * in_dim` is small eliminates SMT wake-up overhead on K/V projections.
5. **Reconcile Minor Text Figures in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`:**
   - Update Section 2.5 prefetch comparison numbers (`0.0693 ms` / `121.05 GFLOPS` on `2048x2048` 8T and `0.1676 ms` / `137.64 GFLOPS` on `2048x5632` 8T) to match `benchmarks/results/benchmarks_prefetch.csv` exactly, and replace machine-local `file:///c:/Users/Lenovo/...` links with repository-relative links.

### `P3` — Optional Optimisation or Polish
6. **4-Row Register Tiling or Activation Quantisation (`Q8_0` dot product) Experiment:**
   - Optional exploration if further `LM Head` (`32000x2048`) throughput gains are desired after 22-layer real-model profiling.

---

## G. New-Session Continuation Prompt (Ready to Paste into Antigravity)

```markdown
You are operating as **Member I (`Q4_0` Quantisation, CPU Kernels, SIMD & Threading Owner)** on the **BIOS V2 — PS08 (`inference-engine`)** project in repository `shreyaas5/inference-engine`.

Follow this exact 10-step autonomous workflow without asking the team lead for background already documented in the repository:

1. **Read the Master Project Context:** Read `AGENTS.md` and `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`.
2. **Read Your Complete Handoff & Shared Interfaces:** Read `docs/team_handoffs/03_I_QUANTISATION_AND_CPU_KERNELS.md`, `docs/team_handoffs/05_SHARED_INTERFACES_AND_DEPENDENCIES.md`, and `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`.
3. **Inspect the Current Git State:** Run `git fetch --all --prune`, `git status`, `git branch -a -vv`, and `git log --all --oneline --graph -n 20`.
4. **Identify the Latest Relevant Commits:** Verify the consolidated baseline (`integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff`) and your latest feature commit on `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1`).
5. **Inspect the Actual Source Files:** Read `include/inference_engine/{quant,thread_pool,kernels}.hpp`, `src/kernels/quant_q4_0.cpp`, `src/kernels/avx2/kernels_avx2.cpp`, `src/kernels/neon/kernels_neon.cpp`, `src/kernels/dispatch.cpp`, `src/kernels/threading/thread_pool.cpp`, `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp`, and (on `origin/feature/cpu-kernel-optimisation`) `benchmarks/bench_kernels.cpp` and `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`.
6. **Compare Implementation Against Documented State:** Verify that `avx2::dot_q4_0_f32` and `avx2::matmul_q4_0_f32` in `b05c4e4` implement vectorized scale accumulation and 2-row register tiling, while `src/kernels/neon/kernels_neon.cpp` still uses per-block `vaddvq_f32` and single-row loops, and `benchmarks/bench_kernels.cpp` currently hardcodes `SimdBackend::Avx2`.
7. **Avoid Reimplementing Completed Functionality:** `Q4_0` block packing, FP16 conversion, `ThreadPool` with `std::exception_ptr`, and AVX2 2-row register tiling (`b05c4e4`) are already implemented and tested. Do not rewrite them.
8. **Identify the Highest-Priority Unfinished Tasks (`P0` / `P1` / `P2`):**
   - Ensure your working branch incorporates both the `33d4bdc` integration baseline and `b05c4e4` (`origin/feature/cpu-kernel-optimisation`) without reverting `tests/test_tokenizer.cpp`.
   - **Task `P1-1`:** Port vectorized per-block scale accumulation (single end-of-row `vaddvq_f32`) and 2-row register tiling (`m` and `m + 1`) to `src/kernels/neon/kernels_neon.cpp` (`neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32`, `neon::matmul_f32`), matching your AVX2 design and `test_simd_matmul_odd_rows_and_tails()`.
   - **Task `P1-2`:** Update `benchmarks/bench_kernels.cpp` to use `detect_best_backend()` rather than hardcoding `SimdBackend::Avx2` so it benchmarks both AVX2 on x86-64 and NEON on ARM64.
   - **Task `P2`:** Fix the minor prefetch CSV figures and machine-local links in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`, and optionally add an adaptive chunk/thread cap in `src/kernels/dispatch.cpp` for small matrices (`out_dim <= 256`) where 16T SMT overhead exceeds 8T latency.
9. **Produce an Initial Repository Audit and Action Plan:** Output a concise audit of your branch state, CPU architecture, and test results, followed by your implementation plan.
10. **Begin Implementation Immediately:** Execute the tasks autonomously within your owned files (`src/kernels/`, `include/inference_engine/{quant,thread_pool,kernels}.hpp`, `tests/test_{quant_packing,kernels_simd}.cpp`, `benchmarks/`, `docs/benchmarks/`). Ask the team lead only if a change requires modifying a cross-module signature in `kernels.hpp`. Update context files per `AGENTS.md` Section 8 before finishing.
```
