# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Limitations & Resolved Ledger

> **Last Updated:** 2026-11-01
> **Consolidated Baseline:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)

---

## 1. Active Issues & Limitations

### `KI-015` — Two Remote Feature Branches (`b05c4e4` and `e64078c`) Pending Merge into Consolidated Integration Baseline
- **Severity:** Medium (`P0` integration task)
- **Owner:** **S** (with **I** and **U**)
- **Description:**
  1. Member I pushed `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1`, parent `5929a17`) containing AVX2 2-row register tiling, vectorized scale accumulation, contiguous 2-row `dispatch.cpp` chunk scheduling, and `inference_engine_bench`. Because it branched from `5929a17` and also touched `tests/test_tokenizer.cpp` and root status files, it must be merged selectively so `tests/test_tokenizer.cpp` and reconciled docs from `33d4bdc` are preserved.
  2. Member U pushed `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d`, parent `5929a17`) adding `target_compile_options(sentencepiece-static PRIVATE -include cstdint)` in `CMakeLists.txt` for GCC 16 compatibility.

### `KI-016` — ARM64 NEON Kernels Lack 2-Row Register Tiling and Deferred End-of-Row Horizontal Reduction (`PARTIALLY IMPLEMENTED`)
- **Severity:** Medium (`P1` performance parity on Apple Silicon)
- **Owner:** **I**
- **Description:** In `origin/feature/cpu-kernel-optimisation` (`b05c4e4`), `src/kernels/avx2/kernels_avx2.cpp` was upgraded with vectorized scale accumulation (single `hsum256_ps` per row) and 2-row register tiling (`m` and `m + 1`), and `src/kernels/dispatch.cpp` was updated to pass contiguous 2-row-aligned slices. However, `src/kernels/neon/kernels_neon.cpp` still performs `vaddvq_f32(sum)` inside every 32-weight block (`64` times per `2048`-dim row) and processes one row at a time. In addition, `benchmarks/bench_kernels.cpp` hardcodes `avx2::matmul_q4_0_f32` and `SimdBackend::Avx2`.
- **Resolution Plan:** Port deferred `vaddvq_f32` accumulation and 2-row register tiling to `src/kernels/neon/kernels_neon.cpp` and update `benchmarks/bench_kernels.cpp` to dispatch via `detect_best_backend()`.

### `KI-017` — CLI Generation Performs Batch Detokenisation After Decode Loop Rather Than Per-Token Streaming (`PARTIALLY IMPLEMENTED`)
- **Severity:** Low-Medium (`P1` UX/runtime polish)
- **Owner:** **P**
- **Description:** In `src/app/main.cpp`, generated token IDs are accumulated into `generated_ids` during the decode loop and decoded in a single batch call (`tokenizer.decode(generated_ids, &generated_text)`) after generation finishes.
- **Resolution Plan:** Add incremental prefix-diff decoding inside the decode loop so each newly generated token piece is flushed to `std::cout` immediately while preserving SentencePiece leading-space (` `) semantics.

### `KI-014` — Single-Threaded GQA Attention (`P2` Optimization)
- **Severity:** Low (`P2` optimization)
- **Owner:** **U**
- **Description:** `scalar::attention_gqa` runs on a single thread across the 32 query heads. Precomputing `[max_seq_len, head_dim / 2]` `cos`/`sin` tables in `TransformerEngine::init()` has resolved RoPE overhead (`KI-006`); parallelizing `attention_gqa` for long contexts (`pos >= 256`) via ThreadPool will further reduce non-matmul latency.

### `KI-018` — 16-Thread SMT Wake-Up Overhead on Small `256x2048` Attention K/V Projections (`P2` Optimization)
- **Severity:** Low (`P2` optimization)
- **Owner:** **I**
- **Description:** On AMD Ryzen 7 7745HX (`benchmarks/results/benchmarks_optimised.csv`), `256x2048` Attention K/V projections achieve `0.0103–0.0112 ms` (`93.6–101.8 GFLOPS`) at 8 threads (physical cores), but regress to `0.0211–0.0336 ms` (`31.2–49.7 GFLOPS`) at 16 threads due to SMT synchronization overhead on a `144 KB` matrix. Also, `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` Section 2.5 has two minor text typos when quoting `benchmarks_prefetch.csv` (`0.0693 ms` / `121.05 GFLOPS` on `2048x2048` 8T and `0.1676 ms` / `137.64 GFLOPS` on `2048x5632` 8T) and machine-local links that should be cleaned up.

---

## 2. Resolved Issues Ledger

| ID | Summary | Resolved By & Commit | Verification Evidence |
| :--- | :--- | :--- | :--- |
| `KI-001` | `Q4_0` `.ie` loader & converter unimplemented in M1 | **S** (`cb176b3`) | `loader_validation` CTest (`6/6`) + `convert_to_inference_engine.py --self-test` |
| `KI-002` | `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` delegated to scalar in M1 | **I** (`f76ab55`, `58c604d`, `b05c4e4`) | `kernels_simd` CTest (`8/8`) + Ryzen 7 7745HX benchmarks (`21.97x` on `2048x2048` 8T) |
| `KI-003` | SentencePiece C++ library not linked in M1 (`Tokenizer` returned `NotImplemented`) | **P** (`ce769c8`, `5929a17`) | `tokenizer` CTest (`4/4`) + `verify_tokenizer_parity.py --self-test` (`15/15` prompts) |
| `KI-004` | Local `model.safetensors` download incomplete | **S & U** (Session 009) | Downloaded 100% (`2,200,119,864` bytes), converted to `tinyllama-1.1b-q4_0.ie` and `tinyllama-1.1b-q4_0-mixed.ie`, verified E2E generation |
| `KI-005` | `Sampler` temperature / top-p sampling unimplemented in M1 | **P** (`ce769c8`) | `sampler` CTest (`6/6`) |
| `KI-006` | RoPE computed runtime trigonometric calls on every token | **U** (Session 009) | Precomputed `rope_cos_` / `rope_sin_` tables in `TransformerEngine::init()`, executed via `scalar::rope_inplace_cached_f32` |
| `KI-007` | CLI generation loop & `--benchmark` harness stubbed in M1 | **P** (`ce769c8`) | Synthetic 1-layer `.ie` + real `tokenizer.model` E2E CLI benchmark (`329.1 tok/s` decode) |
| `KI-011` | Duplicate `TransformerWeightBindings` definitions across `loader.hpp` and `transformer.hpp` | **S & U** (`6dd4fed`, `955bd91`) | Unified in `include/inference_engine/loader.hpp` (`ADR-009`) |
| `KI-012` | `TransformerEngine::forward_token` rejected `Q4_0` weights and ignored `ThreadPool` | **U** (`ce1026a`) | `transformer_q4` CTest (`7/7`) + `verify_transformer_q4.py` (`cos_sim >= 0.996359`) |
| `KI-013` | `ThreadPool::parallel_for` lacked worker exception propagation (`std::terminate` risk) | **I** (`58c604d`) | `std::exception_ptr` capture/rethrow (`ADR-010`), verified in `kernels_simd` CTest |
