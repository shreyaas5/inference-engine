# `CURRENT_STATUS.md` — Single Source of Truth for Project Status

> **Last Updated:** 2026-11-01
> **Current Active Branch:** `docs/team-handoff-reconciliation` (branched from `integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Consolidated Integration Baseline:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Protected / Shared Baselines:** `main` (`177fe492eb349ed17a2ffaeda908d92a0d7e0935`), `integration` (`783b8f44e5d2fc2ecf830559139cd19ed8a328fd`)
> **Unmerged Remote Teammate Branches Ready for Integration:**
> - `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1` — Member I: AVX2 2-row register tiling, vectorized scale accumulation, contiguous 2-row `dispatch.cpp` chunking, `inference_engine_bench` target, Ryzen 7 7745HX benchmark CSVs, `ADR-012`)
> - `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d` — Member U: GCC 16 `<cstdint>` fix for `sentencepiece-static` in `CMakeLists.txt`)

---

## 1. Executive Summary & Milestone State

- **Milestone M0 (Architecture, `.ie` Format, CMake & Scaffold):** **COMPLETE (`VERIFIED`)** (`cb85411`).
- **Milestone M1 (`.ie` Loader, `FP32` Converter, Scalar Kernels, `KvCache`, `FP32` Transformer):** **COMPLETE (`VERIFIED` / `TESTED`)** (`177fe49` on `main`).
- **Milestone M2 (`Q4_0` Packing, NEON/AVX2 Kernels, `ThreadPool`, `Q4_0` & Mixed Transformer, SentencePiece Tokenizer, Sampler, CLI):** **CODE COMPLETE & INTEGRATED ON `integration-ps08-team-sync-2026-10-31` (`33d4bdc`) (`9 / 9` CTest suites passing).**
- **Milestone M3 (CPU Kernel Optimization, Full 22-Layer Pretrained Checkpoint Verification & Cross-Platform Benchmarking):** **IN PROGRESS:**
  - **Completed in M3:**
    - **S (`4515579`):** Live 201-tensor header validation and real-weight `Q4_0` quantisation error audit across all 9 functional tensor roles on `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`MeanCosSim >= 0.9888`, `SQNR = 16.51–16.82 dB`).
    - **I (`58c604d` & `b05c4e4`):** Native AMD Ryzen 7 7745HX AVX2 verification, `ThreadPool` `std::exception_ptr` propagation, AVX2 2-row register tiling (`avx2::matmul_q4_0_f32`, `avx2::matmul_f32`), vectorized per-block scale accumulation (`compute_block_unscaled_dot_vec`), contiguous 2-row `dispatch.cpp` chunk scheduling, and `inference_engine_bench` CSV artifacts (`21.97x` speedup on `2048x2048` 8T, `162.80 GFLOPS` on `2048x5632` 16T).
    - **P (`ce769c8`):** Real-checkpoint SentencePiece C++ vs Python parity (`15/15` prompts exact match against `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`), seeded 64-bit `xorshift*` `Sampler`, and CLI `--benchmark` harness.
  - **Remaining in M3:**
    1. Merge `origin/feature/transformer-runtime` (`e64078c`) and `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) into the consolidated integration baseline.
    2. Resume and finish downloading `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`325,472,256` / `2,200,120,064` bytes currently on disk), export 22-layer `.ie` files (`FP32`: `4,400,213,312` B; pure `Q4_0`: `619,113,792` B; mixed `Q4_0`: `1,069,673,792` B), and run full 22-layer golden forward parity and CLI benchmarks.
    3. Port 2-row register tiling and deferred end-of-row horizontal reduction (`vaddvq_f32`) to ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`) and make `benchmarks/bench_kernels.cpp` backend-agnostic (`detect_best_backend()`).
    4. Add incremental per-token streaming detokenisation to `src/app/main.cpp`.

---

## 2. Build & Automated Test Status (`VERIFIED`)

| Verification Target | Command | Result |
| :--- | :--- | :--- |
| CMake Configure & Release Build | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON && cmake --build build --parallel 4` | **PASS (`0` warnings)** |
| CTest `1/9`: `loader_validation` | `./build/inference_engine_tests --suite loader_validation` | **PASS (`6/6` sub-tests)** |
| CTest `2/9`: `quant_packing` | `./build/inference_engine_tests --suite quant_packing` | **PASS (`4/4` sub-tests)** |
| CTest `3/9`: `kernels_scalar` | `./build/inference_engine_tests --suite kernels_scalar` | **PASS (`7/7` sub-tests)** |
| CTest `4/9`: `kernels_simd` | `./build/inference_engine_tests --suite kernels_simd` | **PASS (`8/8` sub-tests + odd-row tails in `b05c4e4`)** |
| CTest `5/9`: `kv_cache` | `./build/inference_engine_tests --suite kv_cache` | **PASS (`4/4` sub-tests)** |
| CTest `6/9`: `tokenizer` | `./build/inference_engine_tests --suite tokenizer` | **PASS (`4/4` sub-tests on real `tokenizer.model`)** |
| CTest `7/9`: `sampler` | `./build/inference_engine_tests --suite sampler` | **PASS (`6/6` sub-tests)** |
| CTest `8/9`: `golden_forward` | `./build/inference_engine_tests --suite golden_forward` | **PASS (`3/3` sub-tests)** |
| CTest `9/9`: `transformer_q4` | `./build/inference_engine_tests --suite transformer_q4` | **PASS (`7/7` sub-tests)** |
| Python Converter Self-Test | `python3 tools/convert_to_inference_engine.py --self-test` | **PASS (`FP32`, `Q4_0`, `--keep-embeddings-fp32`)** |
| Python `Q4_0` Transformer Parity | `python3 tools/verify_transformer_q4.py` | **PASS (`cos_sim >= 0.996359`)** |
| Python Tokenizer Parity (`15/15`) | `python3 tools/verify_tokenizer_parity.py --self-test` | **PASS (`15/15` prompts on real `tokenizer.model`)** |

---

## 3. Top 3 Project Priorities

1. **Priority 1 (`P0` — Member S):** Merge `origin/feature/transformer-runtime` (`e64078c`) and `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) into the consolidated integration baseline (`33d4bdc`), and complete the download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2.20 GB`) to export the real 22-layer `FP32`, pure `Q4_0`, and mixed `Q4_0` `.ie` models.
2. **Priority 2 (`P0` — Members U & P):** Run `tools/generate_golden_reference.py` and execute full 22-layer pretrained TinyLlama forward-pass parity (`U`) and end-to-end CLI generation + `--benchmark` across 1, 2, 4, and 8 threads (`P`), plus add incremental streaming detokenisation in `src/app/main.cpp` (`P`).
3. **Priority 3 (`P1` — Member I):** Port vectorized scale accumulation and 2-row register tiling from `src/kernels/avx2/kernels_avx2.cpp` (`b05c4e4`) to `src/kernels/neon/kernels_neon.cpp`, and update `benchmarks/bench_kernels.cpp` to dispatch via `detect_best_backend()`.
