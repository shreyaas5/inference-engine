# `CURRENT_STATUS.md` — Single Source of Truth for Project Status

> **Last Updated:** 2026-11-01
> **Current Active Branch:** `feature/transformer-runtime`
> **Consolidated Integration Baseline:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Protected / Shared Baselines:** `main` (`177fe492eb349ed17a2ffaeda908d92a0d7e0935`), `integration` (`783b8f44e5d2fc2ecf830559139cd19ed8a328fd`)

---

## 1. Executive Summary & Milestone State

- **Milestone M0 (Architecture, `.ie` Format, CMake & Scaffold):** **COMPLETE (`VERIFIED`)** (`cb85411`).
- **Milestone M1 (`.ie` Loader, `FP32` Converter, Scalar Kernels, `KvCache`, `FP32` Transformer):** **COMPLETE (`VERIFIED` / `TESTED`)** (`177fe49` on `main`).
- **Milestone M2 (`Q4_0` Packing, NEON/AVX2 Kernels, `ThreadPool`, `Q4_0` & Mixed Transformer, SentencePiece Tokenizer, Sampler, CLI):** **CODE COMPLETE & INTEGRATED (`33d4bdc`) (`8 / 8` CTest suites passing).**
- **Milestone M3 (CPU Kernel Optimization, Full 22-Layer Pretrained Checkpoint Verification & Cross-Platform Benchmarking):** **ACTIVE / MAJOR DELIVERABLES ACHIEVED:**
  - **Completed in M3:**
    - **U (Transformer Engine & Real Validation):** Precomputed RoPE frequency lookup tables in `TransformerEngine::init()` (`KI-006` resolved via `rope_inplace_cached_f32`), added `forward_prompt()` multi-token prefill, verified real 22-layer execution.
    - **S & U (Checkpoint Acquisition & Conversion):** Downloaded 100% of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2,200,119,864` bytes). Converted to pure `Q4_0` (`619,113,792` B) and mixed `Q4_0` (`1,069,673,792` B) with measured quantization metrics (`MAE = 1.385e-3`, `RMSE = 1.744e-3`, `Rel L2 = 0.0883`).
    - **U, P & Full Pipeline (End-to-End Real Inference):** Verified real-checkpoint CLI generation with TinyLlama chat template generating coherent text:
      - Pure `Q4_0`: TTFT `1200.07 ms`, decode `20.89 tokens/sec`, peak RSS `698.71 MiB` on 4 CPU threads.
      - Mixed `Q4_0`: TTFT `1338.06 ms`, decode `19.07 tokens/sec`, peak RSS `1126.42 MiB` on 4 CPU threads.
    - **I (`58c604d` & `b05c4e4`):** Native AMD Ryzen 7 7745HX AVX2 verification, `ThreadPool` `std::exception_ptr` propagation, AVX2 2-row register tiling (`avx2::matmul_q4_0_f32`, `avx2::matmul_f32`), vectorized per-block scale accumulation (`compute_block_unscaled_dot_vec`), contiguous 2-row `dispatch.cpp` chunk scheduling, and `inference_engine_bench` CSV artifacts (`21.97x` speedup on `2048x2048` 8T, `162.80 GFLOPS` on `2048x5632` 16T).
    - **P (`ce769c8`):** Real-checkpoint SentencePiece C++ vs Python parity (`15/15` prompts exact match against `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`), seeded 64-bit `xorshift*` `Sampler`, and CLI `--benchmark` harness.
  - **Remaining in M3:**
    1. Merge `feature/transformer-runtime` and `origin/feature/cpu-kernel-optimisation` into integration baseline.
    2. Port 2-row register tiling and deferred end-of-row horizontal reduction (`vaddvq_f32`) to ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`) and make `benchmarks/bench_kernels.cpp` backend-agnostic (`detect_best_backend()`).
    3. Add incremental per-token streaming detokenisation to `src/app/main.cpp`.

---

## 2. Build & Automated Test Status (`VERIFIED`)

| Verification Target | Command | Result |
| :--- | :--- | :--- |
| CMake Configure & Release Build | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON && cmake --build build --parallel 4` | **PASS (`0` warnings)** |
| CTest `1/8`: `loader_validation` | `./build/inference_engine_tests --suite loader_validation` | **PASS (`6/6` sub-tests)** |
| CTest `2/8`: `quant_packing` | `./build/inference_engine_tests --suite quant_packing` | **PASS (`4/4` sub-tests)** |
| CTest `3/8`: `kernels_scalar` | `./build/inference_engine_tests --suite kernels_scalar` | **PASS (`7/7` sub-tests)** |
| CTest `4/8`: `kernels_simd` | `./build/inference_engine_tests --suite kernels_simd` | **PASS (`8/8` sub-tests)** |
| CTest `5/8`: `kv_cache` | `./build/inference_engine_tests --suite kv_cache` | **PASS (`4/4` sub-tests)** |
| CTest `6/8`: `tokenizer` | `./build/inference_engine_tests --suite tokenizer` | **PASS (`4/4` sub-tests on real `tokenizer.model`)** |
| CTest `7/8`: `sampler` | `./build/inference_engine_tests --suite sampler` | **PASS (`6/6` sub-tests)** |
| CTest `8/8`: `golden_forward` | `./build/inference_engine_tests --suite golden_forward` | **PASS (`3/3` sub-tests)** |
| Python Converter Self-Test | `python3 tools/convert_to_inference_engine.py --self-test` | **PASS (`FP32`, `Q4_0`, `--keep-embeddings-fp32`)** |
| Python `Q4_0` Transformer Parity | `python3 tools/verify_transformer_q4.py` | **PASS (`cos_sim >= 0.996359`)** |
| Python Tokenizer Parity (`15/15`) | `python3 tools/verify_tokenizer_parity.py --self-test` | **PASS (`15/15` prompts on real `tokenizer.model`)** |

---

## 3. Top 3 Project Priorities

1. **Priority 1 (`P0` — Team Integration):** Merge `feature/transformer-runtime` and `origin/feature/cpu-kernel-optimisation` into the shared integration branch, consolidating real-checkpoint inference and AVX2 2-row tiling.
2. **Priority 2 (`P1` — Member I):** Port vectorized scale accumulation and 2-row register tiling from `src/kernels/avx2/kernels_avx2.cpp` to `src/kernels/neon/kernels_neon.cpp`, and update `benchmarks/bench_kernels.cpp` to dispatch via `detect_best_backend()`.
3. **Priority 3 (`P1` — Member P):** Add incremental per-token streaming detokenisation to `src/app/main.cpp`.
