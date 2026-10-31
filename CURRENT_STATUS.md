# `CURRENT_STATUS.md` — Single Source of Truth for Project Status

> **Last Updated:** 2026-11-01
> **Current Active Branch:** `integration/m4-full-inference`
> **Consolidated Integration Baseline:** `integration/m4-full-inference` (branched from `main` at `5657d9278b2f59e1f1ae0605baf1543cb860fde7`)
> **Protected / Shared Baselines:** `main` (`5657d92`), `origin/integration` (`783b8f4`)
> **Integrated Feature Branches:**
> - Member U: `origin/feature/transformer-runtime` (`fb6420b` — RoPE precomputation, `forward_prompt` API, real TinyLlama E2E validation)
> - Member I: `origin/feature/cpu-kernel-optimisation` (`cb8f08d` — AVX2 & NEON 2-row tiling, vectorized scale accumulation, adaptive ThreadPool chunking, `inference_engine_bench` runner, `ADR-012`)

---

## 1. Executive Summary & Milestone State

- **Milestone M0 (Architecture, `.ie` Format, CMake & Scaffold):** **COMPLETE (`VERIFIED`)** (`cb85411`).
- **Milestone M1 (`.ie` Loader, `FP32` Converter, Scalar Kernels, `KvCache`, `FP32` Transformer):** **COMPLETE (`VERIFIED` / `TESTED`)** (`177fe49` on `main`).
- **Milestone M2 (`Q4_0` Packing, NEON/AVX2 Kernels, `ThreadPool`, `Q4_0` & Mixed Transformer, SentencePiece Tokenizer, Sampler, CLI):** **CODE COMPLETE & INTEGRATED (`5657d92`).**
- **Milestone M3 (`Q4_0` Optimization, SIMD Tiling, Real Checkpoint Validation & Cross-Platform Benchmarking):** **INTEGRATED ON `integration/m4-full-inference`:**
  - **Completed:**
    - **U (`fb6420b`):** Precomputed RoPE frequency lookup tables in `TransformerEngine::init()` (`KI-006` resolved via `rope_inplace_cached_f32`), added `forward_prompt()` multi-token prefill, verified real 22-layer execution in `docs/validation/REAL_TINYLLAMA_TRANSFORMER_VALIDATION.md` (pure `Q4_0`: 20.89 tok/s decode, 698.71 MiB peak RSS).
    - **I (`cb8f08d`):** AVX2 & NEON 2-row register tiling (`matmul_q4_0_f32`, `matmul_f32`), vectorized scale accumulation (`compute_block_unscaled_dot_vec`), contiguous 2-row `dispatch.cpp` chunk scheduling with adaptive small-matrix cap (`out_dim <= 256`), backend-agnostic `inference_engine_bench` runner (`detect_best_backend()`, `--backend`), and audited benchmark figures/relative links in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` (`ADR-012`).
    - **P (`eaaeceb`):** Real-checkpoint SentencePiece C++ vs Python parity (`15/15` prompts exact match against `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`), subsegment encoding, seeded 64-bit `xorshift*` `Sampler`, and CLI `--benchmark` harness.
  - **Remaining Actions:**
    1. Integrate Member P's streaming CLI updates (`eaaeceb`).
    2. Resume and finish downloading `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` on local development machine (currently `325,472,256` / `2,200,120,064` bytes).
    3. Run benchmark suite across multi-architecture team matrix.

---

## 2. Build & Automated Test Status (`VERIFIED`)

| Verification Target | Command | Result |
| :--- | :--- | :--- |
| CMake Configure & Release Build | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON && cmake --build build --parallel 4` | **PASS (`0` warnings)** |
| CTest `1/8`: `loader_validation` | `./build/inference_engine_tests --suite loader_validation` | **PASS (`15/15` checks)** |
| CTest `2/8`: `quant_packing` | `./build/inference_engine_tests --suite quant_packing` | **PASS (`4/4` checks)** |
| CTest `3/8`: `kernels_scalar` | `./build/inference_engine_tests --suite kernels_scalar` | **PASS (`7/7` checks)` |
| CTest `4/8`: `kernels_simd` | `./build/inference_engine_tests --suite kernels_simd` | **PASS (`8/8` checks + odd-row tails in `b05c4e4`)** |
| CTest `5/8`: `kv_cache` | `./build/inference_engine_tests --suite kv_cache` | **PASS (`4/4` checks)** |
| CTest `6/8`: `tokenizer` | `./build/inference_engine_tests --suite tokenizer` | **PASS (`4/4` checks on real `tokenizer.model`)** |
| CTest `7/8`: `sampler` | `./build/inference_engine_tests --suite sampler` | **PASS (`6/6` checks)** |
| CTest `8/8`: `golden_forward` | `./build/inference_engine_tests --suite golden_forward` | **PASS (`3/3` checks)** |
| Python Converter Self-Test | `python3 tools/convert_to_inference_engine.py --self-test` | **PASS (`FP32`, `Q4_0`, `--keep-embeddings-fp32`)** |
| Python `Q4_0` Transformer Parity | `python3 tools/verify_transformer_q4.py` | **PASS (`cos_sim >= 0.996359`)** |
| Python Tokenizer Parity (`15/15`) | `python3 tools/verify_tokenizer_parity.py --self-test` | **PASS (`15/15` prompts on real `tokenizer.model`)** |

---

## 3. Top 3 Project Priorities

1. **Priority 1 (`P0` — Integration Sprint):** Complete M4 integration of Member P's streaming CLI branch (`origin/feature/runtime-tokenizer`).
2. **Priority 2 (`P0` — Checkpoint Acquisition):** Resume and complete downloading `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` on local development machine.
3. **Priority 3 (`P1` — Benchmarking):** Execute cross-platform `./build/inference_engine_bench` on ARM64 Apple Silicon and Intel laptops.
