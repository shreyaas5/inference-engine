# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Limitations & Resolved Ledger

> **Last Updated:** 2026-11-01
> **Consolidated Baseline:** `integration-ps08-team-sync-2026-10-31` (`ad246cb`)
> **Active Feature Branch:** `feature/cpu-kernel-optimisation` (Member I)

---

## 1. Active Issues & Limitations

### `KI-004` — Local `TinyLlama-1.1B-Chat-v1.0/model.safetensors` Download Is Incomplete (`BLOCKED ON LOCAL MACHINE`)
- **Severity:** High (`P0` blocker for full 22-layer `.ie` conversion and real-weight inference on local workstation)
- **Owner:** **S** (affects **U**, **I**, **P**)
- **Description:** `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` on this local machine is currently `325,472,256` bytes (`310.4 MB`) out of the expected `2,200,120,064` bytes (`2.20 GB`, ~14.8%). While Member U has validated the 2.20 GB checkpoint end-to-end on his workstation (`docs/validation/REAL_TINYLLAMA_TRANSFORMER_VALIDATION.md`), full 22-layer `.ie` model conversion on this local machine requires resuming the remaining `1.87 GB` download.
- **Workaround / Next Step:** Run `curl -L -C -` or `huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 model.safetensors --local-dir models/TinyLlama-1.1B-Chat-v1.0` and then invoke `tools/convert_to_inference_engine.py`.

### `KI-015` — Remote Feature Branches Integration into Consolidated Integration Baseline
- **Severity:** Medium (`P0` integration task)
- **Owner:** **S** (with **I** and **U**)
- **Description:**
  1. Member I pushed `origin/feature/cpu-kernel-optimisation` (`b05c4e4`, `1c046cb`) containing AVX2 and NEON 2-row register tiling, vectorized scale accumulation, contiguous 2-row `dispatch.cpp` chunk scheduling, `inference_engine_bench`, and audited benchmark figures.
  2. Member U pushed `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d`, parent `5929a17`) adding `target_compile_options(sentencepiece-static PRIVATE -include cstdint)` in `CMakeLists.txt` for GCC 16 compatibility.

### `KI-017` — CLI Generation Performs Batch Detokenisation After Decode Loop Rather Than Per-Token Streaming (`PARTIALLY IMPLEMENTED`)
- **Severity:** Low-Medium (`P1` UX/runtime polish)
- **Owner:** **P**
- **Description:** In `src/app/main.cpp`, generated token IDs are accumulated into `generated_ids` during the decode loop and decoded in a single batch call (`tokenizer.decode(generated_ids, &generated_text)`) after generation finishes.
- **Resolution Plan:** Add incremental prefix-diff decoding inside the decode loop so each newly generated token piece is flushed to `std::cout` immediately while preserving SentencePiece leading-space (` `) semantics.

### `KI-014` — Single-Threaded GQA Attention (`P2` Optimization)
- **Severity:** Low (`P2` optimization)
- **Owner:** **U**
- **Description:** `scalar::attention_gqa` runs on a single thread across the 32 query heads. Precomputing `[max_seq_len, head_dim / 2]` `cos`/`sin` tables in `TransformerEngine::init()` has resolved RoPE overhead (`KI-006`); parallelizing `attention_gqa` for long contexts (`pos >= 256`) via ThreadPool will further reduce non-matmul latency.

---

## 2. Resolved Issues Ledger

| ID | Summary | Resolved By & Commit | Verification Evidence |
| :--- | :--- | :--- | :--- |
| `KI-001` | `Q4_0` `.ie` loader & converter unimplemented in M1 | **S** (`cb176b3`) | `loader_validation` CTest (`6/6`) + `convert_to_inference_engine.py --self-test` |
| `KI-002` | `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` delegated to scalar in M1 | **I** (`f76ab55`, `58c604d`, `b05c4e4`, `1c046cb`) | `kernels_simd` CTest (`8/8`) + Ryzen 7 7745HX benchmarks (`21.97x` on `2048x2048` 8T) + NEON 2-row tiling |
| `KI-003` | SentencePiece C++ library not linked in M1 (`Tokenizer` returned `NotImplemented`) | **P** (`ce769c8`, `5929a17`) | `tokenizer` CTest (`4/4`) + `verify_tokenizer_parity.py --self-test` (`15/15` prompts) |
| `KI-004` | Local `model.safetensors` download incomplete | **S & U** (Session 009) | Downloaded 100% (`2,200,119,864` bytes), converted to `tinyllama-1.1b-q4_0.ie` and `tinyllama-1.1b-q4_0-mixed.ie`, verified E2E generation |
| `KI-005` | `Sampler` temperature / top-p sampling unimplemented in M1 | **P** (`ce769c8`) | `sampler` CTest (`6/6`) |
| `KI-006` | RoPE computed runtime trigonometric calls on every token | **U** (Session 009) | Precomputed `rope_cos_` / `rope_sin_` tables in `TransformerEngine::init()`, executed via `scalar::rope_inplace_cached_f32` |
| `KI-007` | CLI generation loop & `--benchmark` harness stubbed in M1 | **P** (`ce769c8`) | Synthetic 1-layer `.ie` + real `tokenizer.model` E2E CLI benchmark (`329.1 tok/s` decode) |
| `KI-011` | Duplicate `TransformerWeightBindings` definitions across `loader.hpp` and `transformer.hpp` | **S & U** (`6dd4fed`, `955bd91`) | Unified in `include/inference_engine/loader.hpp` (`ADR-009`) |
| `KI-012` | `TransformerEngine::forward_token` rejected `Q4_0` weights and ignored `ThreadPool` | **U** (`ce1026a`) | `transformer_q4` CTest (`7/7`) + `verify_transformer_q4.py` (`cos_sim >= 0.996359`) |
| `KI-013` | `ThreadPool::parallel_for` lacked worker exception propagation (`std::terminate` risk) | **I** (`58c604d`) | `std::exception_ptr` capture/rethrow (`ADR-010`), verified in `kernels_simd` CTest |
| `KI-016` | ARM64 NEON kernels lacked 2-row tiling & benchmark hardcoded AVX2 | **I** (`1c046cb`, Session 015) | NEON 2-row tiling in `kernels_neon.cpp` + `benchmarks/bench_kernels.cpp` uses `detect_best_backend()` and `--backend` |
| `KI-018` | 16T SMT overhead on small matrices & prefetch CSV typos | **I** (Session 015) | Adaptive chunk cap in `dispatch.cpp` (`out_dim <= 256`) + relative links and prefetch CSV reconciled in `CPU_KERNEL_BENCHMARKS.md` |
