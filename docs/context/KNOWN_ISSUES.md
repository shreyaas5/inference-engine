# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Limitations & Resolved Ledger

> **Last Updated:** 2026-11-01  
> **Consolidated Baseline:** `integration/m4-full-inference` (`e46c06e`)  
> **Active Milestone:** Milestone M5 (Real TinyLlama 22-Layer End-to-End Streaming Delivery)

---

## 1. Active Issues & Limitations

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
| `KI-004` | Local `model.safetensors` download incomplete | **S & Team** (M5 Sprint) | Downloaded 100% (`2,200,119,864` bytes), converted to `tinyllama-1.1b-q4_0.ie` (`619,113,792` B) and `tinyllama-1.1b-fp32.ie` (`4,400,213,312` B), verified E2E generation |
| `KI-005` | `Sampler` temperature / top-p sampling unimplemented in M1 | **P** (`ce769c8`) | `sampler` CTest (`6/6`) |
| `KI-006` | RoPE computed runtime trigonometric calls on every token | **U** (`fb6420b`) | Precomputed `rope_cos_` / `rope_sin_` tables in `TransformerEngine::init()`, executed via `scalar::rope_inplace_cached_f32` |
| `KI-007` | CLI generation loop & `--benchmark` harness stubbed in M1 | **P** (`ce769c8`) | Synthetic 1-layer `.ie` + real `tokenizer.model` E2E CLI benchmark (`329.1 tok/s` decode) |
| `KI-011` | Duplicate `TransformerWeightBindings` definitions across `loader.hpp` and `transformer.hpp` | **S & U** (`6dd4fed`, `955bd91`) | Unified in `include/inference_engine/loader.hpp` (`ADR-009`) |
| `KI-012` | `TransformerEngine::forward_token` rejected `Q4_0` weights and ignored `ThreadPool` | **U** (`ce1026a`) | `transformer_q4` CTest (`7/7`) + `verify_transformer_q4.py` (`cos_sim >= 0.996359`) |
| `KI-013` | `ThreadPool::parallel_for` lacked worker exception propagation (`std::terminate` risk) | **I** (`58c604d`) | `std::exception_ptr` capture/rethrow (`ADR-010`), verified in `kernels_simd` CTest |
| `KI-015` | Remote feature branches pending consolidation into baseline | **Team** (`97cf8c3`, `e46c06e`) | Consolidated into `integration/m4-full-inference`, tracking `origin/integration-m4-full-inference`, PR #6 open with 100% green CI |
| `KI-016` | ARM64 NEON kernels lacked 2-row tiling & benchmark hardcoded AVX2 | **I** (`1c046cb`, Session 015) | NEON 2-row tiling in `kernels_neon.cpp` + `benchmarks/bench_kernels.cpp` uses `detect_best_backend()` and `--backend` |
| `KI-017` | CLI generation lacked streaming token output | **P** (`eaaeceb`, Session 016) | Incremental prefix-diff detokenisation flushes tokens in real time |
| `KI-018` | 16T SMT overhead on small matrices & prefetch CSV typos | **I** (Session 015) | Adaptive chunk cap in `dispatch.cpp` (`out_dim <= 256`) + relative links and prefetch CSV reconciled in `CPU_KERNEL_BENCHMARKS.md` |
| `KI-019` | Inference Studio blank response text display bug | **Antigravity** (`d2430fa`, Session 022) | Emitted both `delta` and `text` in `studio_worker.hpp`; updated `App.tsx` to handle `event.delta` and sync `generation_completed`; disabled static asset caching in `index.ts` with `no-cache, no-store, must-revalidate`; verified live via `test_poem.mjs` |
