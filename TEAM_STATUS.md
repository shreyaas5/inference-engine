# `TEAM_STATUS.md` — Team Ownership, Branches, Issues & Status

**Last Updated:** 2026-10-31T23:05:00+05:30
*(Policy: Never invent progress updates or GitHub usernames for teammates. Only verified repository facts are recorded here.)*

---

## 1. Team Roster & Hardware Matrix

| Member | Role | Hardware | OS / SIMD Target | Latest Feature Branch & Tip SHA | Verified Identity |
|---|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4, 24 GB RAM, 1 TB SSD | macOS ARM64 (`NEON`) | `feature/real-checkpoint-validation` (`4515579`), `feature/model-format-q4` (`cb176b3`), `feature/model-loader` (`aa40368`) | `shreyaas5` (`Shreyaas <shreyaas5@users.noreply.github.com>`) |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX, RTX 4060, 16 GB RAM | Windows 11 x86-64 (`AVX2` + `FMA`) | `feature/quantisation-simd-m3` (`58c604d`), `feature/quantisation-simd` (`f76ab55`) | `Ishaan <ishaan.mishra2308@gmail.com>` |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H, 16 GB DDR5 | x86-64 (`AVX2` + `FMA`) | `feature/transformer-q4` (`ce1026a`), `feature/transformer-bindings` (`bce66ae`), `feature/transformer` (`5115467`) | `Utkarsh Gupta <utkarshgupta.vk@gmail.com>` |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U, 8 GB RAM | Windows x86-64 (`AVX2` runtime checked) | `feature/runtime-tokenizer` (`ce769c8`) | `Pranav <pranav1006@users.noreply.github.com>` |

---

## 2. Per-Member Module Ownership & Current Status

### Member S — Project Lead, Model Format & Loader
- **Feature Branches:** `feature/model-loader` (`aa40368`), `feature/model-format-q4` (`cb176b3`), `feature/real-checkpoint-validation` (`4515579`) — all merged into `integration-ps08-team-sync-2026-10-31`.
- **Assigned GitHub Issue:** [#1 — `[M1][Owner: S] Implement TinyLlama safetensors-to-.ie FP32 converter and tensor binding validation`](https://github.com/shreyaas5/inference-engine/issues/1)
- **Status:** **M1 & M2 CONVERTER / LOADER / BINDING & REAL-TENSOR QUANT ERROR EVALUATION COMPLETE**.
  - Streaming `BF16 -> FP32` and `BF16 -> Q4_0` `.ie` converter (`tools/convert_to_inference_engine.py`), `TensorView::data_q4_0()`, and mixed-precision `--keep-embeddings-fp32` `ModelLoader::bind_transformer_weights` implemented and verified (`loader_validation`).
  - Live 201-tensor `TinyLlama-1.1B-Chat-v1.0` header inspection and 9-role real-weight `Q4_0` error evaluation (`tools/inspect_checkpoint.py --eval-real-quant-error`) verified.
  - Pending: Full 2.20 GB `model.safetensors` local download and offline `.ie` export.

---

### Member I — Quantisation, SIMD Kernels & ThreadPool
- **Feature Branches:** `feature/quantisation-simd` (`f76ab55`), `feature/quantisation-simd-m3` (`58c604d`) — merged into `integration-ps08-team-sync-2026-10-31`.
- **Assigned GitHub Issue:** [#2 — `[M1/M2][Owner: I] Validate Q4_0 upstream block parity, CPU feature detection, and SIMD kernel dispatch`](https://github.com/shreyaas5/inference-engine/issues/2)
- **Status:** **M1, M2 & M3 COMPLETE (NEON & NATIVE X86-64 AVX2/FMA VERIFIED)**.
  - Bit-exact upstream GGML `Q4_0` block parity verified (`KI-007`).
  - ARM64 NEON `Q4_0` & `FP32` dot/matmul kernels verified natively on Apple Silicon M4.
  - x86-64 AVX2/FMA `Q4_0` & `FP32` kernels and CPUID Leaf 1/7 + XGETBV runtime detection verified natively on AMD Ryzen 7 7745HX (`KI-013` resolved; 16.79x–20.07x speedup over scalar).
  - `ThreadPool` dynamic task chunking (`ADR-009`) and `std::exception_ptr` propagation (`ADR-010`) verified with 1,000-cycle stress tests.

---

### Member U — Scalar Kernels, KV Cache & Transformer Forward Pass
- **Feature Branches:** `feature/transformer` (`5115467`), `feature/transformer-bindings` (`bce66ae`), `feature/transformer-q4` (`ce1026a`) — merged into `integration-ps08-team-sync-2026-10-31`.
- **Assigned GitHub Issue:** [#3 — `[M1][Owner: U] Implement 22-layer FP32 scalar Transformer forward pass and multi-token golden validation`](https://github.com/shreyaas5/inference-engine/issues/3)
- **Status:** **M1 & M2/M3 `FP32` + `Q4_0` TRANSFORMER COMPLETE (SYNTHETIC & NUMPY VERIFIED)**.
  - 16-step Llama decoder forward pass and GQA causal attention kernel implemented and verified against independent NumPy math (`max_diff = 6.56e-7` for `FP32`).
  - `TransformerEngine::matmul()` dynamically dispatches `FP32` and `Q4_0` weights through `dispatch_matmul_f32` and `dispatch_matmul_q4_0_f32` with `ThreadPool` support and on-the-fly single-row `dequantize_row_q4_0` for `Q4_0` `embed_tokens` (`KI-012` resolved).
  - Verified against independent Python NumPy `Q4_0` reference (`tools/verify_transformer_q4.py`) to 5 decimal digits across prefill and 4 decode steps.
  - Pending: Real 22-layer TinyLlama checkpoint golden run.

---

### Member P — Tokenizer, Sampler, CLI Runtime & Benchmarks
- **Feature Branch:** `feature/runtime-tokenizer` (`ce769c8`) — merged into `integration-ps08-team-sync-2026-10-31`.
- **Assigned GitHub Issue:** [#4 — `[M1][Owner: P] Integrate SentencePiece C++ processor and verify TinyLlama tokenizer & chat template parity`](https://github.com/shreyaas5/inference-engine/issues/4)
- **Status:** **M1, M3 & M4 IMPLEMENTATION COMPLETE & CTEST VERIFIED (`8/8` SUITES PASSING)**.
  - SentencePiece C++ integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`) verified against the real 32,000-piece `TinyLlama-1.1B-Chat-v1.0` `tokenizer.model` in CTest (`tokenizer` suite — `KI-003` resolved).
  - Seeded temperature/top-p nucleus `Sampler` (`src/sampler/sampler.cpp`, `tests/test_sampler.cpp`) verified in CTest (`sampler` suite).
  - Full CLI generation loop and `--benchmark` harness (`src/app/main.cpp`) compiled and verified on macOS ARM64.
  - Pending: Real 22-layer TinyLlama checkpoint CLI generation and 4-laptop benchmark table.
