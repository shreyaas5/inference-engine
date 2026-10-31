# `TEAM_STATUS.md` — Team Ownership, Branches, Issues & Status

**Last Updated:** 2026-10-31
*(Policy: Never invent progress updates or GitHub usernames for teammates. Only verified repository facts are recorded here.)*

---

## 1. Team Roster & Hardware Matrix

| Member | Role | Hardware | OS / SIMD Target | Feature Branch | Verified Identity |
|---|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4, 24 GB RAM, 1 TB SSD | macOS ARM64 (`NEON`) | `feature/model-loader` (`aa40368`, merged to `main`) | `shreyaas5` (GitHub) |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX, RTX 4060, 16 GB RAM | x86-64 (`AVX2` + `FMA`) | `feature/quantisation-simd` (`f76ab55`, merged to `integration`) | `Ishaan <ishaan.mishra2308@gmail.com>` (Verified in commits `a948734` & `f76ab55`) |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H, 16 GB DDR5 | x86-64 (`AVX2` + `FMA`) | `feature/transformer` (`5115467`, merged to `main`) | `Utkarsh Gupta <utkarshgupta.vk@gmail.com>` (Verified in commit `5115467`) |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U, 8 GB RAM | x86-64 (`AVX2` runtime checked) | `feature/runtime-tokenizer` (local `783b8f4` base + uncommitted changes; remote feature still `cb85411`) | *Pending* |

---

## 2. Per-Member Module Ownership & Current Status

### Member S — Project Lead & Model Loader
- **Feature Branch:** `feature/model-loader` (merged into `main` via `177fe49`)
- **Assigned GitHub Issue:** [#1 — `[M1][Owner: S] Implement TinyLlama safetensors-to-.ie FP32 converter and tensor binding validation`](https://github.com/shreyaas5/inference-engine/issues/1)
- **Status:** **COMPLETE ON `MAIN` (SYNTHETIC / METADATA)**. Full 2.2 GB TinyLlama payload download and offline `.ie` conversion pending.

---

### Member I — Quantisation & CPU Optimisation
- **Feature Branch:** `feature/quantisation-simd` (`f76ab55`, merged into `integration` in commit `967f072`)
- **Assigned GitHub Issue:** [#2 — `[M1/M2][Owner: I] Validate Q4_0 upstream block parity, CPU feature detection, and SIMD kernel dispatch`](https://github.com/shreyaas5/inference-engine/issues/2)
- **Status:** **COMPLETE ON `INTEGRATION` (NEON VERIFIED, AVX2 PENDING NATIVE X86 RUN)**.
  - Bit-exact upstream GGML Q4_0 block parity verified (`KI-007`).
  - ARM64 NEON Q4_0 & FP32 dot/matmul kernels verified natively on Apple Silicon.
  - x86-64 AVX2/FMA Q4_0 & FP32 kernels and CPUID runtime detection implemented.
  - ThreadPool task chunking optimization implemented.
  - Dispatched matmul implemented and verified.
  - Pending: Native test run on x86-64 hardware/CI; wiring into `TransformerEngine::forward_token` for M2/M3.

---

### Member U — Scalar Inference & Transformer
- **Feature Branch:** `feature/transformer` (merged into `main` via `177fe49`)
- **Assigned GitHub Issue:** [#3 — `[M1][Owner: U] Implement 22-layer FP32 scalar Transformer forward pass and multi-token golden validation`](https://github.com/shreyaas5/inference-engine/issues/3)
- **Status:** **COMPLETE ON `MAIN` (SYNTHETIC / NUMPY VERIFIED)**.
  - 16-step Llama decoder forward pass and GQA causal attention kernel implemented and verified against independent NumPy math.
  - Pending: Real 22-layer TinyLlama checkpoint golden run; wiring `dispatch_matmul_q4_0_f32` in M2/M3.

---

### Member P — Tokenizer, Runtime & Benchmarks
- **Feature Branch:** `feature/runtime-tokenizer` (local branch fast-forwarded to `783b8f4`; P changes uncommitted; origin feature still at `cb85411`)
- **Assigned GitHub Issue:** [#4 — `[M1][Owner: P] Integrate SentencePiece C++ processor and verify TinyLlama tokenizer & chat template parity`](https://github.com/shreyaas5/inference-engine/issues/4)
- **Status:** **IMPLEMENTED IN WORKTREE; VALIDATION PENDING**.
  - SentencePiece C++ integration and pinned TinyLlama tokenizer parity harness are in place.
  - Seeded temperature/top-p sampler and FP32 generation CLI are implemented.
  - Focused sampler test and offline Python parity self-test passed.
  - Pending: full CMake/CTest, Hugging Face parity, and inference with the real TinyLlama checkpoint. CMake/CTest are unavailable on the current host.
