# `TEAM_STATUS.md` — Team Ownership, Branches, Issues & Status

**Last Updated:** 2026-10-31  
*(Policy: Never invent progress updates or GitHub usernames for teammates. Only verified repository facts are recorded here.)*

---

## 1. Team Roster & Hardware Matrix

| Member | Role | Hardware | OS / SIMD Target | Feature Branch | GitHub Username |
|---|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4, 24 GB RAM, 1 TB SSD | macOS ARM64 (`NEON`) | `feature/model-loader` | `shreyaas5` (Verified) |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX, RTX 4060, 16 GB RAM | x86-64 (`AVX2` + `FMA`) | `feature/quantisation-simd` | *Unknown / Pending invite* |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H, 16 GB DDR5 | x86-64 (`AVX2` + `FMA`) | `feature/transformer` | *Unknown / Pending invite* |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U, 8 GB RAM | x86-64 (`AVX2` runtime checked) | `feature/runtime-tokenizer` | *Unknown / Pending invite* |

---

## 2. Per-Member Module Ownership & Current Status

### Member S — Project Lead & Model Loader
- **Feature Branch:** `feature/model-loader` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#1 — `[M1][Owner: S] Implement TinyLlama safetensors-to-.ie FP32 converter and tensor binding validation`](https://github.com/shreyaas5/inference-engine/issues/1) (Open)
- **Owned Files & Verified Status:**
  - `src/common/common.cpp`, `src/common/tensor.cpp` — **TESTED** (Dtype helpers, CPU backend detection, `TensorDesc` validation).
  - `src/loader/loader.cpp` — **TESTED** (`.ie` v1 binary parser, `build_ie_image`, `mmap` loader, alignment/bounds/overlap checks).
  - `tests/test_loader_validation.cpp` — **TESTED** (Passing in CTest).
  - `tools/inspect_checkpoint.py` — **IN PROGRESS** (CLI and `--print-frozen-contract` implemented; needs local checkpoint test).
  - `tools/convert_to_inference_engine.py` — **IN PROGRESS / STUB** (64-byte header packer implemented; `.safetensors` reader and `.ie` payload writer pending for M1).
  - `CURRENT_STATUS.md` & Context System — **IMPLEMENTED**.
- **Active Blockers:** Needs `safetensors` Python package installed in local environment and local `TinyLlama-1.1B-Chat-v1.0` checkpoint downloaded to run full conversion.

---

### Member I — Quantisation & CPU Optimisation
- **Feature Branch:** `feature/quantisation-simd` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#2 — `[M1/M2][Owner: I] Validate Q4_0 upstream block parity, CPU feature detection, and SIMD kernel dispatch`](https://github.com/shreyaas5/inference-engine/issues/2) (Open)
- **Owned Files & Verified Status:**
  - `src/kernels/quant_q4_0.cpp` — **TESTED** (`fp32_to_fp16`, `fp16_to_fp32`, `quantize_row_q4_0`, `dequantize_row_q4_0` implemented and passing synthetic tests; pending cross-check against external GGML reference).
  - `src/kernels/threading/thread_pool.cpp` — **TESTED** (Persistent `std::thread` pool with `parallel_for`).
  - `src/kernels/dispatch.cpp` — **TESTED** (Runtime dispatch across `Scalar`, `Neon`, `Avx2`).
  - `src/kernels/neon/kernels_neon.cpp` — **IN PROGRESS** (`neon::dot_f32` implemented; `neon::dot_q4_0_f32` currently delegates to scalar).
  - `src/kernels/avx2/kernels_avx2.cpp` — **IN PROGRESS** (`avx2::dot_f32` implemented; `avx2::dot_q4_0_f32` currently delegates to scalar; not yet tested on x86-64 hardware locally).
  - `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp` — **TESTED** (Passing in CTest).
- **Active Blockers:** Pending GitHub collaborator invite to `shreyaas5/inference-engine`.

---

### Member U — Scalar Inference & Transformer
- **Feature Branch:** `feature/transformer` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#3 — `[M1][Owner: U] Implement 22-layer FP32 scalar Transformer forward pass and multi-token golden validation`](https://github.com/shreyaas5/inference-engine/issues/3) (Open)
- **Owned Files & Verified Status:**
  - `src/kernels/scalar/kernels_scalar.cpp` — **IN PROGRESS / PARTIALLY TESTED** (`rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32` implemented and tested; causal GQA multi-head attention kernel still needed).
  - `src/core/kv_cache.cpp` — **TESTED** (`init`, `clear`, `append`, `commit_positions`, `commit_token`, `key_vector`, `value_vector` implemented and passing CTest).
  - `src/transformer/transformer.cpp` — **IN PROGRESS / STUB** (Position indexing scaffolding only; does not yet bind weights or run transformer layer math).
  - `tools/generate_golden_reference.py` — **IN PROGRESS / STUB** (CLI skeleton only).
  - `tests/test_kernels_scalar.cpp`, `tests/test_kv_cache.cpp` — **TESTED** (Passing in CTest).
  - `tests/test_golden_forward.cpp` — **IN PROGRESS** (Tests 12-token position indexing contract on stubbed engine; needs real golden reference comparison in M1).
- **Active Blockers:** Depends on S's `.ie` FP32 converter (`#1`) for full checkpoint loading; pending GitHub collaborator invite.

---

### Member P — Tokenizer, Runtime & Benchmarks
- **Feature Branch:** `feature/runtime-tokenizer` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#4 — `[M1][Owner: P] Integrate SentencePiece C++ processor and verify TinyLlama tokenizer & chat template parity`](https://github.com/shreyaas5/inference-engine/issues/4) (Open)
- **Owned Files & Verified Status:**
  - `src/tokenizer/tokenizer.cpp` — **IN PROGRESS / STUB** (`format_chat_prompt` implemented and tested; `encode`/`decode` use synthetic byte-offset fallback until SentencePiece C++ is linked).
  - `src/sampler/sampler.cpp` — **IN PROGRESS** (`sample_greedy` implemented and tested; stochastic top-p `sample()` delegates to greedy until M4).
  - `src/app/main.cpp` — **IN PROGRESS / STUB** (`--help` and `--info` implemented; inference and benchmark execution loops pending).
  - `tools/verify_tokenizer_parity.py` — **IN PROGRESS** (`--self-test` verifies chat template; full SentencePiece vs HF parity check pending).
  - `.github/workflows/ci.yml` — **IMPLEMENTED** (Ubuntu x86-64 + macOS ARM64 CMake/CTest matrix).
- **Active Blockers:** SentencePiece C++ library integration in `CMakeLists.txt`; pending GitHub collaborator invite.

