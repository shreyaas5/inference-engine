# `TEAM_STATUS.md` — Team Ownership, Branches, Issues & Status

**Last Updated:** 2026-10-31T18:46:00+05:30  
*(Policy: Never invent progress updates or GitHub usernames for teammates. Only verified repository facts are recorded here.)*

---

## 1. Team Roster & Hardware Matrix

| Member | Role | Hardware | OS / SIMD Target | Feature Branch | Verified Identity |
|---|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4, 24 GB RAM, 1 TB SSD | macOS ARM64 (`NEON`) | `feature/model-loader` (`aa40368`, merged to `integration`) | `shreyaas5` (GitHub) |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX, RTX 4060, 16 GB RAM | x86-64 (`AVX2` + `FMA`) | `feature/quantisation-simd` (`cb85411`) | *Pending* |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H, 16 GB DDR5 | x86-64 (`AVX2` + `FMA`) | `feature/transformer` (`5115467`, merged to `integration`) | `Utkarsh Gupta <utkarshgupta.vk@gmail.com>` (Verified in commit `5115467`) |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U, 8 GB RAM | x86-64 (`AVX2` runtime checked) | `feature/runtime-tokenizer` (`cb85411`) | *Pending* |

---

## 2. Per-Member Module Ownership & Current Status

### Member S — Project Lead & Model Loader
- **Feature Branch:** `feature/model-loader` (commit `aa40368`, merged into local `integration`)
- **Assigned GitHub Issue:** [#1 — `[M1][Owner: S] Implement TinyLlama safetensors-to-.ie FP32 converter and tensor binding validation`](https://github.com/shreyaas5/inference-engine/issues/1)
- **Owned Files & Verified Status:**
  - `src/common/common.cpp`, `src/common/tensor.cpp` — **TESTED (SYNTHETIC)** & **REFERENCE VERIFIED (METADATA)**.
  - `src/loader/loader.cpp`, `include/inference_engine/loader.hpp` — **TESTED (SYNTHETIC)** (`.ie` v1 binary parser, `build_ie_image`, POSIX `mmap` loader, `bind_transformer_weights()`, 12 malformed-file checks, bit-exact `FP32` preservation).
  - `tests/test_loader_validation.cpp` — **TESTED (SYNTHETIC)** (Passing in CTest).
  - `tools/inspect_checkpoint.py` — **REFERENCE VERIFIED (METADATA)** (Verified live 23,088-byte `model.safetensors` header and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: `201/201` `BF16` tensors).
  - `tools/convert_to_inference_engine.py` — **REFERENCE VERIFIED (SYNTHETIC / METADATA)** (Two-pass streaming `BF16/F16/F32 -> FP32` `.safetensors -> .ie` converter implemented and verified with `--self-test` + C++ `mmap` reload; `--dtype q4_0` scheduled for M2).
- **Remaining for Full M1 Sign-Off:** Download the 2.20 GB `model.safetensors` payload locally and convert the full 4.40 GB `tinyllama-1.1b-fp32.ie` model.

---

### Member I — Quantisation & CPU Optimisation
- **Feature Branch:** `feature/quantisation-simd` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#2 — `[M1/M2][Owner: I] Validate Q4_0 upstream block parity, CPU feature detection, and SIMD kernel dispatch`](https://github.com/shreyaas5/inference-engine/issues/2)
- **Owned Files & Verified Status:**
  - `src/kernels/quant_q4_0.cpp` — **TESTED (SYNTHETIC)** (`fp32_to_fp16`, `fp16_to_fp32`, `quantize_row_q4_0`, `dequantize_row_q4_0` implemented and passing synthetic tests; pending cross-check against external GGML reference).
  - `src/kernels/threading/thread_pool.cpp` — **TESTED (SYNTHETIC)** (Persistent `std::thread` pool with `parallel_for`).
  - `src/kernels/dispatch.cpp` — **TESTED (SYNTHETIC)** (Runtime dispatch across `Scalar`, `Neon`, `Avx2`).
  - `src/kernels/neon/kernels_neon.cpp` — **IN PROGRESS** (`neon::dot_f32` implemented; `neon::dot_q4_0_f32` currently delegates to scalar).
  - `src/kernels/avx2/kernels_avx2.cpp` — **IN PROGRESS** (`avx2::dot_f32` implemented; `avx2::dot_q4_0_f32` currently delegates to scalar; not yet tested on x86-64 hardware locally).
  - `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp` — **TESTED (SYNTHETIC)** (Passing in CTest).

---

### Member U — Scalar Inference & Transformer
- **Feature Branch:** `feature/transformer` (commit `5115467` on `origin/feature/transformer` and `origin/feature/transformer-m1-fp32`, reviewed and merged into local `integration` at `1f4e2a9`)
- **Assigned GitHub Issue:** [#3 — `[M1][Owner: U] Implement 22-layer FP32 scalar Transformer forward pass and multi-token golden validation`](https://github.com/shreyaas5/inference-engine/issues/3)
- **Owned Files & Verified Status:**
  - `src/kernels/scalar/kernels_scalar.cpp`, `include/inference_engine/kernels.hpp` — **REFERENCE VERIFIED (SYNTHETIC / NUMPY)** (`rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, and `gqa_attention_f32` implemented, unit-tested in `kernels_scalar`, and verified against independent Python NumPy `FP32` math).
  - `src/core/kv_cache.cpp` — **TESTED (SYNTHETIC)** (`init`, `clear`, `append`, `commit_positions`, `commit_token`, `key_vector`, `value_vector` implemented and passing CTest).
  - `src/transformer/transformer.cpp`, `include/inference_engine/transformer.hpp` — **REFERENCE VERIFIED (SYNTHETIC / NUMPY)** (`TransformerEngine::init` and `forward_token` implement all 16 Llama decoder steps with pre-allocated scratch buffers; verified on synthetic 2-layer models against both C++ reference and independent Python NumPy `FP32` reference with `max_diff = 6.55651e-07`; **NOT YET VALIDATED** on the real 22-layer TinyLlama checkpoint).
  - `tools/generate_golden_reference.py` — **TESTED (SYNTHETIC)** (Implements `"GOLD"` v1 binary fixture packing/unpacking and PyTorch/HF greedy decode runner; see `KI-011` for follow-up on per-step decode logit comparison).
  - `tests/test_kernels_scalar.cpp`, `tests/test_kv_cache.cpp`, `tests/test_golden_forward.cpp` — **TESTED (SYNTHETIC)** (All passing in CTest).
- **Review Follow-Ups (`KI-010`, `KI-011`, `KI-012`):**
  - Unify `TransformerEngine::init()` tensor binding with `ModelLoader::bind_transformer_weights()` before M2 (`Q4_0`).
  - Run `tools/generate_golden_reference.py` and `golden_forward` on the real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` checkpoint.

---

### Member P — Tokenizer, Runtime & Benchmarks
- **Feature Branch:** `feature/runtime-tokenizer` (currently at commit `cb85411`)
- **Assigned GitHub Issue:** [#4 — `[M1][Owner: P] Integrate SentencePiece C++ processor and verify TinyLlama tokenizer & chat template parity`](https://github.com/shreyaas5/inference-engine/issues/4)
- **Owned Files & Verified Status:**
  - `src/tokenizer/tokenizer.cpp` — **IN PROGRESS / STUB** (`format_chat_prompt` implemented and tested; `encode`/`decode` use synthetic byte-offset fallback until SentencePiece C++ is linked).
  - `src/sampler/sampler.cpp` — **IN PROGRESS** (`sample_greedy` implemented and tested; stochastic top-p `sample()` delegates to greedy until M4).
  - `src/app/main.cpp` — **IN PROGRESS / STUB** (`--help` and `--info` implemented; inference and benchmark execution loops pending).
  - `tools/verify_tokenizer_parity.py` — **IN PROGRESS** (`--self-test` verifies chat template; full SentencePiece vs HF parity check pending).
  - `.github/workflows/ci.yml` — **IMPLEMENTED** (Ubuntu x86-64 + macOS ARM64 CMake/CTest matrix).
