# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-10-31T19:25:00+05:30  
**Current Milestone:** `M1 — FP32 scalar inference` & `M2/M3 — Quantisation and SIMD` (In Progress — S's Loader `#1`, U's Transformer `#3`, and I's Quantisation & SIMD `#2` merged into `integration`)  
**Maintainer:** S (Project Lead)

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `integration`
- **Branch State Summary:**
  - `main`: `177fe49` (`Merge branch 'integration' into main`, matches `origin/main`)
  - `feature/model-loader`: `aa40368` (matches `origin/feature/model-loader`)
  - `feature/transformer`: `5115467` (matches `origin/feature/transformer`)
  - `feature/quantisation-simd`: `f76ab55` (matches `origin/feature/quantisation-simd`, merged into `integration`)
  - `integration`: `967f072` (`Merge branch 'feature/quantisation-simd' into integration`) plus documentation updates.
  - `feature/runtime-tokenizer`: `cb85411`
- **Working Tree State:** Clean (`build/` ignored).
- **CMake Build Status:** **PASSING** (`AppleClang 17.0.0.17000603`, `CMake 4.4.3`, zero compiler warnings with `-Wall -Wextra -Wpedantic`).
- **CTest Suite Status:** **7 / 7 PASSING** on `integration` (verified on Apple M4 ARM64):
  1. `loader_validation` — **PASSED** (12 malformed-file rejection checks + 21-tensor `mmap` reload, `bind_transformer_weights()`, and bit-exact `FP32` data preservation).
  2. `quant_packing` — **PASSED** (18-byte `BlockQ4_0`, nibble packing, zero-block, out-of-bounds, subnormal/inf FP16, and bit-exact upstream GGML reference parity).
  3. `kernels_scalar` — **PASSED** (Includes unit tests for `rmsnorm_f32`, `rope_inplace_f32`, `softmax_f32`, `swiglu_f32`, and multi-head `gqa_attention_f32`).
  4. `kernels_simd` — **PASSED** (NEON and AVX2 scalar-fallback vector math, row-parallel multithreading scaling across 1–16 threads with task chunking).
  5. `kv_cache` — **PASSED** (Separation of `append()` and `commit_token()`).
  6. `tokenizer` — **PASSED** (Zephyr chat template formatting).
  7. `golden_forward` — **PASSED** (Synthetic 2-layer `FP32` forward parity, prefill + autoregressive decode position contract, KV cache / GQA non-zero checks, and independent NumPy golden verification).
- **Python Offline Tool Self-Tests:**
  - `python3 tools/inspect_checkpoint.py --inspect-remote-hf` — **PASSED** (`201/201` `BF16` tensors in live `TinyLlama/TinyLlama-1.1B-Chat-v1.0` header verified).
  - `python3 tools/convert_to_inference_engine.py --self-test` — **PASSED**
  - `python3 tools/generate_golden_reference.py --self-test` — **PASSED**
  - `python3 tools/verify_tokenizer_parity.py --self-test` — **PASSED**

---

## 2. Honest Implementation Breakdown

### Implemented, Unit-Tested & Validated
- **`.ie` v1 Binary Loader, Streaming `.safetensors` Converter & Tensor Binder (`src/loader/loader.cpp`, `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py` — Owner: S):**
  - **Implemented & Unit-Tested:** `.ie` v1 parser/writer, 64-byte alignment & non-overlap validation, POSIX `mmap` loader, `ModelLoader::bind_transformer_weights()`.
  - **Reference-Validated (Checkpoint Header Metadata):** Verified live 23,088-byte `model.safetensors` header and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`201` `BF16` tensors).
  - **Validated with Synthetic Weights:** Two-pass streaming `BF16 -> FP32` `.safetensors -> .ie` conversion and C++ `mmap` reload verified on 2-layer (`21`-tensor) synthetic checkpoints.
- **Scalar FP32 Kernels, Causal GQA & Transformer Forward Pass (`src/kernels/scalar/kernels_scalar.cpp`, `src/core/kv_cache.cpp`, `src/transformer/transformer.cpp` — Owner: U):**
  - **Implemented & Unit-Tested:** `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, `gqa_attention_f32`, `KvCache`, and full 16-step `TransformerEngine::init` + `forward_token`.
  - **Validated with Synthetic Weights & Independent NumPy Reference:** Verified in `golden_forward` with `max_diff = 6.55651e-07`.
- **`Q4_0` Quantisation, SIMD Kernels & ThreadPool (`src/kernels/quant_q4_0.cpp`, `src/kernels/neon/`, `src/kernels/avx2/`, `src/kernels/threading/`, `src/kernels/dispatch.cpp` — Owner: I):**
  - **Implemented & Unit-Tested:**
    - `BlockQ4_0` packing/dequantisation, verified bit-exact against upstream GGML reference logic (`KI-007`).
    - ARM64 NEON: `neon::dot_f32`, `neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32` verified on Apple Silicon.
    - x86-64 AVX2/FMA: `avx2::dot_f32`, `avx2::dot_q4_0_f32`, `avx2::matmul_q4_0_f32`, with CPUID detection in `src/common/common.cpp` (requires native x86 CI/hardware verification).
    - `ThreadPool`: persistent worker threads with task-chunking optimization eliminating lock contention.
    - Dispatched matmul: `dispatch_matmul_q4_0_f32` uses whole-matrix vector kernels when single-threaded and task-chunked multithreading when pooled.

### In-Progress / Stubbed (`IN PROGRESS` / `NOT STARTED` — Remaining M1–M4 Work)
- **TransformerEngine Wiring to Q4_0 Dispatch (Milestone M2/M3):**
  - `TransformerEngine::forward_token()` currently invokes `scalar::matmul_f32` directly and requires `FP32` weights. Wiring `dispatch_matmul_q4_0_f32` and `ThreadPool` into `TransformerEngine` is scheduled for Milestone M2/M3.
- **Full TinyLlama-1.1B Checkpoint Quantisation & Conversion (Milestone M2):**
  - `tools/convert_to_inference_engine.py` supports `--dtype fp32`. Adding `--dtype q4_0` export is scheduled for Milestone M2.
  - Actual TinyLlama quantised inference has not yet been executed against real model weights.
- **Native x86 AVX2/FMA Verification:**
  - AVX2/FMA vector code compiles cleanly with scalar fallbacks, but needs native test execution on an x86 CPU.
- **SentencePiece C++ Tokenizer Integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt` — Owner: P, Issue `#4`):**
  - `Tokenizer::encode()` and `decode()` still use the M0 synthetic byte-offset fallback (`byte + 3`).
- **CLI Inference Loop (`src/app/main.cpp` — Owner: P):**
  - Handles `--help` and `--info` only.

---

## 3. Active Blockers & Review Follow-Ups

1. **Real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` Weight Payload Not Yet Downloaded Locally:**
   - Required before claiming real TinyLlama inference.
2. **Missing SentencePiece C++ Library (`KI-003` / `KI-008`):**
   - Needed for Issue `#4` (Member P).
3. **Follow-Up Technical Tasks:**
   - `KI-010`: Unify `TransformerEngine::init` with `ModelLoader::bind_transformer_weights()`.
   - `KI-012`: Wire `dispatch_matmul_*` and `ThreadPool` into `TransformerEngine::forward_token()` for M2/M3.
   - Run CTest on x86-64 machine to natively exercise AVX2/FMA vector intrinsics.

---

## 4. Next Three Engineering Priorities

1. **[Owner: P — Issue #4] Integrate SentencePiece C++ Tokenizer (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`) & Wire CLI Generation (`src/app/main.cpp`).**
2. **[Owners: S & U — Issues #1 & #3] Download Real `TinyLlama-1.1B-Chat-v1.0` Checkpoint, Convert to `.ie` FP32, and Run `tools/generate_golden_reference.py` + `golden_forward` CTest.**
3. **[Owners: I & U — Milestone M2] Enable `Q4_0` Dispatch in `TransformerEngine::forward_token()` and Add `--dtype q4_0` Export to `tools/convert_to_inference_engine.py`.**
