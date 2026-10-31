# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Assumptions & Risks

**Last Updated:** 2026-10-31T18:25:00+05:30  
*(Policy: Never delete an entry silently. When an issue is fixed, change its status to `RESOLVED` and cite the commit/test evidence.)*

---

## 1. Functional Stubs & Unimplemented M1–M4 Components

### KI-001: `TransformerEngine::forward_token()` Is an M0 Scaffolding Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#3`, Owner: **U**)
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Description:** `forward_token()` validates arguments, writes zero K/V vectors to `KvCache` across all layers, calls `commit_token()`, and fills `out_logits` with `0.0f`. `ModelLoader::bind_transformer_weights()` is now implemented on `feature/model-loader` and ready to be called by `TransformerEngine::init()`.

### KI-002: Causal Grouped Query Attention (GQA) Kernel Not Yet Implemented
- **Status:** `OPEN` (Scheduled for M1 — Issue `#3`, Owner: **U**)
- **Location:** [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp) / [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp)
- **Description:** Scalar kernels exist for RMSNorm, RoPE, Softmax, SwiGLU, and MatMul, but the multi-head causal GQA orchestration (`32` query heads mapped `8:1` onto `4` KV heads over positions `0..pos`) is not yet implemented.

### KI-003: `Tokenizer::encode()` and `decode()` Use a Synthetic Byte-Offset Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#4`, Owner: **P**)
- **Location:** [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp), [`CMakeLists.txt`](../../CMakeLists.txt)
- **Description:** SentencePiece C++ is not linked yet. `encode()` pushes `static_cast<int32_t>(c) + 3` per byte so M0 BOS/EOS unit tests can run without external files.

### KI-004: Offline Python Scripts (`convert_to_inference_engine.py`, `generate_golden_reference.py`) Are Stubs
- **Status:** `PARTIALLY RESOLVED` (`convert_to_inference_engine.py` FP32 & `inspect_checkpoint.py` **RESOLVED** on `feature/model-loader`; `generate_golden_reference.py` and `Q4_0` conversion remain `OPEN` for Issue `#3` / M2)
- **Evidence:** `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` now implement full `.safetensors` header & payload reading, `3 + 9 * num_layers` (`201`) tensor inventory validation, and two-pass streaming `BF16/F16/F32 -> FP32` `.ie` v1 conversion, verified by `--self-test`, `--inspect-remote-hf`, and C++ `ModelLoader` reload (`tests/test_loader_validation.cpp`).

### KI-005: `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` Delegate to Scalar Reference
- **Status:** `OPEN` (Scheduled for M3 — Issue `#2`, Owner: **I**)
- **Location:** [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp)
- **Description:** While `dot_f32` uses real NEON (`vfmaq_f32`) and AVX2 (`_mm256_fmadd_ps`) intrinsics, `dot_q4_0_f32` currently calls `scalar::dot_q4_0_f32`.

---

## 2. Numerical & Architectural Assumptions to Verify in M1

### KI-006: RoPE Dimension Pairing Convention (`rotate_half` vs Interleaved)
- **Status:** `OPEN — UNVERIFIED AGAINST CHECKPOINT` (Verify in M1 — Issue `#3`)
- **Location:** `scalar::rope_inplace_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp)
- **Description:** `rope_inplace_f32` implements the Hugging Face Llama `rotate_half` convention, pairing element `i` with `i + head_dim / 2` (`i = 0..31` with `32..63`). Since `convert_to_inference_engine.py` preserves unpermuted Hugging Face `.safetensors` row layout directly, `rotate_half` is expected to match directly—verify against Layer 0 Q/K golden reference tensors in M1.

### KI-007: `Q4_0` Rounding Parity with Upstream GGML
- **Status:** `OPEN — UNVERIFIED AGAINST EXTERNAL REFERENCE` (Verify in M1/M2 — Issue `#2`)
- **Location:** `quantize_row_q4_0` in [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp)
- **Description:** Verify bit-exact `BlockQ4_0` output against an independent reference implementation in M2.

---

## 3. Environment, Build & Workflow Issues

### KI-008: Python Dependencies & SentencePiece C++ Library Not Installed in Default Host Env
- **Status:** `PARTIALLY RESOLVED`
- **Description:**
  - `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` were engineered to work with **zero external Python packages** (pure-Python `struct`/`array` `BF16 -> FP32` path) while automatically using `numpy` (`/Users/shreyaas/cwq/miniconda3/bin/python`) for vectorized C-speed conversion when available.
  - `transformers` and `sentencepiece` (Python) and Homebrew `sentencepiece` (C++) are still required before running `generate_golden_reference.py` (Issue `#3`) and linking `Tokenizer` (Issue `#4`).

### KI-009: Unstaged EOF Newline Diffs in Working Tree
- **Status:** `RESOLVED` (Fixed prior to commit `60b103e` on `feature/model-loader`)
- **Evidence:** Identified all 52 EOF-only modified tracked files via `git diff --numstat`, restored them cleanly without touching the 2 modified documentation files or 14 untracked context files, and committed `60b103e`.
