# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Assumptions & Risks

**Last Updated:** 2026-10-31  
*(Policy: Never delete an entry silently. When an issue is fixed, change its status to `RESOLVED` and cite the commit/test evidence.)*

---

## 1. Functional Stubs & Unimplemented M1–M4 Components

### KI-001: `TransformerEngine::forward_token()` Is an M0 Scaffolding Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#3`, Owner: **U**)
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Description:** `forward_token()` validates arguments, writes zero K/V vectors to `KvCache` across all layers, calls `commit_token()`, and fills `out_logits` with `0.0f`. It does not bind `.ie` tensor weights or execute transformer layer operations yet.

### KI-002: Causal Grouped Query Attention (GQA) Kernel Not Yet Implemented
- **Status:** `OPEN` (Scheduled for M1 — Issue `#3`, Owner: **U**)
- **Location:** [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp) / [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp)
- **Description:** Scalar kernels exist for RMSNorm, RoPE, Softmax, SwiGLU, and MatMul, but the multi-head causal GQA orchestration (`32` query heads mapped `8:1` onto `4` KV heads over positions `0..pos`) is not yet implemented.

### KI-003: `Tokenizer::encode()` and `decode()` Use a Synthetic Byte-Offset Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#4`, Owner: **P**)
- **Location:** [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp), [`CMakeLists.txt`](../../CMakeLists.txt)
- **Description:** SentencePiece C++ is not linked yet. `encode()` pushes `static_cast<int32_t>(c) + 3` per byte so M0 BOS/EOS unit tests can run without external files. This will produce completely invalid token IDs for real TinyLlama inference until SentencePiece is linked.

### KI-004: Offline Python Scripts (`convert_to_inference_engine.py`, `generate_golden_reference.py`) Are Stubs
- **Status:** `OPEN` (Scheduled for M1 — Issues `#1` and `#3`, Owners: **S** & **U**)
- **Location:** [`tools/convert_to_inference_engine.py`](../../tools/convert_to_inference_engine.py), [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py)
- **Description:** `convert_to_inference_engine.py` implements 64-byte header packing (`--self-test`), but does not yet parse `.safetensors` or write `.ie` files. `generate_golden_reference.py` is an `argparse` skeleton.

### KI-005: `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` Delegate to Scalar Reference
- **Status:** `OPEN` (Scheduled for M3 — Issue `#2`, Owner: **I**)
- **Location:** [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp)
- **Description:** While `dot_f32` uses real NEON (`vfmaq_f32`) and AVX2 (`_mm256_fmadd_ps`) intrinsics, `dot_q4_0_f32` currently calls `scalar::dot_q4_0_f32`.

---

## 2. Numerical & Architectural Assumptions to Verify in M1

### KI-006: RoPE Dimension Pairing Convention (`rotate_half` vs Interleaved)
- **Status:** `OPEN — UNVERIFIED AGAINST CHECKPOINT` (Verify in M1)
- **Location:** `scalar::rope_inplace_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp)
- **Description:** `rope_inplace_f32` implements the Hugging Face Llama `rotate_half` convention, pairing element `i` with `i + head_dim / 2` (`i = 0..31` with `32..63`). Upstream `TinyLlama/TinyLlama-1.1B-Chat-v1.0` uses Hugging Face `LlamaForCausalLM` (`rotate_half`), whereas `llama.cpp` GGUF conversion permutes Q and K projection weights to interleave `(2i, 2i+1)`. Because we convert directly from Hugging Face `.safetensors` without GGUF weight permutation, `rotate_half` is expected to match directly—**this must be verified against Layer 0 Q/K golden reference tensors in M1**.

### KI-007: `Q4_0` Rounding Parity with Upstream GGML
- **Status:** `OPEN — UNVERIFIED AGAINST EXTERNAL REFERENCE` (Verify in M1/M2 — Issue `#2`)
- **Location:** `quantize_row_q4_0` in [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp)
- **Description:** `quantize_row_q4_0` computes `std::clamp(static_cast<int>(std::floor(x0 + 8.5f)), 0, 15)` (matching GGML's `(int8_t)(x0 + 8.5f)` for non-negative `x0 + 8.5f`). Edge-case negative values where `x0 + 8.5f < 0.0f` clamp to `0` under both `floor` and truncation, but bit-exact block parity must be verified against an independent reference in M2.

---

## 3. Environment, Build & Workflow Issues

### KI-008: Python Dependencies & SentencePiece C++ Library Not Installed in Default Host Env
- **Status:** `OPEN`
- **Description:**
  - Host `/opt/homebrew/bin/python3` (`Python 3.14.6`) lacks `torch`, `transformers`, `safetensors`, `sentencepiece`, and `numpy`.
  - Host Conda environment `/Users/shreyaas/cwq/miniconda3` has `torch 2.8.0` and `numpy 2.2.6`, but still needs `transformers`, `safetensors`, and `sentencepiece` installed before running offline M1 scripts.
  - Homebrew `sentencepiece` C++ formula is not yet installed on S's machine.

### KI-009: Unstaged EOF Newline Diffs in Working Tree
- **Status:** `OPEN`
- **Description:** After commit `cb85411`, the IDE added a trailing blank line (`1 +`) to 53 tracked files in the working tree. Before starting M1 code work on a feature branch, run `git checkout -- .` (after committing the new context-transfer documentation files) so the working tree is completely clean.

