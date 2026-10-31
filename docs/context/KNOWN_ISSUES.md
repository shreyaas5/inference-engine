# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Assumptions & Risks

**Last Updated:** 2026-10-31T18:46:00+05:30  
*(Policy: Never delete an entry silently. When an issue is fixed, change its status to `RESOLVED` and cite the commit/test evidence.)*

---

## 1. Functional Stubs & Unimplemented M1–M4 Components

### KI-001: `TransformerEngine::forward_token()` Is an M0 Scaffolding Stub
- **Status:** `RESOLVED (FOR FP32 SCALAR BASELINE — SYNTHETICALLY & NUMPY VALIDATED; REAL CHECKPOINT PENDING)`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Evidence:** Implemented in commit `5115467` (`feature/transformer`) and merged into `integration` (`1f4e2a9`). Executes all 16 Llama decoder steps across `config_.num_layers` using pre-allocated `FP32` scratch buffers, enforces the `KvCache` commit contract (`append` per layer, `commit_token` once after all layers complete), and matches an independent Python NumPy `FP32` reference (`max_diff = 6.55651e-07` across a 5-token prompt + 4 decode steps). Full 22-layer execution on the real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` checkpoint remains pending local weight download.

### KI-002: Causal Grouped Query Attention (GQA) Kernel Not Yet Implemented
- **Status:** `RESOLVED`
- **Location:** `scalar::gqa_attention_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp) and [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp)
- **Evidence:** Implemented in commit `5115467` and unit-tested in `tests/test_kernels_scalar.cpp` and `tests/test_golden_forward.cpp`. Supports arbitrary `num_query_heads / num_kv_heads` GQA group ratios (`8:1` for TinyLlama), $1/\sqrt{d_{\text{head}}}$ scaling, causal masking over `0..pos`, numerically stable softmax, and value aggregation.

### KI-003: `Tokenizer::encode()` and `decode()` Use a Synthetic Byte-Offset Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#4`, Owner: **P**)
- **Location:** [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp), [`CMakeLists.txt`](../../CMakeLists.txt)
- **Description:** SentencePiece C++ is not linked yet. `encode()` pushes `static_cast<int32_t>(c) + 3` per byte so M0 BOS/EOS unit tests can run without external files.

### KI-004: Offline Python Scripts (`convert_to_inference_engine.py`, `generate_golden_reference.py`) Are Stubs
- **Status:** `PARTIALLY RESOLVED` (`convert_to_inference_engine.py` FP32, `inspect_checkpoint.py`, and `generate_golden_reference.py` binary fixture generator are **IMPLEMENTED & SELF-TESTED**; `Q4_0` conversion in `convert_to_inference_engine.py` remains `OPEN` for M2, and running `generate_golden_reference.py` on the real 2.20 GB checkpoint remains `OPEN` for M1)
- **Evidence:** `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` (commit `aa40368`) and `tools/generate_golden_reference.py` (commit `5115467`) pass `--self-test` and cross-language C++ reload tests.

### KI-005: `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` Delegate to Scalar Reference
- **Status:** `OPEN` (Scheduled for M3 — Issue `#2`, Owner: **I**)
- **Location:** [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp)
- **Description:** While `dot_f32` uses real NEON (`vfmaq_f32`) and AVX2 (`_mm256_fmadd_ps`) intrinsics, `dot_q4_0_f32` currently calls `scalar::dot_q4_0_f32`.

---

## 2. Numerical, Architectural & Code Review Follow-Ups

### KI-006: RoPE Dimension Pairing Convention (`rotate_half` vs Interleaved)
- **Status:** `OPEN — VERIFIED AGAINST HUGGING FACE FORMULA IN NUMPY, PENDING REAL CHECKPOINT GOLDEN RUN`
- **Location:** `scalar::rope_inplace_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp)
- **Description:** `rope_inplace_f32` implements the Hugging Face Llama `rotate_half` convention, pairing element `i` with `i + head_dim / 2` (`i = 0..31` with `32..63`). Verified to match independent NumPy `rotate_half` (`max_diff = 6.56e-7`), and expected to match unpermuted `TinyLlama-1.1B-Chat-v1.0` `.safetensors` weights directly—must still be confirmed against real checkpoint golden fixtures.

### KI-007: `Q4_0` Rounding Parity with Upstream GGML
- **Status:** `OPEN — UNVERIFIED AGAINST EXTERNAL REFERENCE` (Verify in M1/M2 — Issue `#2`)
- **Location:** `quantize_row_q4_0` in [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp)
- **Description:** Verify bit-exact `BlockQ4_0` output against an independent reference implementation in M2.

### KI-010: Duplicate Tensor Binding Logic Between `TransformerEngine::init()` and `ModelLoader::bind_transformer_weights()`
- **Status:** `RESOLVED`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp), [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp), [`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp)
- **Evidence:** Refactored `TransformerEngine` to store `TransformerWeightBindings bindings_{}` and delegate weight validation and binding to `ModelLoader::bind_transformer_weights(&bindings_)`. Tested and verified across all 7 CTest suites.

### KI-011: `generate_golden_reference.py` Decode Logit Indexing & Unchecked `gen_ref_logits` in `test_golden_forward.cpp`
- **Status:** `PARTIALLY RESOLVED (DECODE LOGIT COMPARISON & DYNAMIC IMPORTS IMPLEMENTED)`
- **Location:** [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py), [`tests/test_golden_forward.cpp`](../../tests/test_golden_forward.cpp)
- **Evidence:** Updated `tools/generate_golden_reference.py` so that `gen_logits[step]` captures the forward pass logits after appending `next_token` (evaluating at `pos = prompt_len + step`), switched to dynamic `importlib` imports to prevent IDE static analysis missing-module warnings, and added per-step decode logit comparisons in `tests/test_golden_forward.cpp` (`test_offline_golden_fixture_if_present`). Intermediate layer activations (layers `0`, `11`, `21`) remain planned for deep introspection.

### KI-012: `TransformerEngine::forward_token()` Calls `scalar::matmul_f32` Directly and Ignores `backend_` / `num_threads`
- **Status:** `OPEN — EXPECTED FOR M1 SCALAR BASELINE, WIRE IN M3`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Description:** `TransformerEngine::init()` accepts `SimdBackend backend` and `uint32_t num_threads`, but `forward_token()` invokes `scalar::matmul_f32()` directly instead of `dispatch_matmul_f32(..., backend_, pool_)`. Acceptable for Milestone M1 (`FP32` scalar baseline), but must be wired to `dispatch_matmul_*` and `ThreadPool` in Milestone M3.

---

## 3. Environment, Build & Workflow Issues

### KI-008: Python Dependencies & SentencePiece C++ Library Not Installed in Default Host Env
- **Status:** `PARTIALLY RESOLVED`
- **Description:**
  - `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`, and `tools/generate_golden_reference.py --self-test` run with zero third-party packages on system Python 3, and `/Users/shreyaas/cwq/miniconda3/bin/python` provides `torch 2.8.0` and `numpy 2.4.2`.
  - Python `transformers` + `sentencepiece` and C++ `sentencepiece` (`brew install sentencepiece`) are still required for real Hugging Face golden generation and Issue `#4`.

### KI-009: Unstaged EOF Newline Diffs in Working Tree
- **Status:** `RESOLVED` (Fixed prior to commit `60b103e` on `feature/model-loader`)
- **Evidence:** Restored the 52 EOF-only modified tracked files and committed legitimate documentation changes in `60b103e`.
