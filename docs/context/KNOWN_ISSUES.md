# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Assumptions & Risks

**Last Updated:** 2026-10-31T19:25:00+05:30  
*(Policy: Never delete an entry silently. When an issue is fixed, change its status to `RESOLVED` and cite the commit/test evidence.)*

---

## 1. Functional Stubs & Unimplemented M1–M4 Components

### KI-001: `TransformerEngine::forward_token()` Is an M0 Scaffolding Stub
- **Status:** `RESOLVED (FOR FP32 SCALAR BASELINE — SYNTHETICALLY & NUMPY VALIDATED; REAL CHECKPOINT PENDING)`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Evidence:** Implemented in commit `5115467` (`feature/transformer`) and merged into `integration` and `main`. Executes all 16 Llama decoder steps across `config_.num_layers` using pre-allocated `FP32` scratch buffers, enforces the `KvCache` commit contract, and matches an independent Python NumPy `FP32` reference (`max_diff = 6.55651e-07`). Full 22-layer execution on the real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` checkpoint remains pending local weight download.

### KI-002: Causal Grouped Query Attention (GQA) Kernel Not Yet Implemented
- **Status:** `RESOLVED`
- **Location:** `scalar::gqa_attention_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp) and [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp)
- **Evidence:** Implemented in commit `5115467` and unit-tested in `tests/test_kernels_scalar.cpp` and `tests/test_golden_forward.cpp`.

### KI-003: `Tokenizer::encode()` and `decode()` Use a Synthetic Byte-Offset Stub
- **Status:** `OPEN` (Scheduled for M1 — Issue `#4`, Owner: **P**)
- **Location:** [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp), [`CMakeLists.txt`](../../CMakeLists.txt)
- **Description:** SentencePiece C++ is not linked yet. `encode()` pushes `static_cast<int32_t>(c) + 3` per byte so M0 BOS/EOS unit tests can run without external files.

### KI-004: Offline Python Scripts (`convert_to_inference_engine.py`, `generate_golden_reference.py`) Are Stubs
- **Status:** `RESOLVED (FOR FP32 & Q4_0 CONVERTER, INSPECTOR, AND GOLDEN BINARY PACKER; REAL 2.20 GB CHECKPOINT GOLDEN RUN PENDING)`
- **Evidence:** `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py` (`--dtype fp32` in `aa40368` and `--dtype q4_0` / `--keep-embeddings-fp32` / `--report-error` in `cd98ae8` on `feature/model-format-q4`), and `tools/generate_golden_reference.py` (`5115467` / `6dd4fed`) pass `--self-test`, cross-language C++ `ModelLoader` reload tests, and live `TinyLlama-1.1B-Chat-v1.0` HTTP Range tensor verification.

### KI-005: `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` Delegate to Scalar Reference
- **Status:** `RESOLVED` (Merged into `integration` in commit `967f072`)
- **Location:** [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp)
- **Evidence:** Vectorized Q4_0 dot products and matrix multiplication implemented for both ARM64 NEON (`compute_block_unscaled_dot_neon`, `neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32`) and x86-64 AVX2/FMA (`compute_block_unscaled_dot`, `avx2::dot_q4_0_f32`, `avx2::matmul_q4_0_f32`). NEON verified on Apple Silicon ARM64; AVX2 verified with compiler builds and runtime CPUID guards (native x86 run pending).

---

## 2. Numerical, Architectural & Code Review Follow-Ups

### KI-006: RoPE Dimension Pairing Convention (`rotate_half` vs Interleaved)
- **Status:** `OPEN — VERIFIED AGAINST HUGGING FACE FORMULA IN NUMPY, PENDING REAL CHECKPOINT GOLDEN RUN`
- **Location:** `scalar::rope_inplace_f32` in [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp)
- **Description:** `rope_inplace_f32` implements the Hugging Face Llama `rotate_half` convention, pairing element `i` with `i + head_dim / 2` (`i = 0..31` with `32..63`). Verified to match independent NumPy `rotate_half` (`max_diff = 6.56e-7`), and expected to match unpermuted `TinyLlama-1.1B-Chat-v1.0` `.safetensors` weights directly—must still be confirmed against real checkpoint golden fixtures.

### KI-007: `Q4_0` Rounding Parity with Upstream GGML
- **Status:** `RESOLVED` (Merged into `integration` in commit `967f072`)
- **Location:** `quantize_row_q4_0` in [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp), [`tests/test_quant_packing.cpp`](../../tests/test_quant_packing.cpp)
- **Evidence:** Implemented explicit 1,024-element test comparing scale `d` and packed nibble bytes `qs[0..15]` block-by-block against upstream GGML quantization logic. Verified bit-exact parity in `test_quant_packing`.

### KI-010: Duplicate Tensor Binding Logic Between `TransformerEngine::init()` and `ModelLoader::bind_transformer_weights()`
- **Status:** `RESOLVED` (Fixed in commit `6dd4fed` on `feature/model-format-q4` / `bce66ae` on `feature/transformer-bindings`)
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp), [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp), [`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp)
- **Evidence:** Refactored `TransformerEngine` to store `TransformerWeightBindings bindings_{}` and delegate weight validation and binding to `ModelLoader::bind_transformer_weights(&bindings_)`, supporting both `FP32` and `Q4_0` models via `TensorView` (`data_f32()` / `data_q4_0()`).

### KI-011: `generate_golden_reference.py` Decode Logit Indexing & Unchecked `gen_ref_logits` in `test_golden_forward.cpp`
- **Status:** `PARTIALLY RESOLVED (DECODE LOGIT COMPARISON & DYNAMIC IMPORTS IMPLEMENTED IN 6dd4fed)`
- **Location:** [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py), [`tests/test_golden_forward.cpp`](../../tests/test_golden_forward.cpp)
- **Evidence:** Updated `tools/generate_golden_reference.py` so `gen_logits[step]` records forward logits after appending `next_token` at `pos = prompt_len + step`, and added per-step `gen_ref_logits` verification (`max_diff <= 1e-3f`) in `tests/test_golden_forward.cpp`.

### KI-012: `TransformerEngine::forward_token()` Not Yet Wired to `dispatch_matmul_q4_0_f32`
- **Status:** `RESOLVED` (Implemented and verified in `feature/transformer-q4`)
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp), [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp)
- **Evidence:** Implemented `TransformerEngine::matmul()` which dynamically routes to `dispatch_matmul_f32` or `dispatch_matmul_q4_0_f32` with runtime SIMD selection and optional `ThreadPool` chunking. Implemented per-token row dequantisation for Q4_0 `embed_tokens` via `dequantize_row_q4_0` and Q4_0 `lm_head` projection. Verified end-to-end against independent NumPy implementation across prefill and 4 decode steps, confirming matching numerical output between C++ and NumPy to 5 decimal places.

### KI-013: Native x86-64 AVX2/FMA Verification Pending
- **Status:** `OPEN — HARDWARE / CI VERIFICATION`
- **Location:** [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp), [`src/common/common.cpp`](../../src/common/common.cpp)
- **Description:** The AVX2/FMA vector kernels compile cleanly and fall back gracefully on non-x86 platforms, but need execution on an x86-64 CPU (Member I's machine or GitHub Actions runner) to natively verify performance and numerical output against the hardware FMA pipeline.

### KI-014: Full 2.20 GB TinyLlama Checkpoint Local Download & 22-Layer Golden Run
- **Status:** `OPEN — REPRODUCIBLE COMMANDS & REAL-TENSOR SLICES VERIFIED`
- **Location:** `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`, `models/TinyLlama-1.1B-Chat-v1.0/`
- **Description:** Real checkpoint tensor inventory (`201` `BF16` tensors, `2,200,096,768` payload bytes) and real-tensor `Q4_0` quantisation errors across 9 projection roles are verified via `tools/inspect_checkpoint.py --eval-real-quant-error`. `config.json` and `tokenizer.model` are downloaded locally. Full 2.20 GB `model.safetensors` download requires ~25 minutes on commodity connection (~1.5 MB/s); resume command `curl -C - -L -o models/TinyLlama-1.1B-Chat-v1.0/model.safetensors "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"` is documented for full offline 22-layer execution once Member U wires `dispatch_matmul_q4_0_f32`.

---

## 3. Environment, Build & Workflow Issues

### KI-008: Python Dependencies & SentencePiece C++ Library Not Installed in Default Host Env
- **Status:** `PARTIALLY RESOLVED`
- **Description:** Python `transformers` + `sentencepiece` and C++ `sentencepiece` (`brew install sentencepiece`) are still required for real Hugging Face golden generation and Issue `#4`.

### KI-009: Unstaged EOF Newline Diffs in Working Tree
- **Status:** `RESOLVED`
