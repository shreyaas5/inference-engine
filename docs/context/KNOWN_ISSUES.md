# `KNOWN_ISSUES.md` — Active Bugs, Stubs, Assumptions & Risks

**Last Updated:** 2026-11-01T14:04:00+05:30
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

### KI-003: SentencePiece Tokenizer Has Not Passed Model-Backed Build/Parity Validation
- **Status:** `OPEN — IMPLEMENTED IN WORKTREE, VALIDATION PENDING` (Issue `#4`, Owner: **P**)
- **Location:** [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp), [`CMakeLists.txt`](../../CMakeLists.txt)
- **Description:** SentencePiece C++ integration now replaces the `byte + 3` fallback, validates the TinyLlama vocabulary and special-token IDs, and handles `<unk>`, `<s>`, and `</s>` as exact special tokens. The model-backed CTest and Hugging Face parity harness have not yet been run against the C++ binary.

### KI-004: Offline Python Scripts (`convert_to_inference_engine.py`, `generate_golden_reference.py`) Are Stubs
- **Status:** `PARTIALLY RESOLVED` (`convert_to_inference_engine.py` FP32, `inspect_checkpoint.py`, and `generate_golden_reference.py` binary fixture generator are **IMPLEMENTED & SELF-TESTED**; `Q4_0` conversion in `convert_to_inference_engine.py` remains `OPEN` for M2, and running `generate_golden_reference.py` on the real 2.20 GB checkpoint remains `OPEN` for M1)
- **Evidence:** `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` (commit `aa40368`) and `tools/generate_golden_reference.py` (commit `5115467`) pass `--self-test` and cross-language C++ reload tests.

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
- **Status:** `OPEN — REVIEW WARNING (NON-BLOCKING FOR M1 FP32, FIX BEFORE M2)`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp), [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp), [`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp)
- **Description:** Refactor `TransformerEngine` to use `TransformerWeightBindings` (`TensorView`) before Milestone M2 so Q4_0 quantised tensors can be bound alongside FP32 norms.

### KI-011: `generate_golden_reference.py` Decode Logit Indexing & Unchecked `gen_ref_logits` in `test_golden_forward.cpp`
- **Status:** `OPEN — REVIEW WARNING`
- **Location:** [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py), [`tests/test_golden_forward.cpp`](../../tests/test_golden_forward.cpp)

### KI-012: `TransformerEngine::forward_token()` Not Yet Wired to `dispatch_matmul_q4_0_f32`
- **Status:** `OPEN — SCHEDULED FOR M2/M3`
- **Location:** [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Description:** `dispatch_matmul_q4_0_f32`, `dispatch_matmul_f32`, and multithreaded `ThreadPool` chunking are fully implemented and tested in `src/kernels/dispatch.cpp` and `src/kernels/threading/thread_pool.cpp`. Wiring them into `TransformerEngine::forward_token()` for mixed-precision decoding is scheduled for Milestone M2/M3.

### KI-013: Native x86-64 AVX2/FMA Verification Pending
- **Status:** `OPEN — HARDWARE / CI VERIFICATION`
- **Location:** [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp), [`src/common/common.cpp`](../../src/common/common.cpp)
- **Description:** The AVX2/FMA vector kernels compile cleanly and fall back gracefully on non-x86 platforms, but need execution on an x86-64 CPU (Member I's machine or GitHub Actions runner) to natively verify performance and numerical output against the hardware FMA pipeline.

---

## 3. Environment, Build & Workflow Issues

### KI-008: Python Dependencies & SentencePiece C++ Library Not Installed in Default Host Env
- **Status:** `PARTIALLY RESOLVED`
- **Description:** CMake discovers SentencePiece >=0.2.0 through pkg-config or fetches a pinned upstream revision. CI installs the Python Hugging Face parity dependencies. This Windows host has Python and MinGW but no CMake/CTest, so full validation is still blocked here.

### KI-009: Unstaged EOF Newline Diffs in Working Tree
- **Status:** `RESOLVED`

### KI-014: Full Feature-Branch Build and Parity Run Not Yet Available Locally
- **Status:** `OPEN — ENVIRONMENT VALIDATION BLOCKER`
- **Location:** `CMakeLists.txt`, `.github/workflows/ci.yml`
- **Description:** The installed `C:\Program Files\CMake\bin\cmake.exe` cannot find its `share/cmake-4.4` modules; CMake Tools returns `-1` without diagnostics. MinGW GCC 6.3 lacks `<string_view>`. The complete build/CTest, model-backed parity, and CLI runtime tests remain pending a configured environment. Python `transformers`/`sentencepiece`, tokenizer assets, executable, and real model are also absent.

### KI-015: `golden_forward` CTest Was Not Dispatched by the Test Runner
- **Status:** `RESOLVED IN SOURCE — NATIVE CTEST VALIDATION PENDING`
- **Location:** `CMakeLists.txt`, `tests/test_main.cpp`
- **Evidence:** CMake already registered nine test names, including `golden_forward`, while the C++ runner omitted that name. Added `{"golden_forward", run_golden_forward_tests}` to the runner. CMake could not configure and CTest did not execute, so runtime verification remains pending.
