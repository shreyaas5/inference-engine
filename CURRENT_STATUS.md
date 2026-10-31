# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-10-31T18:46:00+05:30  
**Current Milestone:** `M1 — FP32 scalar inference` (In Progress — S's Loader (`#1`) and U's Scalar Transformer (`#3`) merged into `integration`)  
**Maintainer:** S (Project Lead)

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `integration`
- **Branch State Summary:**
  - `main`: `cb85411` (`tag: m0-foundation`, matches `origin/main`)
  - `feature/model-loader`: `aa40368` (ahead of `origin/feature/model-loader` by 2 commits: `60b103e` + `aa40368`)
  - `feature/transformer`: `5115467` (matches `origin/feature/transformer` and `origin/feature/transformer-m1-fp32`, authored by Utkarsh Gupta)
  - `integration`: Merged `feature/model-loader` (`aa40368`) and `feature/transformer` (`5115467` via `--no-ff` merge commit `1f4e2a9`), plus this context-update commit. Not pushed to `origin` pending approval.
  - `feature/quantisation-simd`: `cb85411`
  - `feature/runtime-tokenizer`: `cb85411`
- **Working Tree State:** Clean (`git status` reports nothing to commit; temporary `fixtures/` and `build/` cleaned or ignored).
- **CMake Build Status:** **PASSING** (`AppleClang 17.0.0.17000603`, `CMake 4.4.3`, zero compiler warnings with `-Wall -Wextra -Wpedantic`).
- **CTest Suite Status:** **7 / 7 PASSING** on `integration` (verified on Apple M4 ARM64):
  1. `loader_validation` — **PASSED** (12 malformed-file rejection checks + 21-tensor `mmap` reload, `bind_transformer_weights()`, and bit-exact `FP32` data preservation).
  2. `quant_packing` — **PASSED**
  3. `kernels_scalar` — **PASSED** (Includes unit tests for `rmsnorm_f32`, `rope_inplace_f32`, `softmax_f32`, `swiglu_f32`, and multi-head `gqa_attention_f32`).
  4. `kernels_simd` — **PASSED**
  5. `kv_cache` — **PASSED**
  6. `tokenizer` — **PASSED**
  7. `golden_forward` — **PASSED** (Synthetic 2-layer `FP32` forward parity, prefill + autoregressive decode position contract, KV cache / GQA non-zero checks, error handling contracts, and independent NumPy golden fixture verification with `max_diff = 6.55651e-07`).
- **Python Offline Tool Self-Tests:**
  - `python3 tools/inspect_checkpoint.py --inspect-remote-hf` — **PASSED** (`201/201` `BF16` tensors in live `TinyLlama/TinyLlama-1.1B-Chat-v1.0` header verified).
  - `python3 tools/convert_to_inference_engine.py --self-test` — **PASSED**
  - `python3 tools/generate_golden_reference.py --self-test` — **PASSED**
  - `python3 tools/verify_tokenizer_parity.py --self-test` — **PASSED**

---

## 2. Honest Implementation Breakdown (Distinguishing Synthetic vs Real Checkpoint Validation)

### Implemented, Unit-Tested & Validated with Synthetic Weights / Metadata
- **`.ie` v1 Binary Loader, Streaming `.safetensors` Converter & Tensor Binder (`src/loader/loader.cpp`, `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py` — Owner: S):**
  - **Implemented & Unit-Tested:** `.ie` v1 parser/writer, 64-byte alignment & non-overlap validation, POSIX `mmap` loader, `ModelLoader::bind_transformer_weights()`.
  - **Reference-Validated (Checkpoint Header Metadata):** Verified live 23,088-byte `model.safetensors` header and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`201` `BF16` tensors, `2,200,096,768` bytes).
  - **Validated with Synthetic Weights:** Two-pass streaming `BF16 -> FP32` `.safetensors -> .ie` conversion and C++ `mmap` reload verified on 2-layer (`21`-tensor) synthetic checkpoints.
  - **Not Yet Validated with Actual TinyLlama Checkpoint Payload:** Full 2.20 GB `model.safetensors` weight file has not been downloaded/converted on disk yet.
- **Scalar FP32 Kernels, Causal GQA & Transformer Forward Pass (`src/kernels/scalar/kernels_scalar.cpp`, `src/core/kv_cache.cpp`, `src/transformer/transformer.cpp` — Owner: U):**
  - **Implemented & Unit-Tested:** `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, `gqa_attention_f32`, `KvCache` (`append` decoupled from `commit_token`), and full multi-layer `TransformerEngine::init` + `TransformerEngine::forward_token` (16-step Llama decoder forward pass with pre-allocated scratch buffers).
  - **Validated with Synthetic Weights & Independent NumPy Reference:** Verified in `golden_forward` on 2-layer synthetic `.ie` models against both an inline C++ reference and an independent Python NumPy `FP32` reference across a 5-token prompt + 4 greedy decode steps (`max_diff = 6.55651e-07`).
  - **Not Yet Validated with Actual TinyLlama Checkpoint:** Full 22-layer `TinyLlama-1.1B-Chat-v1.0` `FP32` forward pass and Hugging Face PyTorch golden comparison remain pending until the 2.20 GB checkpoint is downloaded locally.
- **`Q4_0` Quantisation, `FP32` SIMD Dot Products & `ThreadPool` (`src/kernels/quant_q4_0.cpp`, `src/kernels/neon/`, `src/kernels/avx2/`, `src/kernels/threading/` — Owner: I):**
  - **Implemented & Unit-Tested (Synthetic):** `BlockQ4_0` packing/dequantisation, `neon::dot_f32`, `avx2::dot_f32`, and `ThreadPool::parallel_for`.

### In-Progress / Stubbed (`IN PROGRESS` / `NOT STARTED` — Remaining M1–M4 Work)
- **Full TinyLlama-1.1B Checkpoint Conversion & PyTorch Golden Run (Owners: S & U, Issue `#1` / `#3`):**
  - `tools/generate_golden_reference.py` is implemented, but `fixtures/golden/` has not yet been generated from the real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` checkpoint.
- **SentencePiece C++ Tokenizer Integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt` — Owner: P, Issue `#4`):**
  - **STUB:** `Tokenizer::encode()` and `decode()` still use the M0 synthetic byte-offset fallback (`byte + 3`).
- **CLI Inference & Benchmark Loops (`src/app/main.cpp` — Owner: P, Milestone M1/M4):**
  - **STUB:** Handles `--help` and `--info` only; end-to-end generation loop not yet wired.
- **`Q4_0` Export & Mixed-Precision Forward Dispatch (Owners: S, I, U — Milestone M2):**
  - `TransformerEngine` currently restricts `weight_dtype` to `DataType::FP32` and uses private `LayerWeights` (`const float*`) rather than `ModelLoader::bind_transformer_weights()`.
- **Vectorized `Q4_0` SIMD Kernels & `ThreadPool` Dispatch in `TransformerEngine` (Owner: I/U — Milestone M3):**
  - `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` still delegate to scalar, and `TransformerEngine::forward_token()` calls `scalar::matmul_f32` directly rather than `dispatch_matmul_f32`.

---

## 3. Active Blockers & Review Follow-Ups

1. **Real 2.20 GB `TinyLlama-1.1B-Chat-v1.0` Weight Payload Not Yet Downloaded Locally:**
   - Required before claiming real TinyLlama inference or real Hugging Face golden validation.
2. **Follow-Up Warnings from U's Transformer Review (`5115467`):**
   - `KI-010`: `TransformerEngine::init` uses its own file-local `bind_tensor()` and `const float*` `LayerWeights` struct instead of `ModelLoader::bind_transformer_weights()`; unify before M2 (`Q4_0`) and validate that `config` matches `loader->config()`.
   - `KI-011`: `tools/generate_golden_reference.py` records pre-step logits in `gen_logits` (`gen_logits[0]` duplicates `prompt_final_logits`), and `test_offline_golden_fixture_if_present()` does not compare `gen_ref_logits` at each decode step or export intermediate layer activations.
   - `KI-012`: `TransformerEngine::forward_token()` calls `scalar::matmul_f32` directly and ignores `backend_` and `num_threads` (acceptable for M1 scalar baseline; wire to `dispatch_matmul_f32` + `ThreadPool` in M3).
3. **Missing SentencePiece C++ Library (`KI-003` / `KI-008`):**
   - Needed for Issue `#4` (Member P).

---

## 4. Next Three Engineering Priorities (Milestone M1)

1. **[Owner: P — Issue #4] Integrate SentencePiece C++ Tokenizer (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`) & Wire CLI Generation (`src/app/main.cpp`).**
2. **[Owners: S & U — Issues #1 & #3] Download Real `TinyLlama-1.1B-Chat-v1.0` Checkpoint, Convert to `.ie` FP32, and Run `tools/generate_golden_reference.py` + `golden_forward` CTest.**
3. **[Owners: U & S — Follow-Up `KI-010` / `KI-011`] Unify `TransformerEngine::init()` with `ModelLoader::bind_transformer_weights()` and Compare Per-Step Decode Logits in `test_offline_golden_fixture_if_present()`.**
