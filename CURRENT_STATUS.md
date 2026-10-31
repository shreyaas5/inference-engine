# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-10-31T18:25:00+05:30  
**Current Milestone:** `M1 — FP32 scalar inference` (In Progress — Issue `#1` implemented on `feature/model-loader`)  
**Maintainer:** S (Project Lead)

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `feature/model-loader`
- **Commits on `feature/model-loader` ahead of `origin/feature/model-loader` (`cb85411`):**
  1. `60b103e` — `docs: add context-transfer documentation system and M0 status audit`
  2. Latest M1 implementation commit (see `git log -n 2`) — implements Issue `#1`: `TinyLlama` `.safetensors` inspection, streaming `BF16 -> FP32` `.ie` v1 conversion, `ModelLoader::bind_transformer_weights()`, and expanded malformed-file + bit-exact numerical preservation tests.
- **Working Tree State:** Clean (`git status` reports nothing to commit; `models/` and `build/` are untracked/cleaned). Not pushed to `origin` pending review/approval.
- **CMake Build Status:** **PASSING** (`AppleClang 17.0.0.17000603`, `CMake 4.4.3`, zero compiler warnings with `-Wall -Wextra -Wpedantic`).
- **CTest Suite Status:** **7 / 7 PASSING** (verified on Apple M4 ARM64):
  1. `loader_validation` — **PASSED** (12 malformed-file rejection checks + 21-tensor `mmap` reload, `bind_transformer_weights()`, and bit-exact `FP32` data preservation).
  2. `quant_packing` — **PASSED**
  3. `kernels_scalar` — **PASSED**
  4. `kernels_simd` — **PASSED**
  5. `kv_cache` — **PASSED**
  6. `tokenizer` — **PASSED**
  7. `golden_forward` — **PASSED** (synthetic position-indexing contract test)
- **Python Offline Tool Self-Tests:**
  - `python3 tools/inspect_checkpoint.py --inspect-remote-hf` — **PASSED** (Fetched live 23,088-byte `model.safetensors` header and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via HTTP Range and verified all `201/201` `BF16` tensors and `2,200,096,768` payload bytes against `ModelConfig`).
  - `python3 tools/convert_to_inference_engine.py --self-test` — **PASSED** (Verified streaming `BF16 -> FP32` `.safetensors -> .ie` conversion, 64-byte alignment, bit-exact `FP32` values, and C++ `ModelLoader` reload across `21` synthetic 2-layer Llama tensors).

---

## 2. Honest Implementation Breakdown: What Actually Works vs What Is Stubbed

### Completed & Tested in Code (`IMPLEMENTED` / `TESTED` / `REFERENCE VERIFIED`)
- **`.ie` v1 Binary Format Parser, Writer, Validator & Tensor Binder (`src/loader/loader.cpp`, `include/inference_engine/loader.hpp`):**
  - Explicit little-endian 64-byte header and variable-length tensor directory serialization/parsing.
  - Checked 64-bit integer arithmetic (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`), non-finite float (`NaN`/`Inf`) rejection, and trailing-byte rejection.
  - 64-byte payload alignment verification (`payload_offset % 64 == 0`), exact byte-size checks, `Q4_0` innermost-dimension divisibility by 32 check, and sorted interval non-overlap verification.
  - POSIX `mmap` file loader (`load_from_file`) with verified 64-byte aligned virtual addresses (`reinterpret_cast<uintptr_t>(raw_data()) % 64 == 0`).
  - `ModelLoader::validate_tensor_binding()` and `ModelLoader::bind_transformer_weights(TransformerWeightBindings*)` verifying the complete `3 + 9 * num_layers` (`201` for TinyLlama) tensor inventory, shapes, and dtypes (`FP32` for 1D norms, `config.weight_dtype` for 2D matrices).
- **Offline `.safetensors` Inspector & Streaming `FP32` Converter (`tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`):**
  - Parses `.safetensors` 8-byte header size and JSON metadata, validates all `3 + 9 * num_layers` tensor names/shapes/offsets against `config.json`, preserves exact tensor ordering, and performs two-pass streaming `BF16/F16/F32 -> FP32` conversion to `.ie` v1 in bounded 4 MiB chunks without loading multiple model copies into RAM.
  - Works with zero external Python dependencies (pure-Python `struct`/`array` path) and automatically accelerates `BF16 -> FP32` via `numpy` when available.
- **`Q4_0` Block Packing, Quantisation & Dequantisation (`src/kernels/quant_q4_0.cpp`):**
  - 18-byte `BlockQ4_0` layout (`uint16_t d` + `uint8_t qs[16]`), IEEE 754 `fp32_to_fp16` / `fp16_to_fp32` conversion, split-half nibble packing (`0..15` low nibble, `16..31` high nibble), signed max scale calculation (`d = signed_max / -8.0f`), and zero-block safe handling.
- **Core Scalar Kernels (`src/kernels/scalar/kernels_scalar.cpp`):**
  - `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, and `softmax_f32`.
- **SIMD `FP32` Dot Product & Persistent `ThreadPool` (`src/kernels/neon/`, `src/kernels/avx2/`, `src/kernels/threading/`):**
  - `neon::dot_f32`, `avx2::dot_f32`, `ThreadPool::parallel_for`, and runtime CPU feature detection.
- **`KvCache` Storage & Explicit Commit Contract (`src/core/kv_cache.cpp`):**
  - Contiguous `FP32` key/value storage with `append()` decoupled from `commit_token()`.
- **TinyLlama Chat Template & Greedy Sampler (`src/tokenizer/tokenizer.cpp`, `src/sampler/sampler.cpp`):**
  - `Tokenizer::format_chat_prompt()` and `Sampler::sample_greedy()`.

### In-Progress / Stubbed (`IN PROGRESS` / `NOT STARTED` — Remaining M1–M4 Work)
- **Full 22-Layer Transformer Forward Pass (`src/transformer/transformer.cpp` — Owner: U, Issue `#3`):**
  - **STUB:** `TransformerEngine::forward_token()` still outputs zero KV and zero logits; it now has `ModelLoader::bind_transformer_weights()` ready to bind all 201 tensors once U wires the layer math.
- **Causal Grouped Query Attention (GQA) Kernel (`src/kernels/scalar/` — Owner: U, Issue `#3`):**
  - **NOT STARTED:** Multi-head / grouped-query causal attention score calculation (`8:1` Q-to-KV head mapping).
- **Golden Reference Generator (`tools/generate_golden_reference.py` — Owner: U, Issue `#3`):**
  - **STUB:** CLI argument parser only.
- **SentencePiece C++ Tokenizer Integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt` — Owner: P, Issue `#4`):**
  - **STUB:** `Tokenizer::encode()` and `decode()` still use the M0 synthetic byte-offset fallback (`byte + 3`).
- **`Q4_0` Export in `tools/convert_to_inference_engine.py` (Owner: S/I, Milestone M2):**
  - `--dtype fp32` is implemented and tested; `--dtype q4_0` returns an explicit M2 status message.
- **Vectorized `Q4_0` SIMD Kernels (`src/kernels/neon/kernels_neon.cpp`, `src/kernels/avx2/kernels_avx2.cpp` — Owner: I, Milestone M3):**
  - **STUB:** `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` currently delegate to `scalar::dot_q4_0_f32`.

---

## 3. Active Blockers & Limitations

1. **Full 2.2 GB `model.safetensors` Payload Not Stored on Disk Yet:**
   - The live 23,088-byte `model.safetensors` header and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` were fetched and verified via HTTP Range (`201/201` tensors verified), and end-to-end `.safetensors -> .ie` conversion + C++ `mmap` reload was verified on a 21-tensor synthetic checkpoint. Downloading the full 2.20 GB payload over a single HTTP stream was ~1.2 MB/s (~30 min), so full 4.40 GB `.ie` file generation should be run offline once `model.safetensors` is downloaded locally.
2. **Missing SentencePiece C++ Library on Host (`KI-008`):**
   - Needed for Issue `#4` (Member P).

---

## 4. Next Three Engineering Priorities (Milestone M1)

1. **[Owner: U — Issue #3] Implement Causal GQA Kernel & Wire 22-Layer FP32 Forward Pass (`src/transformer/transformer.cpp`):**
   - Call `loader->bind_transformer_weights(&weights_)` in `TransformerEngine::init()` and execute all 22 transformer layers in `forward_token()`.
2. **[Owner: P — Issue #4] Integrate SentencePiece C++ Tokenizer (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`):**
   - Link SentencePiece C++ and verify token ID parity via `tools/verify_tokenizer_parity.py`.
3. **[Owner: U & S — Issue #3] Download Full TinyLlama Checkpoint Offline, Convert to `.ie` FP32, and Run Golden Validation:**
   - Convert full `TinyLlama-1.1B-Chat-v1.0` with `tools/convert_to_inference_engine.py --dtype fp32` and compare multi-token logits against Hugging Face FP32 golden references.
