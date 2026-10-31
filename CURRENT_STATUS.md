# `CURRENT_STATUS.md` — Single Source of Truth for Project Progress

**Last Updated:** 2026-10-31T17:55:00+05:30  
**Current Milestone:** Transitioning from `M0 — Repository foundation` (Complete) to `M1 — FP32 scalar inference` (In Progress / Next)  
**Maintainer:** S (Project Lead)

---

## 1. Git & Build State (Verified)

- **Active Local Branch:** `main` (Note: switch to a `feature/*` branch before making code changes!)
- **Latest Commit on `main` / `integration` / `feature/*`:** `cb85411` (`feat: initialize inference-engine M0 repository foundation, contracts, and CI`)
- **Latest Tag:** `m0-foundation` -> `cb85411`
- **Remote Status:** All 6 branches (`main`, `integration`, `feature/model-loader`, `feature/quantisation-simd`, `feature/transformer`, `feature/runtime-tokenizer`) and tag `m0-foundation` are pushed to `https://github.com/shreyaas5/inference-engine.git`.
- **Working Tree State:**
  - New/updated context-transfer documentation files (`AGENTS.md`, `PROJECT_CONTEXT.md`, `CURRENT_STATUS.md`, `SESSION_HANDOFF.md`, `DECISIONS.md`, `ROADMAP.md`, `TEAM_STATUS.md`, `docs/context/*.md`) are uncommitted in the working tree (per explicit user instruction not to commit or push until requested).
  - Note: 53 existing files also show a 1-line trailing newline diff (`+1` blank line at EOF) introduced by the IDE buffer formatter after commit `cb85411`; no functional code differs from `cb85411`.
- **CMake Build Status:** **PASSING** (`AppleClang 17.0.0.17000603`, `CMake 4.4.3`, zero compiler warnings with `-Wall -Wextra -Wpedantic`).
- **CTest Suite Status:** **7 / 7 PASSING** (verified on Apple M4 ARM64):
  1. `loader_validation` — **PASSED**
  2. `quant_packing` — **PASSED**
  3. `kernels_scalar` — **PASSED**
  4. `kernels_simd` — **PASSED**
  5. `kv_cache` — **PASSED**
  6. `tokenizer` — **PASSED**
  7. `golden_forward` — **PASSED** (synthetic contract test only; not yet real TinyLlama weights)

---

## 2. Honest Implementation Breakdown: What Actually Works vs What Is Stubbed

### Completed & Tested in Code (`IMPLEMENTED` / `TESTED`)
- **`.ie` v1 Binary Format Parser, Writer & Validator (`src/loader/loader.cpp`, `src/common/tensor.cpp`):**
  - Explicit little-endian 64-byte header and variable-length tensor directory serialization/parsing.
  - Checked 64-bit integer arithmetic (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`).
  - 64-byte payload alignment verification (`payload_offset % 64 == 0`), exact byte-size checks, `Q4_0` innermost-dimension divisibility by 32 check, and sorted interval non-overlap verification.
  - POSIX `mmap` file loader (`load_from_file`) and in-memory buffer loader (`load_from_buffer`).
- **`Q4_0` Block Packing, Quantisation & Dequantisation (`src/kernels/quant_q4_0.cpp`):**
  - 18-byte `BlockQ4_0` layout (`uint16_t d` + `uint8_t qs[16]`), IEEE 754 `fp32_to_fp16` / `fp16_to_fp32` conversion, split-half nibble packing (`0..15` low nibble, `16..31` high nibble), signed max scale calculation (`d = signed_max / -8.0f`), and zero-block safe handling.
- **Core Scalar Kernels (`src/kernels/scalar/kernels_scalar.cpp`):**
  - `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, and `softmax_f32` are implemented and unit-tested on synthetic vectors.
- **SIMD `FP32` Dot Product & Persistent `ThreadPool` (`src/kernels/neon/`, `src/kernels/avx2/`, `src/kernels/threading/`):**
  - `neon::dot_f32` (`vfmaq_f32` + `vaddvq_f32`) and `avx2::dot_f32` (`_mm256_fmadd_ps`) are implemented.
  - Persistent `std::thread` pool (`ThreadPool::parallel_for`) and runtime CPU feature detection (`detect_best_backend`, `is_backend_available`) are implemented and tested.
- **`KvCache` Storage & Explicit Commit Contract (`src/core/kv_cache.cpp`):**
  - Contiguous `FP32` key/value storage (`[num_layers, max_seq_len, kv_dim]`).
  - `append(layer_idx, pos, k, v)` writes K/V without mutating `seq_len()`; `commit_token()` / `commit_positions()` explicitly advances `seq_len()`.
- **TinyLlama Chat Template & Greedy Sampler (`src/tokenizer/tokenizer.cpp`, `src/sampler/sampler.cpp`):**
  - `Tokenizer::format_chat_prompt()` formats `<|system|>`, `<|user|>`, and `<|assistant|>` Zephyr tags.
  - `Sampler::sample_greedy()` implements deterministic argmax with lowest-index tie-breaking.

### In-Progress / Stubbed (`IN PROGRESS` / `NOT STARTED` — Must Be Completed in M1–M4)
- **Full 22-Layer Transformer Forward Pass (`src/transformer/transformer.cpp`):**
  - **STUB:** `TransformerEngine::forward_token()` currently validates bounds, appends zero K/V vectors across `num_layers`, commits the token position, and fills `out_logits` with `0.0f`. It does **not** yet bind `.ie` tensor weights or execute embedding lookup, RMSNorm, QKV projections, RoPE, causal GQA attention, SwiGLU MLP, or the LM head.
- **Causal Grouped Query Attention (GQA) Kernel (`src/kernels/scalar/`):**
  - **NOT STARTED:** Multi-head / grouped-query causal attention score calculation ($Q K^T / \sqrt{64}$ across cached positions $0..\text{pos}$, softmax, and weighted $V$ sum with `8:1` Q-to-KV head mapping) is not yet implemented as a standalone kernel or inside `transformer.cpp`.
- **Offline Checkpoint-to-`.ie` Converter (`tools/convert_to_inference_engine.py`):**
  - **STUB:** Header packing (`pack_header`) and `--self-test` are implemented, but reading Hugging Face `.safetensors`, casting `BF16 -> FP32`, quantising to `Q4_0`, and writing full `.ie` model files is not yet implemented.
- **Golden Reference Generator (`tools/generate_golden_reference.py`):**
  - **STUB:** CLI argument parser only; does not yet load Hugging Face `TinyLlama-1.1B-Chat-v1.0` or export reference tensors.
- **SentencePiece C++ Tokenizer Integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`):**
  - **STUB:** `Tokenizer::encode()` and `decode()` currently use a synthetic byte-offset fallback (`byte + 3`) for M0 unit testing. SentencePiece C++ library is not yet linked in `CMakeLists.txt` or installed via Homebrew.
- **Vectorized `Q4_0` SIMD Kernels (`src/kernels/neon/kernels_neon.cpp`, `src/kernels/avx2/kernels_avx2.cpp`):**
  - **STUB:** `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` currently delegate to `scalar::dot_q4_0_f32`.
- **Stochastic Sampling (`src/sampler/sampler.cpp`) & End-to-End CLI (`src/app/main.cpp`):**
  - **STUB:** `Sampler::sample()` falls back to greedy argmax; `src/app/main.cpp` handles `--help` and `--info` only.

---

## 3. Active Blockers & Environment Findings

1. **Missing Python ML Packages in Default System Python:**
   - System `/opt/homebrew/bin/python3` (`Python 3.14.6`) does not have `torch`, `transformers`, `safetensors`, `sentencepiece`, or `numpy`.
   - However, Miniconda environments on S's machine (`/Users/shreyaas/cwq/miniconda3` and `/Users/shreyaas/cwq/miniconda3/envs/cwq`) already have `torch 2.8.0` and `numpy` installed; `transformers`, `safetensors`, and `sentencepiece` still need to be installed in a Python environment before running offline conversion/golden scripts.
2. **Missing SentencePiece C++ Library on Host:**
   - `protobuf` and `abseil` are installed via Homebrew on S's machine, but `sentencepiece` C++ library (`brew install sentencepiece` or CMake `FetchContent`) is not yet installed/configured.
3. **No Local TinyLlama Checkpoint Downloaded Yet:**
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` weights are not yet downloaded locally under `models/` or `checkpoints/`.
4. **Teammate GitHub Collaborator Invites Pending:**
   - Collaborator usernames for **I**, **U**, and **P** have not been provided yet.

---

## 4. Next Three Engineering Priorities (Milestone M1)

1. **[Owner: S — Issue #1] Implement Full `safetensors -> .ie` Converter (`tools/convert_to_inference_engine.py`):**
   - Implement streaming/memory-efficient `safetensors` header & tensor reading, `BF16 -> FP32` conversion, and `.ie` v1 binary file emission, plus tensor inventory verification against TinyLlama's 201 weight tensors.
2. **[Owner: U — Issue #3] Implement Causal GQA Attention & Full 22-Layer FP32 Forward Pass (`src/transformer/transformer.cpp`):**
   - Bind all TinyLlama layer weights from `ModelLoader`, implement causal GQA (`32` Q heads, `4` KV heads, `head_dim = 64`), and wire the full 22-layer forward pass in `TransformerEngine::forward_token()`.
3. **[Owner: P — Issue #4] Integrate SentencePiece C++ Tokenizer (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`):**
   - Integrate SentencePiece C++ processor, replace the byte-offset fallback in `Tokenizer::encode`/`decode`, and verify exact token ID parity with `tools/verify_tokenizer_parity.py`.
