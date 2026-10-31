# `IMPLEMENTATION_STATUS.md` — Module-by-Module Truth Table

**Last Updated:** 2026-10-31  
**Baseline Commit:** `cb85411` (`m0-foundation`)

Allowed Status Labels:
- `NOT STARTED` — No implementation exists yet (or CLI argument parser skeleton only).
- `IN PROGRESS` — Partial implementation or M0 scaffolding stub exists; core functionality still pending.
- `IMPLEMENTED` — Full logic written in code, pending unit/reference verification.
- `TESTED` — Full logic implemented and verified by deterministic synthetic unit tests in CTest.
- `REFERENCE VERIFIED` — Verified against external trusted reference (Hugging Face FP32 golden outputs or upstream GGML/SentencePiece).

---

## Subsystem Implementation Matrix

| Module | Primary Files | Planned | Implemented | Tested (CTest) | Verified vs Reference | Status | Notes & Evidence |
|---|---|---|---|---|---|---|---|
| **Common & Dtype Metadata** | `include/inference_engine/common.hpp`, `src/common/common.cpp` | Yes | Yes | Yes | N/A | `TESTED` | Checked `u64` arithmetic, `DataType` block metadata, runtime `NEON`/`AVX2` detection. |
| **Model Config & Tensor Descriptors** | `include/inference_engine/config.hpp`, `include/inference_engine/tensor.hpp`, `src/common/tensor.cpp` | Yes | Yes | Yes | N/A | `TESTED` | Validates TinyLlama dims, 64B alignment, exact byte size, and `Q4_0` innermost dim `% 32 == 0`. |
| **`.ie` v1 Model Loader & Writer** | `include/inference_engine/loader.hpp`, `src/loader/loader.cpp` | Yes | Yes | Yes | No ( awaits real `.ie` checkpoint) | `TESTED` | Little-endian header/directory parser, `mmap` loader, `build_ie_image`, bounds & overlap checks (`loader_validation` suite). |
| **Offline `.safetensors -> .ie` Converter** | `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py` | Yes | Partial (Header only) | Self-test only | No | `IN PROGRESS` | 64B header packer works (`--self-test`); reading `.safetensors` and writing tensor payloads is **not yet implemented** (M1 Issue `#1`). |
| **`Q4_0` Quantisation & Dequantisation** | `include/inference_engine/quant.hpp`, `src/kernels/quant_q4_0.cpp` | Yes | Yes | Yes | No (pending GGML cross-check) | `TESTED` | 18B `BlockQ4_0`, split-half nibbles, IEEE 754 FP16 conversion, zero-block handling (`quant_packing` suite). |
| **Scalar Reference Kernels** | `include/inference_engine/kernels.hpp`, `src/kernels/scalar/kernels_scalar.cpp` | Yes | Partial (Missing GQA) | Yes (Core ops) | No (pending HF golden check) | `IN PROGRESS` | `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32` are `TESTED`; causal GQA kernel is `NOT STARTED`. |
| **SIMD Kernels (ARM64 NEON)** | `src/kernels/neon/kernels_neon.cpp` | Yes | Partial (`FP32` only) | Yes (`FP32` dot) | No | `IN PROGRESS` | `neon::dot_f32` (`vfmaq_f32`) is `TESTED` on M4; `neon::dot_q4_0_f32` is a stub delegating to `scalar::dot_q4_0_f32` (M3). |
| **SIMD Kernels (x86-64 AVX2)** | `src/kernels/avx2/kernels_avx2.cpp` | Yes | Partial (`FP32` only) | Compiled (Scalar fallback on ARM64) | No | `IN PROGRESS` | `avx2::dot_f32` (`_mm256_fmadd_ps`) is written; `avx2::dot_q4_0_f32` delegates to scalar (M3). Needs x86-64 hardware/CI test run. |
| **Thread Pool & Kernel Dispatch** | `include/inference_engine/thread_pool.hpp`, `src/kernels/threading/thread_pool.cpp`, `src/kernels/dispatch.cpp` | Yes | Yes | Yes | N/A | `TESTED` | Persistent `std::thread` pool (`parallel_for`) and row-parallel `dispatch_matmul_*` tested in `kernels_simd`. |
| **KV Cache** | `include/inference_engine/kv_cache.hpp`, `src/core/kv_cache.cpp` | Yes | Yes | Yes | N/A | `TESTED` | Contiguous `FP32` cache with strict separation between `append()` and `commit_token()` (`kv_cache` suite). |
| **Transformer Forward Pass** | `include/inference_engine/transformer.hpp`, `src/transformer/transformer.cpp` | Yes | Stub only | Contract test only | No | `IN PROGRESS` | `forward_token()` currently appends zero KV and returns zero logits to test position indexing (`0..11` -> `12`). Full 22-layer execution is `NOT STARTED` (M1 Issue `#3`). |
| **Golden Reference Generator** | `tools/generate_golden_reference.py` | Yes | No (CLI stub) | No | No | `NOT STARTED` | Argument parser skeleton only; needs PyTorch/HF implementation in M1. |
| **Tokenizer & Chat Template** | `include/inference_engine/tokenizer.hpp`, `src/tokenizer/tokenizer.cpp`, `tools/verify_tokenizer_parity.py` | Yes | Partial (Chat template only) | Yes (Chat template) | Yes (Zephyr template format) | `IN PROGRESS` | `format_chat_prompt()` is `TESTED`; `encode`/`decode` use a synthetic `byte + 3` stub until SentencePiece C++ is linked (M1 Issue `#4`). |
| **Sampler** | `include/inference_engine/sampler.hpp`, `src/sampler/sampler.cpp` | Yes | Partial (Greedy only) | Yes (Greedy) | N/A | `IN PROGRESS` | `sample_greedy()` is `TESTED`; stochastic temperature/top-p `sample()` currently delegates to `sample_greedy()`. |
| **CLI Application & Benchmark Runner** | `src/app/main.cpp` | Yes | Partial (`--help`/`--info`) | Smoke-tested | N/A | `IN PROGRESS` | Handles `--help` and `--info`; model inference and `--benchmark` loops are `NOT STARTED` (M1/M4). |

