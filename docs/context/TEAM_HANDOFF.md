# `TEAM_HANDOFF.md` — Four-Member Division & Branch Handoff Guide

**Last Updated:** 2026-10-31  
*(Preserves exact file ownership and branch assignments from [`docs/team/OWNERSHIP.md`](../team/OWNERSHIP.md). No GitHub usernames are invented.)*

---

## 1. Ownership & Branch Map

| Member | Role | Hardware | Feature Branch | GitHub Issue |
|---|---|---|---|---|
| **S** | Project Lead, Model Loader & Checkpoint Conversion | Apple MacBook Air M4 (24 GB RAM, 1 TB SSD, `NEON`) | `feature/model-loader` | `#1` |
| **I** | Quantisation, SIMD (`NEON`/`AVX2`), ThreadPool & Dispatch | AMD Ryzen 7 7745HX (16 GB RAM, RTX 4060 unused, `AVX2`) | `feature/quantisation-simd` | `#2` |
| **U** | Scalar Reference Kernels, `KvCache` & `TransformerEngine` | Intel Core Ultra 7 155H (16 GB DDR5, `AVX2`) | `feature/transformer` | `#3` |
| **P** | SentencePiece Tokenizer, Sampler, CLI, CI & Benchmarks | Intel Core i5-10210U (8 GB RAM, `AVX2` runtime checked) | `feature/runtime-tokenizer` | `#4` |

---

## 2. Per-Member Handoff Briefs

### Member S — Project Lead & Model Loader (`feature/model-loader`)
- **Owned Files:**
  - `src/loader/loader.cpp`, `include/inference_engine/loader.hpp`
  - `src/common/common.cpp`, `src/common/tensor.cpp`, `include/inference_engine/common.hpp`, `include/inference_engine/tensor.hpp`, `include/inference_engine/config.hpp`
  - `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`
  - `tests/test_loader_validation.cpp`
  - `CURRENT_STATUS.md` and `docs/context/*`
- **Completed (M0 & M1 Issue `#1` in commit `aa40368`, merged to `integration`):** `.ie` v1 binary specification, `ModelLoader::load_from_buffer()`, `ModelLoader::load_from_file()` (`mmap`), `ModelLoader::bind_transformer_weights()`, two-pass streaming `BF16 -> FP32` `.safetensors -> .ie` conversion (`tools/convert_to_inference_engine.py`), and live 201-tensor header verification (`tools/inspect_checkpoint.py --inspect-remote-hf`).
- **Immediate Next Step:** Download the 2.20 GB `model.safetensors` payload locally to generate the full 4.40 GB `tinyllama-1.1b-fp32.ie` model and run real-checkpoint golden validation with U.
- **Start Command:**
  ```bash
  cd ~/Developer/inference-engine && git checkout integration
  ```

---

### Member I — Quantisation & CPU Optimisation (`feature/quantisation-simd`)
- **Owned Files:**
  - `src/kernels/quant_q4_0.cpp`, `include/inference_engine/quant.hpp`
  - `src/kernels/neon/kernels_neon.cpp`
  - `src/kernels/avx2/kernels_avx2.cpp`
  - `src/kernels/threading/thread_pool.cpp`, `include/inference_engine/thread_pool.hpp`
  - `src/kernels/dispatch.cpp`, `include/inference_engine/kernels.hpp`
  - `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp`
- **Already Done (M0):** `BlockQ4_0` 18-byte struct, `fp32_to_fp16` / `fp16_to_fp32`, `quantize_row_q4_0`, `dequantize_row_q4_0`, persistent `ThreadPool`, runtime `NEON`/`AVX2` detection, and vectorized `FP32` dot products (`neon::dot_f32`, `avx2::dot_f32`).
- **Immediate Next Step (M1/M2 — Issue `#2`):** Verify `quantize_row_q4_0` bit-level parity against upstream GGML reference (`KI-007`), test `avx2::dot_f32` natively on Ryzen 7 7745HX, and implement vectorized `avx2::dot_q4_0_f32` and `neon::dot_q4_0_f32` (testing NEON on S's M4).
- **Start Command:**
  ```bash
  git clone https://github.com/shreyaas5/inference-engine.git && cd inference-engine && git checkout feature/quantisation-simd
  ```

---

### Member U — Scalar Inference & Transformer (`feature/transformer`)
- **Owned Files:**
  - `src/kernels/scalar/kernels_scalar.cpp`
  - `src/core/kv_cache.cpp`, `include/inference_engine/kv_cache.hpp`
  - `src/transformer/transformer.cpp`, `include/inference_engine/transformer.hpp`
  - `tools/generate_golden_reference.py`
  - `tests/test_kernels_scalar.cpp`, `tests/test_kv_cache.cpp`, `tests/test_golden_forward.cpp`
- **Completed (M0 & M1 Issue `#3` in commit `5115467`, merged to `integration`):** Scalar `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, and `gqa_attention_f32`; `KvCache` (`append` vs `commit_token`); 16-step `FP32` forward pass in `TransformerEngine::forward_token()`; and `"GOLD"` v1 fixture generator in `tools/generate_golden_reference.py`.
- **Immediate Next Step:** Unify `TransformerEngine::init()` with `ModelLoader::bind_transformer_weights()` (`KI-010`), fix decode logit comparison in `test_offline_golden_fixture_if_present()` (`KI-011`), and run full 22-layer golden validation on the real `TinyLlama-1.1B-Chat-v1.0` checkpoint.
- **Start Command:**
  ```bash
  git clone https://github.com/shreyaas5/inference-engine.git && cd inference-engine && git checkout feature/transformer
  ```

---

### Member P — Tokenizer, Runtime & Benchmarks (`feature/runtime-tokenizer`)
- **Owned Files:**
  - `src/tokenizer/tokenizer.cpp`, `include/inference_engine/tokenizer.hpp`
  - `src/sampler/sampler.cpp`, `include/inference_engine/sampler.hpp`
  - `src/app/main.cpp`
  - `tools/verify_tokenizer_parity.py`
  - `tests/test_tokenizer.cpp`
  - `.github/workflows/ci.yml`
- **Already Done (M0):** TinyLlama Zephyr chat template (`Tokenizer::format_chat_prompt`), greedy argmax (`Sampler::sample_greedy`), CLI `--help`/`--info` flags, and GitHub Actions CI workflow.
- **Immediate Next Step (M1 — Issue `#4`):** Link SentencePiece C++ in `CMakeLists.txt` and `src/tokenizer/tokenizer.cpp`, replace the M0 byte-offset fallback with real SentencePiece encoding/decoding, verify parity via `tools/verify_tokenizer_parity.py`, and wire the end-to-end generation CLI in `src/app/main.cpp`.
- **Start Command:**
  ```bash
  git clone https://github.com/shreyaas5/inference-engine.git && cd inference-engine && git checkout feature/runtime-tokenizer
  ```

---

## 3. Integration Rules Across Feature Branches

1. **Public Header Changes (`include/inference_engine/*.hpp`):** Coordinate any signature changes with **S** before merging into `integration` so parallel feature branches do not break compilation.
2. **Pull Request Flow:** Open PRs from `feature/*` -> `integration`. Ensure GitHub Actions CI passes on both Ubuntu x86-64 and macOS ARM64 and obtain 1 peer review before merging.

