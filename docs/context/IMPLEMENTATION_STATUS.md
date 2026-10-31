# `IMPLEMENTATION_STATUS.md` — Module-by-Module Truth Table

> **Last Updated:** 2026-11-01
> **Consolidated Baseline:** `integration-ps08-team-sync-2026-10-31` (`ad246cb`)
> **Active Feature Branch:** `feature/cpu-kernel-optimisation` (Member I)

---

## 1. Evidence Classification Legend

- **`VERIFIED`:** Directly inspected in repository code and Git history with reproducible evidence.
- **`TESTED`:** Automated unit tests in `tests/` exist and pass.
- **`SYNTHETICALLY VALIDATED`:** Verified end-to-end using deterministic synthetic weights or generated reference fixtures.
- **`REAL-CHECKPOINT VALIDATED`:** Verified directly against real `TinyLlama/TinyLlama-1.1B-Chat-v1.0` assets (`config.json`, `tokenizer.model`, or live `model.safetensors`).
- **`PARTIALLY IMPLEMENTED`:** Core functionality works, but a sub-feature or platform optimization remains incomplete.
- **`BLOCKED`:** Cannot complete until an upstream dependency (specifically the remaining 1.87 GB of `model.safetensors`) is available.

---

## 2. Module-by-Module Implementation Matrix

| Module / File(s) | Owner | Status Classification | Verification Evidence & Notes |
| :--- | :---: | :--- | :--- |
| `include/inference_engine/common.hpp`<br>`src/common/status.cpp` | **S** | `VERIFIED` / `TESTED` | `Status`, `StatusCode` (`0..7`), `DataType` (`FP32=0, FP16=1, Q4_0=2, I32=3`), `SimdBackend`, checked 64-bit arithmetic (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`). Tested in `loader_validation`. |
| `include/inference_engine/config.hpp` | **S** | `REAL-CHECKPOINT VALIDATED` | `TinyLlamaConfig::tinyllama_1_1b()` & `ModelFileHeader` (`64` bytes). Verified against `models/TinyLlama-1.1B-Chat-v1.0/config.json`. |
| `include/inference_engine/tensor.hpp`<br>`include/inference_engine/loader.hpp`<br>`src/loader/loader.cpp` | **S** | `VERIFIED` / `TESTED` | POSIX `mmap` + Win32 `MapViewOfFile`, strict bounds/alignment/non-overlap checks, `TensorView::data_f32()` & `data_q4_0()`, `bind_tinyllama_weights()`. Passes `loader_validation` (`6/6`). |
| `tools/inspect_checkpoint.py` | **S** | `REAL-CHECKPOINT VALIDATED` | Verifies 201-tensor TinyLlama contract and evaluates real `Q4_0` quantisation error across all 9 functional tensor roles (`--eval-real-quant-error`) on `model.safetensors` (`MeanCosSim >= 0.9888`, `SQNR = 16.51–16.82 dB`). |
| `tools/convert_to_inference_engine.py` | **S** | `VERIFIED` / `TESTED` | Streaming `BF16`/`FP16`/`FP32` $\rightarrow$ `.ie` v1 converter supporting `--dtype {fp32,q4_0}` and `--keep-embeddings-fp32`. Passes `--self-test`. Full 22-layer `.ie` export is `BLOCKED` on completing the 2.20 GB `model.safetensors` download (`KI-004`). |
| `include/inference_engine/quant.hpp`<br>`src/kernels/quant_q4_0.cpp` | **I** | `VERIFIED` / `TESTED` | `BlockQ4_0` (`18` bytes / `32` weights, split-half nibbles), bitwise `fp16_to_fp32` / `fp32_to_fp16`, `quantize_row_q4_0`, `dequantize_row_q4_0`, `scalar::dot_q4_0_f32`, `scalar::matmul_q4_0_f32`. Passes `quant_packing` (`4/4`). |
| `src/kernels/avx2/kernels_avx2.cpp` | **I** | `VERIFIED` / `TESTED` | Runtime CPUID + OS XSAVE (`xgetbv`) check, vectorized `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`. On `feature/cpu-kernel-optimisation` (`b05c4e4`): adds vectorized per-block scale accumulation (`compute_block_unscaled_dot_vec` + single end-of-row `hsum256_ps`) and 2-row register tiling (`21.97x` speedup on `2048x2048` 8T, `162.80 GFLOPS` on `2048x5632` 16T on Ryzen 7 7745HX). |
| `src/kernels/neon/kernels_neon.cpp` | **I** | `VERIFIED` / `TESTED` | ARM64 NEON intrinsics (`vld1q_u8`, `vcvt_f32_f16`, `vfmaq_f32`). Upgraded in `1c046cb` with vectorized per-block scale accumulation (single end-of-row `vaddvq_f32`) and 2-row register tiling (`m` and `m + 1`), with odd-row and tail fallback. Verified via AArch64 cross-compilation and `test_simd_matmul_odd_rows_and_tails()`. |
| `include/inference_engine/thread_pool.hpp`<br>`src/kernels/threading/thread_pool.cpp`<br>`src/kernels/dispatch.cpp` | **I** | `VERIFIED` / `TESTED` | Persistent worker pool with dynamic task chunking and `std::exception_ptr` capture/rethrow (`ADR-010`). On `b05c4e4`, `dispatch.cpp` slices `out_dim` into contiguous 2-row-aligned chunks (`ADR-012`). Passes `kernels_simd` (`8/8`). |
| `benchmarks/bench_kernels.cpp`<br>`benchmarks/results/*.csv` | **I** | `IN PROGRESS` / `VERIFIED` | Added on `feature/cpu-kernel-optimisation` (`b05c4e4`). Verified on Ryzen 7 7745HX across all 8 TinyLlama shapes (1T/2T/4T/8T/16T). Upgrading to dynamic `detect_best_backend()` and `--backend` flag is `P1-2`. |
| `src/kernels/scalar/kernels_scalar.cpp` | **U** | `VERIFIED` / `TESTED` | Scalar reference `rmsnorm`, `rope_inplace` (`rotate_half`), `attention_gqa` (causal 8:1 GQA), `swiglu`, `softmax_inplace`, `matmul_f32`, `dot_f32`. Passes `kernels_scalar` (`7/7`). |
| `include/inference_engine/kv_cache.hpp`<br>`src/core/kv_cache.cpp` | **U** | `VERIFIED` / `TESTED` | Contiguous `FP32` KV cache (`[22, 2048, 4, 64]`, `45.09 MB`). Enforces `append()` vs `commit_token()`. Passes `kv_cache` (`4/4`). |
| `include/inference_engine/transformer.hpp`<br>`src/transformer/transformer.cpp` | **U** | `SYNTHETICALLY VALIDATED` | Full 16-step `TransformerEngine::forward_token()` supporting pure `FP32`, pure `Q4_0`, and mixed `Q4_0` (`--keep-embeddings-fp32`), single-row `Q4_0` embedding dequantisation, and `ThreadPool` dispatch. Passes `golden_forward` (`3/3`), `transformer_q4` (`7/7`), and `verify_transformer_q4.py` (`cos_sim >= 0.996359`). Full 22-layer real-checkpoint forward pass is `BLOCKED` on `KI-004`. |
| `include/inference_engine/tokenizer.hpp`<br>`src/tokenizer/tokenizer.cpp`<br>`tools/verify_tokenizer_parity.py` | **P** | `REAL-CHECKPOINT VALIDATED` | SentencePiece C++ `Tokenizer` (`v0.2.0` via `FetchContent`) with dual processors (`processor_` + `processor_no_prefix_`), special-token splitting, and Zephyr `apply_chat_template`. Passes `tokenizer` CTest (`4/4`) and `15/15` prompts in `verify_tokenizer_parity.py --self-test` against real `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`. |
| `include/inference_engine/sampler.hpp`<br>`src/sampler/sampler.cpp` | **P** | `VERIFIED` / `TESTED` | Greedy argmax, temperature scaling, top-p nucleus filtering, `NaN`/`-Inf` sanitization, and seeded 64-bit `xorshift*` PRNG. Passes `sampler` CTest (`6/6`). |
| `src/app/main.cpp` | **P** | `SYNTHETICALLY VALIDATED` / `PARTIALLY IMPLEMENTED` | Full CLI (`--info`, `--tokenize`, `--decode`, generation loop, `--benchmark` reporting TTFT, prefill/decode tok/s, CPU %, Peak RSS). Verified E2E on synthetic 1-layer `.ie` model + real `tokenizer.model`. Currently batch-decodes generated IDs after decode loop (incremental streaming detokenisation is `P1`); 22-layer real-model run is `BLOCKED` on `KI-004`. |
