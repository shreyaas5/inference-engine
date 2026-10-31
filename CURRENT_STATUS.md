# Current Implementation Status (`CURRENT_STATUS.md`)

**Last Updated:** 2026-10-31  
**Active Milestone:** `M0 — Repository Foundation` (`m0-foundation`)  
**Maintainer:** S (Project Lead)

## Milestone Tracker

| Milestone | Tag | Status | Summary |
|---|---|---|---|
| **M0 — Repository Foundation** | `m0-foundation` | **Complete** | CMake scaffolding, public headers, contract tests, docs, CI, Git branches, and GitHub issues initialized. |
| **M1 — FP32 Scalar Inference** | `m1-fp32` | **Pending** | Full safetensors -> `.ie` FP32 converter, SentencePiece integration, 22-layer scalar forward pass, HF golden parity. |
| **M2 — Q4_0 Quantisation** | `m2-q4` | **Pending** | `.ie` Q4_0 model export, mixed-precision loader & matmul integration, Q4_0 numerical validation. |
| **M3 — CPU Optimisation** | `m3-optimised` | **Pending** | Vectorized NEON & AVX2 Q4_0 dot-product kernels, multi-threaded row dispatch, profiling & thread scaling. |
| **M4 — Final Demonstration** | `m4-demo` | **Pending** | Production CLI, benchmark runner, 4-laptop benchmark report, and `llama.cpp` comparison. |

## Subsystem Status

| Subsystem | Owner | Status | Notes |
|---|---|---|---|
| `.ie` v1 Format & `ModelLoader` | **S** | M0 Contract Ready | 64-byte alignment, checked arithmetic, non-overlap checks, and `mmap` verified in `loader_validation`. |
| `Q4_0` Packing & Dequantisation | **I** | M0 Contract Ready | 18-byte `BlockQ4_0`, split-half nibble layout, FP16 conversion, and zero-block handling verified in `quant_packing`. |
| SIMD (`NEON` / `AVX2`) & `ThreadPool` | **I** | M0 Scaffolding Ready | Runtime CPU detection, persistent `ThreadPool`, and FP32 dot/matmul dispatch active; vectorized `Q4_0` SIMD scheduled for M3. |
| Scalar Kernels & `KvCache` | **U** | M0 Contract Ready | Reference RMSNorm, RoPE, Softmax, SwiGLU, and `KvCache` (`append` vs `commit_token`) verified in CTest. |
| `TransformerEngine` Forward Pass | **U** | M0 Scaffolding Ready | Position indexing contract (`0..11` prefill -> `12` decode) verified; full 22-layer weight binding scheduled for M1. |
| `Tokenizer`, `Sampler` & CLI | **P** | M0 Scaffolding Ready | TinyLlama Zephyr chat template, greedy argmax sampler, and `--help`/`--info` CLI active; SentencePiece link scheduled for M1. |
