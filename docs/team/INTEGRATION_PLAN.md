# Milestone & Integration Plan

## Milestone Roadmap

### M0 — Repository Foundation (`m0-foundation`)
- **Goal:** Establish buildable C++17 scaffolding, frozen public headers, `.ie` and `Q4_0` contracts, deterministic CTest suite, GitHub Actions CI, and team workflow.
- **Exit Criteria:** CMake builds cleanly on macOS ARM64 and Ubuntu x86-64; all 7 foundational CTest suites pass; branches and initial GitHub issues are active.

### M1 — FP32 Scalar Inference (`m1-fp32`)
- **Goal:** End-to-end correct FP32 inference of `TinyLlama-1.1B-Chat-v1.0` using scalar reference kernels.
- **Integration Touchpoints:**
  1. **S** delivers `convert_to_inference_engine.py --dtype fp32` and `ModelLoader` exposing all 201 TinyLlama tensors via `find_tensor()`.
  2. **P** delivers SentencePiece `Tokenizer` passing `verify_tokenizer_parity.py` and greedy `Sampler`.
  3. **U** wires `ModelLoader` weights + `KvCache` + scalar kernels into `TransformerEngine::forward_token()` and verifies multi-token golden parity against Hugging Face FP32 outputs.

### M2 — Q4_0 Quantised Inference (`m2-q4`)
- **Goal:** Compress 2D linear weights to `Q4_0` (~630 MB `.ie` file) and run end-to-end quantised inference across all laptops (including P's 8 GB machine).
- **Integration Touchpoints:**
  1. **I** finalizes `quantize_row_q4_0` / `dequantize_row_q4_0` and scalar `dot_q4_0_f32` parity tests.
  2. **S** integrates `Q4_0` block emission into `convert_to_inference_engine.py --dtype q4_0`.
  3. **U** enables mixed-precision weight dispatch (`FP32` norms + `Q4_0` projections) in `TransformerEngine` and calibrates Q4_0 logit tolerances.

### M3 — CPU Optimisation (`m3-optimised`)
- **Goal:** Maximize CPU prefill and decode throughput using NEON, AVX2, and multi-threaded execution.
- **Integration Touchpoints:**
  1. **I** implements vectorized `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` plus `ThreadPool` row-parallel matrix-vector multiplication.
  2. **S** and **I** profile and validate NEON on Apple M4 and AVX2 on Ryzen 7 7745HX.
  3. **U** and **P** validate AVX2 and thread scaling on Intel Core Ultra 7 155H and Core i5-10210U.

### M4 — Final Demonstration & Benchmarking (`m4-demo`)
- **Goal:** Polished CLI, automated benchmark harness, four-laptop performance report, and head-to-head comparison against `llama.cpp`.
- **Integration Touchpoints:**
  1. **P** finalizes CLI flags (`--prompt`, `--chat`, `--benchmark`, `--threads`, `--temp`, `--top-p`).
  2. **All members (S, I, U, P)** execute the standardized benchmark protocol on their respective hardware and commit verified results to `docs/BENCHMARKING.md`.
