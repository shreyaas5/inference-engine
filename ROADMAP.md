# `ROADMAP.md` — Milestone Plan, Dependencies & Completion Status

**Last Updated:** 2026-10-31  
**Competition:** BIOS V2 — Problem Statement PS08 (Own Inference Engine)

---

## Milestone Overview

| Milestone | Git Tag | GitHub Milestone | Status | Target Deliverable |
|---|---|---|---|---|
| **M0 — Repository foundation** | `m0-foundation` | `#1` (Closed) | **COMPLETE** | Buildable C++17 scaffolding, frozen public headers, `.ie` & `Q4_0` contracts, 7 CTest suites, CI, context-transfer system. |
| **M1 — FP32 scalar inference** | `m1-fp32` | `#2` (Open) | **IN PROGRESS** | Full `.safetensors -> .ie` (FP32) converter, SentencePiece tokenizer parity, 22-layer scalar forward pass, HF golden validation. |
| **M2 — Q4_0 quantisation** | `m2-q4` | `#3` (Open) | **NOT STARTED** | `.ie` Q4_0 model export, mixed-precision loader & matmul integration, Q4_0 logit/continuation validation. |
| **M3 — CPU optimisation** | `m3-optimised` | `#4` (Open) | **NOT STARTED** | Vectorized NEON & AVX2 `Q4_0` dot-product kernels, multi-threaded row dispatch, profiling & thread scaling. |
| **M4 — Final demonstration** | `m4-demo` | `#5` (Open) | **NOT STARTED** | Production CLI, benchmark harness, 4-laptop benchmark report, head-to-head `llama.cpp` comparison. |

---

## Detailed Milestone Breakdown & Acceptance Criteria

### M0 — Repository Foundation (`m0-foundation`) — COMPLETE
- **Dependencies:** None.
- **Acceptance Criteria:**
  - [x] Local Git repository initialized at `~/Developer/inference-engine` and pushed to `https://github.com/shreyaas5/inference-engine`.
  - [x] Branch hierarchy created (`main`, `integration`, `feature/model-loader`, `feature/quantisation-simd`, `feature/transformer`, `feature/runtime-tokenizer`).
  - [x] Public C++17 headers (`include/inference_engine/*.hpp`) and CMake build configuration (`CMakeLists.txt`) compile with zero warnings.
  - [x] Core contracts implemented and unit-tested (`loader_validation`, `quant_packing`, `kernels_scalar`, `kernels_simd`, `kv_cache`, `tokenizer`, `golden_forward` — 7/7 passing).
  - [x] GitHub Actions CI workflow (`.github/workflows/ci.yml`) and context-transfer documentation system (`AGENTS.md`, `docs/context/*`) established.

---

### M1 — FP32 Scalar Inference (`m1-fp32`) — NEXT / IN PROGRESS
- **Dependencies:** M0 complete.
- **Assigned GitHub Issues:** `#1` (Owner: **S**), `#3` (Owner: **U**), `#4` (Owner: **P**).
- **Acceptance Criteria:**
  - [ ] `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py --dtype fp32` inspect and convert all 201 `TinyLlama-1.1B-Chat-v1.0` tensors (casting `BF16 -> FP32`) into a valid `.ie` v1 binary file.
  - [ ] `src/tokenizer/tokenizer.cpp` links SentencePiece C++ and achieves 100% token ID, BOS/EOS, byte-fallback, decode, and Zephyr chat template parity against Hugging Face (`tools/verify_tokenizer_parity.py`).
  - [ ] `src/transformer/transformer.cpp` binds all `.ie` tensors and executes the complete 22-layer FP32 scalar forward pass (embedding -> RMSNorm -> QKV -> RoPE ->KvCache `append` -> causal GQA -> `o_proj` -> residual -> RMSNorm -> SwiGLU -> `down_proj` -> residual -> `commit_token` -> final RMSNorm -> `lm_head`).
  - [ ] `tools/generate_golden_reference.py` exports FP32 Hugging Face intermediate activations and logits, and `tests/test_golden_forward.cpp` verifies multi-token prefill (`pos = 0..11`) and greedy decode continuation (`pos = 12+`) within calibrated empirical FP32 tolerances.

---

### M2 — Q4_0 Quantisation (`m2-q4`)
- **Dependencies:** M1 FP32 scalar pipeline verified against Hugging Face golden reference; Issue `#2` (Owner: **I**).
- **Acceptance Criteria:**
  - [ ] `tools/convert_to_inference_engine.py --dtype q4_0` quantises 2D linear weight matrices into 18-byte `BlockQ4_0` records while keeping 1D RMSNorm weights in `FP32`.
  - [ ] `Q4_0` quantiser/dequantiser validated against an independent trusted reference implementation (`tests/test_quant_packing.cpp`).
  - [ ] `TransformerEngine` supports mixed-precision `.ie` models (`FP32` + `Q4_0`) using `scalar::matmul_q4_0_f32` / `dispatch_matmul_q4_0_f32`.
  - [ ] End-to-end `Q4_0` inference runs within P's 8 GB RAM constraint and passes logit RMSE / cosine similarity / greedy continuation checks against the M1 FP32 baseline.

---

### M3 — CPU Optimisation (`m3-optimised`)
- **Dependencies:** M2 Q4_0 correctness verified.
- **Acceptance Criteria:**
  - [ ] `src/kernels/neon/kernels_neon.cpp` implements vectorized ARM64 NEON `dot_q4_0_f32` (replacing the scalar fallback).
  - [ ] `src/kernels/avx2/kernels_avx2.cpp` implements vectorized x86-64 AVX2+FMA `dot_q4_0_f32` (replacing the scalar fallback).
  - [ ] `tests/test_kernels_simd.cpp` verifies numerical parity between `scalar`, `neon`, and `avx2` `Q4_0` matrix-vector multiplication.
  - [ ] `ThreadPool` parallelizes matrix-vector projections across output rows with measured thread-scaling efficiency across `1, 2, 4, 8` threads.

---

### M4 — Final Demonstration (`m4-demo`)
- **Dependencies:** M3 SIMD and multi-threading verified.
- **Acceptance Criteria:**
  - [ ] `src/app/main.cpp` provides full CLI generation (`--model`, `--tokenizer`, `--prompt`, `--chat`, `--threads`, `--max-tokens`, `--temp`, `--top-p`, `--benchmark`).
  - [ ] Standardized CPU-only benchmarks executed across all four team laptops (S: Apple M4, I: Ryzen 7 7745HX, U: Core Ultra 7 155H, P: Core i5-10210U) recording load time, TTFT, prefill tok/s, decode tok/s, peak RSS, and thread scaling.
  - [ ] Controlled head-to-head comparison against `llama.cpp` (`--n-gpu-layers 0`, identical `Q4_0` model, prompts, and thread counts) documented in `docs/BENCHMARKING.md`.

