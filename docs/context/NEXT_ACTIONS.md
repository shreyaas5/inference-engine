# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

**Last Updated:** 2026-10-31T23:05:00+05:30
**Current Phase:** Consolidated Team Integration (`integration-ps08-team-sync-2026-10-31`) — Preparing for Real 2.20 GB Checkpoint Execution & M4 Benchmarking

---

## Priority 0 — Immediate Repository Hygiene & Branch Integration (Completed)

- [x] **0.1 Commit Context-Transfer System & Clean Working Tree.**
- [x] **0.2 Merge `feature/model-loader` and `feature/transformer` into `main` (`177fe49`).**
- [x] **0.3 Merge `feature/quantisation-simd` into `integration` (`783b8f4`).**
- [x] **0.4 Consolidate S (`4515579`), U (`ce1026a`), I (`58c604d`), and P (`ce769c8`) onto `integration-ps08-team-sync-2026-10-31` and Verify `8/8` CTest Suites.**

---

## Priority 1 — FP32 & Q4_0 Checkpoint Inspection, Conversion & Loading (Owner: S, Issue `#1`)

- [x] **1.1 Zero-Dependency & NumPy-Accelerated Python Converter Setup.**
- [x] **1.2 Complete `tools/inspect_checkpoint.py` & Verify Real TinyLlama Tensor Inventory (`201/201` tensors verified) + Evaluate Real `Q4_0` Error across 9 projection roles (`--eval-real-quant-error`).**
- [x] **1.3 Implement Streaming `FP32` and `Q4_0` Conversion (`tools/convert_to_inference_engine.py`) & C++ Tensor Binding.**
- [ ] **1.4 Download Full 2.20 GB `model.safetensors` Payload (`curl -C -`) & Convert Full `tinyllama-1.1b-fp32.ie` (`4.40 GB`) and `tinyllama-1.1b-q4_0.ie` (`619.11 MB`):**
  - Local metadata (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model` 499 KB) are already downloaded in `models/TinyLlama-1.1B-Chat-v1.0/`.

---

## Priority 2 — Scalar FP32 Kernels & KV Cache Correctness (Owner: U, Issue `#3` — Completed)

- [x] **2.1 Core Scalar Operations (`rmsnorm_f32`, `rope_inplace_f32`, `matmul_f32`, `swiglu_f32`, `softmax_f32`).**
- [x] **2.2 KV Cache Storage & Explicit Commit (`KvCache::append` vs `KvCache::commit_token`).**
- [x] **2.3 Implement Causal Grouped Query Attention (GQA) Kernel (`scalar::gqa_attention_f32`).**

---

## Priority 3 — Tokenizer Parity, Sampler & SentencePiece Integration (Owner: P, Issue `#4`)

- [x] **3.1 TinyLlama Zephyr Chat Template (`Tokenizer::format_chat_prompt`).**
- [x] **3.2 Link SentencePiece C++ in `CMakeLists.txt` & `src/tokenizer/tokenizer.cpp` and Verify Model-Backed `tokenizer` CTest (`KI-003` resolved).**
- [x] **3.3 Implement & Verify Greedy, Temperature, Top-p, Seed Reproducibility, and Invalid Logit Filtering (`src/sampler/sampler.cpp`, `tests/test_sampler.cpp`).**
- [ ] **3.4 Run Live Hugging Face `transformers` vs C++ Parity (`tools/verify_tokenizer_parity.py --model-dir models/TinyLlama-1.1B-Chat-v1.0 --engine-bin ./build/inference-engine`):**
  - Offline `--self-test` and C++ SentencePiece `tokenizer` CTest pass; full Python `transformers` cross-check requires a Python venv with `transformers` and `sentencepiece` installed (also configured in `.github/workflows/ci.yml`).

---

## Priority 4 — Full Multi-Layer TinyLlama Forward Pass (`FP32` & `Q4_0`) (Owner: U, Issue `#3` — Completed)

- [x] **4.1 Implement Pre-Allocated Activation Workspace & `TransformerEngine::forward_token()` (`5115467`).**
- [x] **4.2 Synthetic Small-Model End-to-End & Independent NumPy Reference Test (`max_diff = 6.56e-7`).**
- [x] **4.3 Unify `TransformerEngine::init()` Tensor Binding with `ModelLoader::bind_transformer_weights()` (`KI-010` — Completed in `6dd4fed`).**
- [x] **4.4 Wire `dispatch_matmul_f32`, `dispatch_matmul_q4_0_f32`, `dequantize_row_q4_0`, and `ThreadPool` into `TransformerEngine::forward_token()` (`KI-012` — Completed in `ce1026a` and verified via `tools/verify_transformer_q4.py`).**

---

## Priority 5 — Real Hugging Face Golden Reference Comparison (Owners: U & S, Issue `#3`)

- [x] **5.1 Implement `tools/generate_golden_reference.py` Binary Fixture Packer & Runner.**
- [x] **5.2 Fix Per-Step Decode Logit Indexing & Compare `gen_ref_logits` in `test_golden_forward.cpp` (`KI-011` — Completed in `6dd4fed`).**
- [ ] **5.3 Execute Full 22-Layer Golden Validation Against Real `TinyLlama-1.1B-Chat-v1.0` Weights (`FP32` and `Q4_0`).**

---

## Priority 6 — `Q4_0` Conversion & SIMD Integration (Owners: I, S & U, Issue `#2`, Milestone M2/M3 — Completed)

- [x] **6.1 Verify `quantize_row_q4_0` Against Upstream GGML Reference (`KI-007` — Merged on `integration`).**
- [x] **6.2 Vectorized ARM64 NEON Q4_0 & FP32 Kernels (`neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32` — Verified on Apple M4 ARM64).**
- [x] **6.3 Vectorized x86-64 AVX2/FMA Q4_0 & FP32 Kernels and CPUID Detection (`avx2::dot_q4_0_f32`).**
- [x] **6.4 Multithreaded ThreadPool Task Chunking (`ADR-009`) & Exception Propagation (`ADR-010`).**
- [x] **6.5 Run Native Verification of AVX2/FMA Kernels on x86-64 Hardware (`KI-013` — Verified on AMD Ryzen 7 7745HX in `58c604d`).**
- [x] **6.6 Add `--dtype q4_0` Export, Mixed-Precision Loader Validation & `TensorView::data_q4_0()` (`cb176b3`).**
- [x] **6.7 Wire `dispatch_matmul_q4_0_f32` and `ThreadPool` into `TransformerEngine::forward_token()` (`ce1026a`).**

---

## Priority 7 — CLI, Four-Laptop Benchmarking & `llama.cpp` Comparison (Owners: P & All, Milestone M4)

- [x] **7.1 Implement & Build CLI Generation Loop and `--benchmark` Harness (`src/app/main.cpp` — Verified with real `tokenizer.model` and synthetic `.ie` models).**
- [ ] **7.2 Run Real 22-Layer TinyLlama CLI Text Generation (`FP32` and `Q4_0`) Once `model.safetensors` Is Converted.**
- [ ] **7.3 Run Standardized CPU Benchmarks on All 4 Laptops (S, I, U, P) and Compare Against `llama.cpp`.**
