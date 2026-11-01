# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

**Last Updated:** 2026-11-01
**Current Phase:** Milestone M1 (`m1-fp32`); M7 validation is blocked by the local CMake/C++ toolchain, on `feature/runtime-tokenizer` at `eaaeceb`

---

## Priority 0 — Immediate Repository Hygiene (Completed)

- [x] **0.1 Commit Context-Transfer System & Clean Working Tree.**
- [x] **0.2 Merge `feature/model-loader` and `feature/transformer` into `main` (`177fe49`).**
- [x] **0.3 Merge `feature/quantisation-simd` into `integration` (`967f072`).**

---

## Priority 1 — FP32 Checkpoint Inspection, Conversion & Loading (Owner: S, Issue `#1`)

- [x] **1.1 Zero-Dependency & NumPy-Accelerated Python Converter Setup.**
- [x] **1.2 Complete `tools/inspect_checkpoint.py` & Verify Real TinyLlama Tensor Inventory (`201/201` tensors verified).**
- [x] **1.3 Implement Streaming `FP32` Conversion (`tools/convert_to_inference_engine.py`) & C++ Tensor Binding.**
- [ ] **1.4 Download Real 2.20 GB `model.safetensors` Payload & Convert Full 4.40 GB `tinyllama-1.1b-fp32.ie`.**

---

## Priority 2 — Scalar FP32 Kernels & KV Cache Correctness (Owner: U, Issue `#3` — Completed)

- [x] **2.1 Core Scalar Operations (`rmsnorm_f32`, `rope_inplace_f32`, `matmul_f32`, `swiglu_f32`, `softmax_f32`).**
- [x] **2.2 KV Cache Storage & Explicit Commit (`KvCache::append` vs `KvCache::commit_token`).**
- [x] **2.3 Implement Causal Grouped Query Attention (GQA) Kernel (`scalar::gqa_attention_f32`).**

---

## Priority 3 — Tokenizer Parity & SentencePiece Integration (Owner: P, Issue `#4` — Next Active Blocker)

- [x] **3.1 TinyLlama Zephyr Chat Template (`Tokenizer::format_chat_prompt`).**
- [ ] **3.2 Link SentencePiece C++ in `CMakeLists.txt` & `src/tokenizer/tokenizer.cpp` (implemented in working tree; build/test pending):**
  - CMake discovers SentencePiece >=0.2.0 through pkg-config or fetches a pinned upstream revision. The tokenizer validates TinyLlama's 32000-piece vocabulary and special-token IDs and handles model special tokens explicitly.
- [ ] **3.3 Verify Exact Tokenizer Parity (`tools/verify_tokenizer_parity.py`; harness implemented, full run pending):**
  - Compare IDs, BOS/EOS, whitespace, byte fallback/Unicode, decode output, and chat template against Hugging Face using the pinned `tokenizer.model` and `tokenizer_config.json`.
- [ ] **3.4 Verify Greedy, Temperature, Top-p, Seed Reproducibility, and Invalid Logits:**
  - Focused MinGW sampler test passed; full CTest build remains pending.

---

## Priority 4 — Full Multi-Layer TinyLlama Forward Pass (Owner: U, Issue `#3`)

- [x] **4.1 Implement Pre-Allocated Activation Workspace & `TransformerEngine::forward_token()`:**
  - *Completed in commit `5115467` and merged to `main`.*
- [x] **4.2 Synthetic Small-Model End-to-End & Independent NumPy Reference Test (`max_diff = 6.56e-7`).**
- [ ] **4.3 Unify `TransformerEngine::init()` Tensor Binding with `ModelLoader::bind_transformer_weights()` (`KI-010`).**

---

## Priority 5 — Real Hugging Face Golden Reference Comparison (Owners: U & S, Issue `#3`)

- [x] **5.1 Implement `tools/generate_golden_reference.py` Binary Fixture Packer & Runner.**
- [ ] **5.2 Fix Per-Step Decode Logit Indexing & Compare `gen_ref_logits` in `test_golden_forward.cpp` (`KI-011`).**
- [ ] **5.3 Execute Full 22-Layer Golden Validation Against Real `TinyLlama-1.1B-Chat-v1.0` Weights.**

---

## Priority 6 — `Q4_0` Conversion & SIMD Integration (Owners: I & S, Issue `#2`, Milestone M2/M3)

- [x] **6.1 Verify `quantize_row_q4_0` Against Upstream GGML Reference (`KI-007` — Merged on `integration`).**
- [x] **6.2 Vectorized ARM64 NEON Q4_0 & FP32 Kernels (`neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32` — Merged on `integration`).**
- [x] **6.3 Vectorized x86-64 AVX2/FMA Q4_0 & FP32 Kernels and CPUID Detection (`avx2::dot_q4_0_f32` — Merged on `integration`).**
- [x] **6.4 Multithreaded ThreadPool Task Chunking Optimization (`ThreadPool::parallel_for` — Merged on `integration`).**
- [ ] **6.5 Run Native Verification of AVX2/FMA Kernels on x86-64 Hardware / CI (`KI-013`).**
- [ ] **6.6 Add `--dtype q4_0` Export to `tools/convert_to_inference_engine.py`.**
- [ ] **6.7 Wire `dispatch_matmul_q4_0_f32` and `ThreadPool` into `TransformerEngine::forward_token()` (`KI-012`).**

---

## Priority 7 — CLI, Four-Laptop Benchmarking & `llama.cpp` Comparison (Owner: P & All, Milestone M4)

- [ ] **7.1 Validate the FP32 generation CLI and `--benchmark` harness (`src/app/main.cpp` implementation is in the working tree; build/real-model validation pending; current CMake install is incomplete and MinGW 6.3 lacks `<string_view>`).**
- [ ] **7.2 Run Standardized CPU Benchmarks on All 4 Laptops (S, I, U, P) and Compare Against `llama.cpp`.**
- [ ] **7.3 Complete M7 runtime validation: build and run all 9 CTest suites, full Hugging Face tokenizer parity, CLI errors/context limits, and KV-cache behavior.**
