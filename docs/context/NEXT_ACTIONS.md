# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

**Last Updated:** 2026-10-31T18:46:00+05:30  
**Current Phase:** Milestone M1 (`m1-fp32`) — Priorities 0, 1, 2 & 4 Merged into `integration` (Synthetic Validation Complete; Real Checkpoint & Tokenizer Pending)

---

## Priority 0 — Immediate Repository Hygiene (Completed)

- [x] **0.1 Commit Context-Transfer System & Clean Working Tree:**
  - *Completed in commit `60b103e` on `feature/model-loader`.*
- [x] **0.2 Merge `feature/model-loader` (`aa40368`) and `feature/transformer` (`5115467`) into `integration`:**
  - *Completed in merge commit `1f4e2a9` on `integration`.*

---

## Priority 1 — FP32 Checkpoint Inspection, Conversion & Loading (Owner: S, Issue `#1`)

- [x] **1.1 Zero-Dependency & NumPy-Accelerated Python Converter Setup.**
- [x] **1.2 Complete `tools/inspect_checkpoint.py` & Verify Real TinyLlama Tensor Inventory (`201/201` tensors verified via `--inspect-remote-hf`).**
- [x] **1.3 Implement Streaming `FP32` Conversion (`tools/convert_to_inference_engine.py`) & C++ Tensor Binding (`ModelLoader::bind_transformer_weights`).**
- [ ] **1.4 Download Real 2.20 GB `model.safetensors` Payload & Convert Full 4.40 GB `tinyllama-1.1b-fp32.ie`:**
  - Run `convert_to_inference_engine.py` on the real `TinyLlama-1.1B-Chat-v1.0` checkpoint and verify `mmap` reload with `INFERENCE_ENGINE_TEST_IE_PATH`.

---

## Priority 2 — Scalar FP32 Kernels & KV Cache Correctness (Owner: U, Issue `#3` — Completed)

- [x] **2.1 Core Scalar Operations (`rmsnorm_f32`, `rope_inplace_f32`, `matmul_f32`, `swiglu_f32`, `softmax_f32`).**
- [x] **2.2 KV Cache Storage & Explicit Commit (`KvCache::append` vs `KvCache::commit_token`).**
- [x] **2.3 Implement Causal Grouped Query Attention (GQA) Kernel (`scalar::gqa_attention_f32`):**
  - *Completed in commit `5115467` and verified in `tests/test_kernels_scalar.cpp` and `tests/test_golden_forward.cpp`.*

---

## Priority 3 — Tokenizer Parity & SentencePiece Integration (Owner: P, Issue `#4` — Next Active Blocker)

- [x] **3.1 TinyLlama Zephyr Chat Template (`Tokenizer::format_chat_prompt`).**
- [ ] **3.2 Link SentencePiece C++ in `CMakeLists.txt` & `src/tokenizer/tokenizer.cpp`:**
  - Install/discover `sentencepiece` C++ library (`find_package` / `pkg-config` / Homebrew `/opt/homebrew`), replace the M0 `byte + 3` fallback in `Tokenizer::encode()` and `Tokenizer::decode()` with `sentencepiece::SentencePieceProcessor`, and keep unit tests green.
- [ ] **3.3 Verify Exact Tokenizer Parity (`tools/verify_tokenizer_parity.py`):**
  - Compare token IDs, BOS/EOS handling, whitespace (`▁`), byte fallback (`<0x..>`), and chat template rendering against Hugging Face `AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`.

---

## Priority 4 — Full Multi-Layer TinyLlama Forward Pass (Owner: U, Issue `#3`)

- [x] **4.1 Implement Pre-Allocated Activation Workspace & `TransformerEngine::forward_token()`:**
  - *Completed in commit `5115467` (`src/transformer/transformer.cpp`).*
- [x] **4.2 Synthetic Small-Model End-to-End & Independent NumPy Reference Test:**
  - *Completed in `tests/test_golden_forward.cpp` and verified against an independent Python NumPy `FP32` reference (`max_diff = 6.55651e-07` across prompt + 4 decode steps).*
- [ ] **4.3 Unify `TransformerEngine::init()` Tensor Binding with `ModelLoader::bind_transformer_weights()` (`KI-010`):**
  - Replace the file-local `bind_tensor()` / `const float*` `LayerWeights` in `TransformerEngine` with `TransformerWeightBindings` (`TensorView`) so full tensor-count validation and M2 `Q4_0` weights are supported without duplication.

---

## Priority 5 — Real Hugging Face Golden Reference Comparison (Owners: U & S, Issue `#3`)

- [x] **5.1 Implement `tools/generate_golden_reference.py` Binary Fixture Packer & Runner (`5115467`).**
- [ ] **5.2 Fix Per-Step Decode Logit Indexing & Compare `gen_ref_logits` in `test_golden_forward.cpp` (`KI-011`).**
- [ ] **5.3 Execute Full 22-Layer Golden Validation Against Real `TinyLlama-1.1B-Chat-v1.0` Weights:**
  - Generate `fixtures/golden/golden_reference.bin` and `fixtures/golden/model.ie` from the real checkpoint, confirm RoPE pairing convention (`KI-006`), and verify multi-token logit + greedy decode parity in `golden_forward`.

---

## Priority 6 — `Q4_0` Conversion & Mixed-Precision Validation (Owners: I & S, Issue `#2`, Milestone M2)

- [ ] **6.1 Cross-Check `quantize_row_q4_0` Against Independent Reference (`KI-007`).**
- [ ] **6.2 Add `--dtype q4_0` Export to `tools/convert_to_inference_engine.py`.**
- [ ] **6.3 Enable Mixed-Precision (`FP32` + `Q4_0`) Dispatch in `TransformerEngine::forward_token()`.**

---

## Priority 7 — SIMD (`NEON` / `AVX2`) & Multithreading Optimisation (Owner: I, Milestone M3)

- [ ] **7.1 Implement Vectorized `neon::dot_q4_0_f32` (`src/kernels/neon/kernels_neon.cpp`).**
- [ ] **7.2 Implement Vectorized `avx2::dot_q4_0_f32` (`src/kernels/avx2/kernels_avx2.cpp`).**
- [ ] **7.3 Wire `dispatch_matmul_*` & `ThreadPool` into `TransformerEngine::forward_token()` (`KI-012`).**

---

## Priority 8 — CLI, Four-Laptop Benchmarking & `llama.cpp` Comparison (Owner: P & All, Milestone M4)

- [ ] **8.1 Complete Interactive/Batch CLI & `--benchmark` Harness in `src/app/main.cpp`.**
- [ ] **8.2 Run Standardized CPU Benchmarks on All 4 Laptops (S, I, U, P) and Compare Against `llama.cpp`.**
