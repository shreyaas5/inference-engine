# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

**Last Updated:** 2026-10-31T18:25:00+05:30  
**Current Phase:** Milestone M1 (`m1-fp32`) — Priority 0 & Priority 1 Completed on `feature/model-loader`

---

## Priority 0 — Immediate Repository Hygiene (Before Writing M1 Code)

- [x] **0.1 Commit Context-Transfer System & Clean Working Tree:**
  - *Completed in commit `60b103e` on `feature/model-loader` (restored the 52 EOF-only diffs and committed all 16 context-transfer documentation files).*
- [x] **0.2 Check Out Target Feature Branch:**
  - *Completed: active branch is `feature/model-loader`.*

---

## Priority 1 — FP32 Checkpoint Inspection, Conversion & Loading (Owner: S, Issue `#1`)

- [x] **1.1 Zero-Dependency & NumPy-Accelerated Python Converter Setup:**
  - *Completed: `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` run on both standard Python 3 and Conda `numpy` without requiring heavy third-party packages.*
- [x] **1.2 Complete `tools/inspect_checkpoint.py` & Verify Real TinyLlama Tensor Inventory:**
  - *Completed: verified all `201` tensors (`3 + 9 * 22`, `BF16`, `2,200,096,768` payload bytes) and `config.json` from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via `--inspect-remote-hf`.*
- [x] **1.3 Implement Streaming `FP32` Conversion (`tools/convert_to_inference_engine.py`) & C++ Tensor Binding (`ModelLoader::bind_transformer_weights`):**
  - *Completed: implemented two-pass streaming `BF16 -> FP32` conversion to 64-byte aligned `.ie` v1 files, added `TransformerWeightBindings` and `ModelLoader::bind_transformer_weights()`, and verified end-to-end Python conversion -> C++ `mmap` reload and bit-exact `FP32` preservation in `tests/test_loader_validation.cpp`.*

---

## Priority 2 — Scalar FP32 Kernels & KV Cache Correctness (Owner: U, Issue `#3`)

- [x] **2.1 Core Scalar Operations (`rmsnorm_f32`, `rope_inplace_f32`, `matmul_f32`, `swiglu_f32`, `softmax_f32`):**
  - *Completed & tested in M0 (`src/kernels/scalar/kernels_scalar.cpp`, `tests/test_kernels_scalar.cpp`).*
- [x] **2.2 KV Cache Storage & Explicit Commit (`KvCache::append` vs `KvCache::commit_token`):**
  - *Completed & tested in M0 (`src/core/kv_cache.cpp`, `tests/test_kv_cache.cpp`).*
- [ ] **2.3 Implement Causal Grouped Query Attention (GQA) Kernel:**
  - Implement `scalar::gqa_attention_f32` in `src/kernels/scalar/kernels_scalar.cpp` (and declare in `include/inference_engine/kernels.hpp`) supporting `32` query heads, `4` KV heads (`8` Q heads per KV head), `head_dim = 64`, scale $1/\sqrt{64} = 0.125$, causal masking over `0..pos`, and unit-test it in `tests/test_kernels_scalar.cpp`.

---

## Priority 3 — Tokenizer Parity & SentencePiece Integration (Owner: P, Issue `#4`)

- [x] **3.1 TinyLlama Zephyr Chat Template (`Tokenizer::format_chat_prompt`):**
  - *Completed & tested in M0 (`src/tokenizer/tokenizer.cpp`, `tests/test_tokenizer.cpp`).*
- [ ] **3.2 Link SentencePiece C++ in `CMakeLists.txt` & `src/tokenizer/tokenizer.cpp`:**
  - Install/discover `sentencepiece` C++ library (`find_package` / `pkg-config` / Homebrew `/opt/homebrew`), replace the M0 `byte + 3` fallback in `Tokenizer::encode()` and `Tokenizer::decode()` with `sentencepiece::SentencePieceProcessor`, and keep unit tests green.
- [ ] **3.3 Verify Exact Tokenizer Parity (`tools/verify_tokenizer_parity.py`):**
  - Compare token IDs, BOS/EOS handling, whitespace (`▁`), byte fallback (`<0x..>`), and chat template rendering against Hugging Face `AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`.

---

## Priority 4 — Full 22-Layer TinyLlama Forward Pass (Owner: U, Issue `#3`)

- [ ] **4.1 Call `ModelLoader::bind_transformer_weights()` in `TransformerEngine::init()`:**
  - Use the new `TransformerWeightBindings` API from `include/inference_engine/loader.hpp` to bind `embed_tokens`, all 22 layers' 9 tensors, `final_norm`, and `lm_head`.
- [ ] **4.2 Implement Pre-Allocated Activation Workspace & `TransformerEngine::forward_token()`:**
  - Allocate reusable `FP32` activation buffers in `init()` (zero heap allocations during `forward_token()`) and implement the full 22-layer forward pass in `src/transformer/transformer.cpp`.
- [ ] **4.3 Synthetic Small-Model End-to-End Test:**
  - Construct a tiny 2-layer synthetic `.ie` model in `tests/test_golden_forward.cpp` with known non-zero weights and verify deterministic non-zero logit output in CI without requiring the 4.4 GB checkpoint.

---

## Priority 5 — Hugging Face Golden Reference Comparison (Owners: U & S, Issue `#3`)

- [ ] **5.1 Implement `tools/generate_golden_reference.py`:**
  - Load `TinyLlama-1.1B-Chat-v1.0` in `torch.float32`, run a 12-token prompt (`pos = 0..11`) plus greedy continuation (`pos = 12..27`), and export layer-0/11/21 activations, final logits, and greedy token IDs.
- [ ] **5.2 Verify Multi-Token & Greedy Continuation Parity:**
  - Confirm RoPE pairing convention (`KI-006`), calibrate empirical FP32 tolerances across all 22 layers, and verify exact greedy token sequence parity.

---

## Priority 6 — `Q4_0` Conversion & Mixed-Precision Validation (Owners: I & S, Issue `#2`)

- [ ] **6.1 Cross-Check `quantize_row_q4_0` Against Independent Reference (`KI-007`).**
- [ ] **6.2 Add `--dtype q4_0` Export to `tools/convert_to_inference_engine.py`** (keeping 1D `layernorm` weights in `FP32` and quantising 2D weights to `Q4_0`).
- [ ] **6.3 Enable Mixed-Precision (`FP32` + `Q4_0`) Dispatch in `TransformerEngine::forward_token()`** and measure `Q4_0` vs `FP32` logit RMSE/cosine similarity.

---

## Priority 7 — SIMD (`NEON` / `AVX2`) & Multithreading Optimisation (Owner: I, Milestone M3)

- [ ] **7.1 Implement Vectorized `neon::dot_q4_0_f32` (`src/kernels/neon/kernels_neon.cpp`).**
- [ ] **7.2 Implement Vectorized `avx2::dot_q4_0_f32` (`src/kernels/avx2/kernels_avx2.cpp`).**
- [ ] **7.3 Wire `ThreadPool` into `TransformerEngine` Projections & Profile Thread Scaling (`1, 2, 4, 8` threads).**

---

## Priority 8 — CLI, Four-Laptop Benchmarking & `llama.cpp` Comparison (Owner: P & All, Milestone M4)

- [ ] **8.1 Complete Interactive/Batch CLI & `--benchmark` Harness in `src/app/main.cpp`.**
- [ ] **8.2 Run Standardized CPU Benchmarks on All 4 Laptops (S, I, U, P) and Compare Against `llama.cpp`.**
