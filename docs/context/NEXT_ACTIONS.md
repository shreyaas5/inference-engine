# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

**Last Updated:** 2026-10-31  
**Current Phase:** Milestone M1 (`m1-fp32`) Kickoff

---

## Priority 0 — Immediate Repository Hygiene (Before Writing M1 Code)

- [ ] **0.1 Commit Context-Transfer System & Clean Working Tree:**
  - Discard the IDE's 1-line EOF newline diffs on existing code files (`git checkout -- CMakeLists.txt .gitignore .github/ fixtures/ include/ src/ tests/ tools/`), stage the context-transfer documentation (`AGENTS.md`, `PROJECT_CONTEXT.md`, `CURRENT_STATUS.md`, `SESSION_HANDOFF.md`, `DECISIONS.md`, `ROADMAP.md`, `TEAM_STATUS.md`, `docs/DECISION_LOG.md`, `docs/context/*.md`), and commit when approved by the user.
- [ ] **0.2 Check Out Target Feature Branch:**
  - Never work on `main`. Check out `feature/model-loader` (S), `feature/transformer` (U), `feature/quantisation-simd` (I), or `feature/runtime-tokenizer` (P).

---

## Priority 1 — FP32 Checkpoint Inspection, Conversion & Loading (Owner: S, Issue `#1`)

- [ ] **1.1 Set Up Offline Python Environment:**
  - Install `safetensors`, `transformers`, and `sentencepiece` inside the local Conda environment (`/Users/shreyaas/cwq/miniconda3`, which already has `torch 2.8.0` and `numpy 2.2.6`).
- [ ] **1.2 Complete `tools/inspect_checkpoint.py`:**
  - Parse `.safetensors` headers directly to print all 201 tensor names, dtypes (`BF16`), shapes, and byte sizes without loading full multi-gigabyte weights into memory at once.
- [ ] **1.3 Implement Full `FP32` Conversion in `tools/convert_to_inference_engine.py`:**
  - Read `TinyLlama-1.1B-Chat-v1.0` `.safetensors`, cast `BF16 -> FP32`, align each payload to 64 bytes (`align_up(offset, 64)`), write the `.ie` v1 binary file, and verify that `ModelLoader::load_from_file()` maps all 201 tensors cleanly.

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

- [ ] **4.1 Bind `.ie` Tensor Views in `TransformerEngine::init()`:**
  - Resolve and validate pointers/views for `model.embed_tokens.weight`, all 22 layers' 9 tensors (`input_layernorm`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `post_attention_layernorm`, `gate_proj`, `up_proj`, `down_proj`), `model.norm.weight`, and `lm_head.weight`.
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

