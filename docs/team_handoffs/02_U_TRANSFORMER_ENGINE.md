# `02_U_TRANSFORMER_ENGINE.md` — Member U Module Handoff

**Owner:** Member U (Scalar Reference Kernels, `KvCache`, 16-Step `TransformerEngine` `FP32` & `Q4_0` Execution, and Golden Reference Tooling)
**Consolidated Branch:** `integration-ps08-team-sync-2026-10-31`
**Merged Feature Branches:**
- `origin/feature/transformer` (`5115467`) — M1 scalar kernels, causal GQA, `KvCache`, 16-step `FP32` `TransformerEngine`, and `tools/generate_golden_reference.py`
- `origin/feature/transformer-bindings` (`bce66ae`) — Consolidated `TransformerWeightBindings` with `ModelLoader::bind_transformer_weights`
- `origin/feature/transformer-q4` (`ce1026a`) — `Q4_0` + mixed-precision `TransformerEngine::forward_token()`, `dispatch_matmul_*` + `ThreadPool` wiring, on-the-fly single-row `Q4_0` embedding dequantisation, and `tools/verify_transformer_q4.py`

---

## 1. Owned Files & Responsibilities

| File Path | Role | Status |
|---|---|---|
| [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp), [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp) | Scalar reference kernels: `rmsnorm_f32`, `rope_inplace_f32`, `dot_f32`, `dot_q4_0_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, `gqa_attention_f32` | `REFERENCE VERIFIED (NumPy)` |
| [`include/inference_engine/kv_cache.hpp`](../../include/inference_engine/kv_cache.hpp), [`src/core/kv_cache.cpp`](../../src/core/kv_cache.cpp) | Pre-allocated contiguous `FP32` key/value cache (`[num_layers, max_seq_len, num_kv_heads * head_dim]`) with explicit `append()` vs `commit_token()` separation | `TESTED (8/8 CTest)` |
| [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp), [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp) | `TransformerEngine::init`, `reset_cache`, `forward_token`, `matmul` (`FP32` and `Q4_0` + `ThreadPool` dispatch), and pre-allocated workspace buffers | `REFERENCE VERIFIED (NumPy)` |
| [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py) | Offline `"GOLD"` v1 binary fixture generator (`--self-test` + Hugging Face PyTorch `FP32` mode) | `TESTED (SYNTHETIC)` |
| [`tools/verify_transformer_q4.py`](../../tools/verify_transformer_q4.py) | Cross-language independent Python NumPy vs C++ `Q4_0` multi-token forward parity verifier | `REFERENCE VERIFIED` |
| [`tests/test_kernels_scalar.cpp`](../../tests/test_kernels_scalar.cpp), [`tests/test_kv_cache.cpp`](../../tests/test_kv_cache.cpp), [`tests/test_golden_forward.cpp`](../../tests/test_golden_forward.cpp) | Unit and golden forward parity test suites | `TESTED (8/8 CTest)` |

---

## 2. Frozen Technical Contracts Owned by U

### 2.1 16-Step `TransformerEngine::forward_token(token_id, pos, logits)` Contract
1. **Input Validation:** Verifies `initialized_`, `logits != nullptr`, `token_id in [0, vocab_size)`, `pos < max_seq_len`, and `pos == kv_cache_.seq_len()`.
2. **Embedding Lookup (Step 1):**
   - If `bindings_.tok_embeddings->dtype() == DataType::FP32`, copies row `token_id` (`hidden_dim = 2048` floats) into `x_`.
   - If `bindings_.tok_embeddings->dtype() == DataType::Q4_0`, dequantises only row `token_id` (`2048 / 32 = 64` `BlockQ4_0` blocks) directly into `x_` via `dequantize_row_q4_0(row_blocks, x_.data(), hidden_dim)` — zero full-matrix dequantisation buffer allocation.
3. **Per-Layer Decoder Loop (`layer_idx = 0 .. num_layers - 1`):**
   - **Pre-Attention RMSNorm:** `scalar::rmsnorm_f32(x_, attn_norm, xb_, hidden_dim, rms_norm_eps)`
   - **QKV Projections:** `matmul(wq, xb_, q_, q_dim, hidden_dim)`, `matmul(wk, xb_, k_, kv_dim, hidden_dim)`, `matmul(wv, xb_, v_, kv_dim, hidden_dim)`
   - **RoPE (`rotate_half` pairing `i` and `i + 32`):** `scalar::rope_inplace_f32(q_, pos, 32, 64, 10000.0f)` and `scalar::rope_inplace_f32(k_, pos, 4, 64, 10000.0f)`
   - **KV Cache Append:** `kv_cache_.append(layer_idx, pos, k_.data(), v_.data())` (writes K/V for `layer_idx` at `pos` **without** incrementing `seq_len()`)
   - **Causal GQA (`8` query heads per KV head `h_kv = h_q / 8`):** `scalar::gqa_attention_f32(q_, layer_k_cache, layer_v_cache, att_out_, att_scores_, pos, 32, 4, 64)` attending over `t = 0 .. pos`
   - **Output Projection & Residual Add:** `matmul(wo, att_out_, xb_, hidden_dim, q_dim)`; `x_[i] += xb_[i]`
   - **Pre-MLP RMSNorm:** `scalar::rmsnorm_f32(x_, ffn_norm, xb_, hidden_dim, rms_norm_eps)`
   - **SwiGLU MLP Projections:** `matmul(w_gate, xb_, gate_, intermediate_dim, hidden_dim)`, `matmul(w_up, xb_, up_, intermediate_dim, hidden_dim)`, `scalar::swiglu_f32(gate_, up_, hb_, intermediate_dim)`, `matmul(w_down, hb_, xb_, hidden_dim, intermediate_dim)`; `x_[i] += xb_[i]`
4. **KV Cache Commit:** After all `num_layers` complete, calls `kv_cache_.commit_token()` exactly once to advance `seq_len()` from `pos` to `pos + 1`.
5. **Final RMSNorm & Classifier (`lm_head`):**
   - `scalar::rmsnorm_f32(x_, final_norm, xb_, hidden_dim, rms_norm_eps)`
   - `matmul(lm_head, xb_, logits, vocab_size, hidden_dim)` (automatically uses `FP32` or `Q4_0` dispatch according to `bindings_.lm_head->dtype()`).

### 2.2 Numerical Parity Evidence
- **`FP32` Forward Parity (`tests/test_golden_forward.cpp`):** Maximum absolute logit difference between C++ `TransformerEngine` and independent Python NumPy `FP32` reference across prefill and decode steps is `6.55651e-07` (`<= 1e-5` tolerance).
- **`Q4_0` Forward Parity (`tools/verify_transformer_q4.py` & `tests/test_golden_forward.cpp`):** C++ `Q4_0` logits match an independent Python NumPy `Q4_0` implementation to 5 decimal places across 3 prefill tokens and 4 autoregressive decode tokens, and multi-threaded `ThreadPool` + SIMD execution matches single-threaded scalar execution within `<= 1e-4`.

---

## 3. Ready-to-Paste Antigravity Continuation Prompt for Member U

```markdown
You are acting as Member U (Scalar Reference Kernels, KV Cache, Transformer Forward Pass & Golden Validation) for `inference-engine` (BIOS V2 — PS08) in `/Users/shreyaas/Developer/inference-engine`.

Before making any changes, read:
1. `AGENTS.md`
2. `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`
3. `docs/team_handoffs/02_U_TRANSFORMER_ENGINE.md`
4. `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`

Current branch: `integration-ps08-team-sync-2026-10-31` (all 8/8 CTest suites and `tools/verify_transformer_q4.py` pass).

Your next engineering tasks:
1. Once `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` is downloaded and converted to `models/tinyllama-1.1b-fp32.ie` and `models/tinyllama-1.1b-q4_0.ie`, generate a real 22-layer golden reference fixture using `tools/generate_golden_reference.py`:
   `python3 tools/generate_golden_reference.py --checkpoint models/TinyLlama-1.1B-Chat-v1.0 --output models/golden_tinyllama_fp32.bin`
2. Run a C++ golden comparison against `models/golden_tinyllama_fp32.bin` to confirm that `rope_inplace_f32` (`rotate_half` convention, `KI-006`) and all 22 layers match Hugging Face PyTorch `FP32` logits (`max_abs_err <= 1e-3`, top-1 argmax token match).
3. Measure `Q4_0` vs `FP32` top-1 logit agreement and KL divergence / cosine similarity on the real 22-layer checkpoint.
```
