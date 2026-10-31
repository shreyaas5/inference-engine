# `02_U_TRANSFORMER_ENGINE.md` — Member U Handoff: Transformer Engine, Scalar Reference Kernels, `KvCache` & Full-Model Inference

> **Member:** **U** (Transformer Engine, Scalar Reference Kernels, `KvCache`, and Numerical Parity Owner)
> **Consolidated Baseline Branch:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Historical Feature Branches (Merged into `33d4bdc`):**
> - `origin/feature/transformer` / `origin/feature/transformer-m1-fp32` (`511546750ba3ac3283ddcfc8df89e1554c0f6fa8`)
> - `origin/feature/transformer-bindings` (`955bd914df93e8d13157a00303cee1addfdb93d0`, parent `bce66ae`)
> - `origin/feature/transformer-q4` (`ce1026a7ef90511425f3c258db1ae139dfc77dee` — commits `a06b0fc`, `26b6b48`, `9d6b6dc`, `ce1026a`, merged via `a687095`)
> **Active Unmerged Remote Feature Branch:**
> - `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d` — adds `target_compile_options(sentencepiece-static PRIVATE -include cstdint)` in `CMakeLists.txt` for GCC 16 compatibility)

---

## A. Role and Ownership

### 1. Primary Responsibilities
- **Scalar Numerical Reference Kernels (`VERIFIED`):** Own `src/kernels/scalar/kernels_scalar.cpp` as the mathematical source of truth for RMSNorm, in-place RoPE (`rotate_half`), causal Grouped-Query Attention (GQA), SwiGLU, vector primitives, `FP32` matmul/dot, and `Q4_0` matmul/dot.
- **Contiguous `FP32` Key-Value Cache (`VERIFIED`):** Own `KvCache` (`include/inference_engine/kv_cache.hpp`, `src/core/kv_cache.cpp`) and enforce the strict separation between `append()` and `commit_token()`.
- **Full Transformer Decoder Forward Pass (`VERIFIED`):** Own `TransformerEngine` (`include/inference_engine/transformer.hpp`, `src/transformer/transformer.cpp`) supporting pure `FP32`, pure `Q4_0`, and mixed-precision (`FP32` embeddings/`lm_head` + `Q4_0` layer projections) execution routed through `dispatch_matmul_*` and `ThreadPool`.
- **Golden Numerical Validation (`VERIFIED`):** Own `tools/generate_golden_reference.py`, `tools/verify_transformer_q4.py`, and the C++ test suites `kernels_scalar`, `kv_cache`, `golden_forward`, and `transformer_q4`.

### 2. Files Owned and Modified by U
- **Owned Public Headers:**
  - [`include/inference_engine/kv_cache.hpp`](../../include/inference_engine/kv_cache.hpp)
  - [`include/inference_engine/transformer.hpp`](../../include/inference_engine/transformer.hpp)
  - Co-owned declarations in [`include/inference_engine/kernels.hpp`](../../include/inference_engine/kernels.hpp) (`scalar::*` namespace).
- **Owned C++ Implementation:**
  - [`src/kernels/scalar/kernels_scalar.cpp`](../../src/kernels/scalar/kernels_scalar.cpp)
  - [`src/core/kv_cache.cpp`](../../src/core/kv_cache.cpp)
  - [`src/transformer/transformer.cpp`](../../src/transformer/transformer.cpp)
- **Owned Tests & Offline Verification Scripts:**
  - [`tests/test_kernels_scalar.cpp`](../../tests/test_kernels_scalar.cpp)
  - [`tests/test_kv_cache.cpp`](../../tests/test_kv_cache.cpp)
  - [`tests/test_golden_forward.cpp`](../../tests/test_golden_forward.cpp)
  - [`tests/test_transformer_q4.cpp`](../../tests/test_transformer_q4.cpp)
  - [`tools/generate_golden_reference.py`](../../tools/generate_golden_reference.py)
  - [`tools/verify_transformer_q4.py`](../../tools/verify_transformer_q4.py)

### 3. Ownership Boundaries (What U Must NOT Independently Modify)
- Do **not** modify `.ie` binary header/directory parsing in `src/loader/loader.cpp` without coordinating with **S**.
- Do **not** modify AVX2 (`src/kernels/avx2/`), NEON (`src/kernels/neon/`), or `ThreadPool` (`src/kernels/threading/`) implementations without coordinating with **I**.
- Do **not** modify SentencePiece tokenisation (`src/tokenizer/`), sampling (`src/sampler/`), or CLI flag contracts (`src/app/main.cpp`) without coordinating with **P**.

---

## B. Technical Background & Exact Implementation Details

### 1. Scalar Reference Kernels (`src/kernels/scalar/kernels_scalar.cpp`) (`VERIFIED`)
- **RMSNorm (`scalar::rmsnorm`):**
  $$\text{out}[i] = x[i] \cdot \frac{1}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1} x[j]^2 + \epsilon}} \cdot w[i] \quad (\epsilon = 10^{-5})$$
- **RoPE (`scalar::rope_inplace` — Hugging Face Llama `rotate_half` convention):**
  - Splits each head vector of length $\text{head\_dim} = 64$ into two contiguous 32-element halves: $x_0 = \text{head}[i]$ and $x_1 = \text{head}[i + 32]$ for $i \in [0, 31]$.
  - Frequency: $\theta_i = \text{pos} \times \text{rope\_theta}^{-2i / \text{head\_dim}} = \text{pos} \times 10000.0^{-2i / 64}$.
  - Update: $\text{head}[i] = x_0 \cos\theta_i - x_1 \sin\theta_i$, $\text{head}[i + 32] = x_1 \cos\theta_i + x_0 \sin\theta_i$.
  - Applied in-place to all `32` query heads (`q_buf_`) and all `4` key heads (`k_buf_`).
- **Causal Grouped-Query Attention (`scalar::attention_gqa`):**
  - `total_seq_len = pos + 1` (attends causally over cached positions `0 .. pos` inclusive).
  - GQA mapping: `group_size = num_query_heads / num_kv_heads = 32 / 4 = 8`; query head `h_q` maps to KV head `h_kv = h_q / 8`.
  - Scale factor: `scale = 1.0f / std::sqrt(64.0f) = 0.125f`.
  - Numerically stable softmax (`scalar::softmax_inplace`) subtracts $\max_{t \in [0, \text{pos}]} s_t$ before exponentiation.
- **SwiGLU MLP (`scalar::swiglu`):**
  $$\text{out}[i] = \frac{\text{gate}[i]}{1 + \exp(-\text{gate}[i])} \times \text{up}[i]$$

### 2. `KvCache` Layout and Commit Contract (`src/core/kv_cache.cpp`) (`VERIFIED`)
- Contiguous `FP32` storage for `k_cache_` and `v_cache_`, each indexed as `[layer_idx][pos][kv_head][dim]` with total elements per buffer `22 * 2048 * 4 * 64 = 11,534,336` floats (`22.54 MB` K + `22.54 MB` V = `45.09 MB` total).
- **`KvCache::append(layer_idx, pos, k_data, v_data)`:** Copies `num_kv_heads * head_dim` (`256`) floats into `layer_idx` at `pos`. **Never increments `seq_len_`.**
- **`KvCache::commit_token()` (`commit_positions(1)`):** Increments `seq_len_` by `1` **once per token** at the end of `TransformerEngine::forward_token()` after all layers have succeeded.
- **`KvCache::clear()`:** Resets `seq_len_ = 0` and zeroes both K and V buffers. (Note: `TransformerEngine` exposes `engine.kv_cache().clear()`, not a separate `reset_cache()` method.)

### 3. `TransformerEngine::forward_token` & Mixed-Precision Routing (`src/transformer/transformer.cpp`) (`VERIFIED`)
- **Initialization (`TransformerEngine::init`):**
  1. Validates `config` (`config.is_valid()`) and checks every tensor in `TransformerWeightBindings` (`embed_tokens`, `layers[0..L-1]`, `final_norm`, `lm_head`) for exact shape and supported `DataType` (`FP32` for 1D norms; `FP32` or `Q4_0` for `embed_tokens`, `lm_head`, and 2D layer projections).
  2. Resolves `SimdBackend` via `resolve_backend(backend)` and initializes `KvCache`.
  3. Pre-allocates all 11 scratch buffers (`x_`, `norm_buf_`, `q_buf_`, `k_buf_`, `v_buf_`, `attn_out_buf_`, `proj_out_buf_`, `gate_buf_`, `up_buf_`, `mlp_hidden_buf_`, `mlp_out_buf_`) so `forward_token` performs **zero heap allocations**.
- **Linear Projection Routing (`TransformerEngine::project_linear`):**
  ```cpp
  Status TransformerEngine::project_linear(const TensorView& weight, const float* input,
                                           float* output, size_t out_dim, size_t in_dim) const {
    if (weight.dtype() == DataType::FP32) {
      dispatch_matmul_f32(output, weight.data_f32(), input, out_dim, in_dim, backend_, &thread_pool_);
      return Status::Ok();
    }
    if (weight.dtype() == DataType::Q4_0) {
      dispatch_matmul_q4_0_f32(output, weight.data_q4_0(), input, out_dim, in_dim, backend_, &thread_pool_);
      return Status::Ok();
    }
    return Status(StatusCode::UnsupportedFormat, "Unsupported weight dtype in linear projection");
  }
  ```
- **Single-Row `Q4_0` Embedding Lookup:** When `weights_.embed_tokens.dtype() == DataType::Q4_0`, `forward_token` computes `row_blocks = weights_.embed_tokens.data_q4_0() + token_id * (hidden_dim / 32)` and calls `dequantize_row_q4_0(row_blocks, x_.data(), hidden_dim)` (`O(hidden_dim)` work).

---

## C. Completed Work, Commits, and Verification Evidence

| Deliverable | Commit(s) | Classification | Evidence |
| :--- | :--- | :--- | :--- |
| Scalar reference kernels (`rmsnorm`, `rope_inplace`, `attention_gqa`, `swiglu`, `matmul_f32`, `dot_f32`) | `5115467` (`origin/feature/transformer`) | `VERIFIED` / `TESTED` | `kernels_scalar` CTest (`7/7` passed, `MaxAbsDiff < 1e-6`) |
| Contiguous `KvCache` (`append` vs `commit_token`, bounds validation, 22-layer simulation) | `5115467` (`origin/feature/transformer`) | `VERIFIED` / `TESTED` | `kv_cache` CTest (`4/4` passed) |
| 16-step `FP32` `TransformerEngine::forward_token` + NumPy golden forward verification | `5115467`, `955bd91` | `SYNTHETICALLY VALIDATED` | `golden_forward` CTest (`3/3` passed, `MaxAbsDiff = 2.98e-7`) |
| `Q4_0` & mixed-precision `TransformerEngine`, `dispatch_matmul_*` + `ThreadPool` wiring, single-row `Q4_0` embedding lookup | `a06b0fc`, `26b6b48`, `9d6b6dc`, `ce1026a` (`origin/feature/transformer-q4`) | `VERIFIED` / `SYNTHETICALLY VALIDATED` | `transformer_q4` CTest (`7/7` passed) + `tools/verify_transformer_q4.py` (`cos_sim >= 0.996359`) |
| GCC 16 `-include cstdint` build fix for `sentencepiece-static` | `e64078c` (`origin/feature/transformer-runtime`, unmerged) | `VERIFIED` | Inspected via `git show e64078c84e7832deb0444f51ffad40c45154438d` |

---

## D. Current State: Precise Boundary Between Synthetic Validation and Pretrained Checkpoint Inference

- **What Has Been Verified (`SYNTHETICALLY VALIDATED`):**
  - All 16 steps of `TransformerEngine::forward_token()` across multi-layer, multi-token causal prefill and incremental decode have been verified in C++ (`golden_forward`, `transformer_q4`) and against independent Python NumPy references (`tools/verify_transformer_q4.py`) for:
    1. Pure `FP32` `.ie` models (`MaxAbsDiff = 2.98e-7` vs Python `FP32`).
    2. Pure `Q4_0` `.ie` models (`cos_sim = 0.996359` vs `FP32`; `MaxAbsDiff <= 2.38e-6` vs Python `Q4_0` reference).
    3. Mixed-precision `Q4_0` (`--keep-embeddings-fp32`) `.ie` models (`cos_sim = 0.997088` vs `FP32`).
    4. Scalar vs SIMD (`NEON` / `AVX2`) parity (`MaxAbsDiff <= 7.15e-7`) and 1-thread vs 4-thread `ThreadPool` determinism (`MaxAbsDiff == 0.0`).
- **What Has NOT Yet Been Executed (`BLOCKED` on 2.20 GB `model.safetensors` Download):**
  - Full 22-layer `TransformerEngine::forward_token()` execution against the real pretrained `TinyLlama-1.1B-Chat-v1.0` `.ie` weights (`models/tinyllama-fp32.ie` and `models/tinyllama-q4_0.ie`) and PyTorch golden reference fixtures (`tests/fixtures/golden/`).

---

## E. Dependencies on Other Team Members

- **On Member S:** Completion of the 2.20 GB `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` download and conversion into `models/tinyllama-fp32.ie`, `models/tinyllama-q4_0.ie`, and `models/tinyllama-q4_0-mixed.ie`, plus merging U's `e64078c` GCC 16 `<cstdint>` fix into the integration branch.
- **On Member I:** `dispatch_matmul_f32` and `dispatch_matmul_q4_0_f32` (already wired into `TransformerEngine::project_linear`). Once S merges I's `b05c4e4`, `TransformerEngine` will automatically benefit from AVX2 2-row register tiling and contiguous 2-row `ThreadPool` chunking without any code changes in `src/transformer/transformer.cpp`.
- **Downstream Consumer (Member P):** P's CLI (`src/app/main.cpp`) calls `engine.init(...)` and `engine.forward_token(token_id, pos, logits.data())`.

---

## F. Remaining Tasks (Priority Order)

### `P0` — Blocks the Entire Project
1. **Generate Real 22-Layer PyTorch Golden Reference Fixtures Once `model.safetensors` Download Completes:**
   - Run:
     ```bash
     python3 tools/generate_golden_reference.py --model-id models/TinyLlama-1.1B-Chat-v1.0 --output-dir tests/fixtures/golden
     ```
   - Verify that `tools/generate_golden_reference.py` casts `BF16` weights to `FP32` (`ADR-007`) and exports prompt token IDs, reference hidden states, and top-k logits.
2. **Execute Full 22-Layer Pretrained Checkpoint Forward Pass (`FP32`, Pure `Q4_0`, Mixed `Q4_0`):**
   - Run C++ forward pass against `models/tinyllama-fp32.ie`, `models/tinyllama-q4_0.ie`, and `models/tinyllama-q4_0-mixed.ie`.
   - Record exact `FP32` vs PyTorch logit `MaxAbsDiff` (target `< 1e-4`), `Q4_0` vs `FP32` logit cosine similarity (target `> 0.98`), and top-1/top-5 token agreement.

### `P1` — Required for the Next Major Milestone
3. **Compare Pure `Q4_0` (`619.11 MB`) vs Mixed `Q4_0` (`--keep-embeddings-fp32`, `1.07 GB`) Quality on Real 22-Layer Weights:**
   - Quantify logit cosine similarity, top-1 agreement rate, and memory footprint trade-off across 22 layers to document the recommended default quantisation policy for official PS08 demos.

### `P2` — Important Improvements
4. **Precompute RoPE $(\cos\theta, \sin\theta)$ Lookup Table (`KI-006`) and Parallelize Multi-Head GQA Attention (`KI-014`):**
   - Currently `scalar::rope_inplace` computes `std::pow`, `std::cos`, and `std::sin` on every token (`32 * 22 = 704` trig pairs/token). Precomputing a `[max_seq_len, head_dim / 2]` table (`2048 * 32 * 4 = 256 KB` each for `cos` and `sin`) in `TransformerEngine::init()` eliminates all runtime transcendental calls in RoPE.
   - For long contexts (`pos >= 256`), parallelizing `attention_gqa` across the `32` query heads via `thread_pool_` reduces attention latency.

### `P3` — Optional Optimisation or Polish
5. **Fused RMSNorm / Vector Primitives via `dispatch.cpp`:**
   - Optionally route `rmsnorm` or residual vector addition through SIMD helpers if profiling shows non-negligible share (<2% of decode time).

---

## G. New-Session Continuation Prompt (Ready to Paste into Antigravity)

```markdown
You are operating as **Member U (Transformer Engine, Scalar Reference Kernels, `KvCache` & Golden Verification Owner)** on the **BIOS V2 — PS08 (`inference-engine`)** project in repository `shreyaas5/inference-engine`.

Follow this exact 10-step autonomous workflow without asking the team lead for background already documented in the repository:

1. **Read the Master Project Context:** Read `AGENTS.md` and `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`.
2. **Read Your Complete Handoff & Shared Interfaces:** Read `docs/team_handoffs/02_U_TRANSFORMER_ENGINE.md`, `docs/team_handoffs/05_SHARED_INTERFACES_AND_DEPENDENCIES.md`, and `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`.
3. **Inspect the Current Git State:** Run `git fetch --all --prune`, `git status`, `git branch -a -vv`, and `git log --all --oneline --graph -n 20`.
4. **Identify the Latest Relevant Commits:** Verify the consolidated baseline (`integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff`), your merged `Q4_0` transformer commit (`ce1026a7ef90511425f3c258db1ae139dfc77dee`), and your GCC 16 toolchain commit on `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d`).
5. **Inspect the Actual Source Files:** Read `include/inference_engine/{kv_cache,transformer}.hpp`, `src/core/kv_cache.cpp`, `src/kernels/scalar/kernels_scalar.cpp`, `src/transformer/transformer.cpp`, `tools/generate_golden_reference.py`, `tools/verify_transformer_q4.py`, `tests/test_golden_forward.cpp`, and `tests/test_transformer_q4.cpp`. Also check `ls -lh models/TinyLlama-1.1B-Chat-v1.0/` and `ls -lh models/`.
6. **Compare Implementation Against Documented State:** Confirm that `TransformerEngine::forward_token` implements all 16 steps (`FP32`, pure `Q4_0`, and `--keep-embeddings-fp32`), routes projections through `dispatch_matmul_*` with `&thread_pool_`, and commits `KvCache` once per token.
7. **Avoid Reimplementing Completed Functionality:** `KvCache`, scalar kernels, `FP32`/`Q4_0`/mixed `TransformerEngine::forward_token`, and synthetic parity tests (`kernels_scalar`, `kv_cache`, `golden_forward`, `transformer_q4`) are already implemented and passing. Do not rewrite them.
8. **Identify the Highest-Priority Unfinished Tasks (`P0` / `P1` / `P2`):**
   - Check if the full `2,200,120,064`-byte `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` is present on disk (or resume its download if needed), generate real 22-layer golden reference fixtures with `python3 tools/generate_golden_reference.py --model-id models/TinyLlama-1.1B-Chat-v1.0 --output-dir tests/fixtures/golden`, convert/load the 22-layer `.ie` models (`FP32`, pure `Q4_0`, mixed `Q4_0`), and run 22-layer golden forward parity verification.
   - If blocked on external checkpoint download/PyTorch availability, proceed immediately to `P2`: precompute the RoPE `(cos, sin)` lookup table (`[max_seq_len, head_dim / 2]`) in `TransformerEngine::init()` to eliminate per-token `std::pow`/`std::cos`/`std::sin` overhead (`KI-006`), while preserving bit-accurate parity with `scalar::rope_inplace`.
9. **Produce an Initial Repository Audit and Action Plan:** Output a brief audit of your branch state, test status, and checkpoint availability, followed by your concrete execution plan.
10. **Begin Implementation Immediately:** Work autonomously within your owned files (`src/transformer/`, `src/core/`, `src/kernels/scalar/`, `tests/`, `tools/`). Ask the team lead only if an architectural decision changes a shared public header (`include/inference_engine/transformer.hpp`) in a breaking way or if an external resource is unavailable. Update context files per `AGENTS.md` Section 8 before finishing.
```
