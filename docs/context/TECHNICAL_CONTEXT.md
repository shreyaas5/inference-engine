# `TECHNICAL_CONTEXT.md` — Frozen Contracts & Implementation Details

**Last Updated:** 2026-10-31  
*(Each section explicitly labels what is **VERIFIED IN CODE** vs **PLANNED FOR M1–M4**.)*

---

## 1. TinyLlama-1.1B-Chat-v1.0 Architecture & Hyperparameters

- **Status:** **VERIFIED IN CODE** ([`include/inference_engine/config.hpp`](../../include/inference_engine/config.hpp))
- **Constants:**
  - `kTinyLlamaNumLayers = 22`
  - `kTinyLlamaHiddenDim = 2048`
  - `kTinyLlamaIntermediateDim = 5632`
  - `kTinyLlamaNumQueryHeads = 32` (`q_dim() = 32 * 64 = 2048`)
  - `kTinyLlamaNumKvHeads = 4` (`kv_dim() = 4 * 64 = 256`, `gqa_group_size() = 8`)
  - `kTinyLlamaHeadDim = 64`
  - `kTinyLlamaVocabSize = 32000`
  - `kTinyLlamaMaxSeqLen = 2048`
  - `kTinyLlamaRmsNormEps = 1e-5f`
  - `kTinyLlamaRopeTheta = 10000.0f`

---

## 2. Canonical Tensor Naming & Shapes (201 Tensors)

- **Status:** **VERIFIED AGAINST REAL CHECKPOINT & IN CODE** ([`tools/inspect_checkpoint.py`](../../tools/inspect_checkpoint.py), [`tools/convert_to_inference_engine.py`](../../tools/convert_to_inference_engine.py), `ModelLoader::bind_transformer_weights()` in [`src/loader/loader.cpp`](../../src/loader/loader.cpp)).
- **Verified `TinyLlama/TinyLlama-1.1B-Chat-v1.0` `model.safetensors` Inventory (`23,088`-byte header, `2,200,096,768` bytes `BF16` payload, `3 + 9 * num_layers = 201` tensors):**
  - `lm_head.weight`: `[32000, 2048]` (source `BF16`, offsets `[0, 131072000]`; `FP32` in M1; `FP32` or `Q4_0` in M2; untied from `embed_tokens`)
  - `model.embed_tokens.weight`: `[32000, 2048]` (source `BF16`, offsets `[131072000, 262144000]`; `FP32` in M1; `FP32` or `Q4_0` in M2)
  - For each `i` in `0 .. 21` (22 layers $\times$ 9 tensors = 198 tensors; stored in `.safetensors` in per-layer order `input_layernorm`, `mlp.down_proj`, `mlp.gate_proj`, `mlp.up_proj`, `post_attention_layernorm`, `self_attn.k_proj`, `self_attn.o_proj`, `self_attn.q_proj`, `self_attn.v_proj`):
    - `model.layers.{i}.input_layernorm.weight`: `[2048]` (`FP32`)
    - `model.layers.{i}.self_attn.q_proj.weight`: `[2048, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.self_attn.k_proj.weight`: `[256, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.self_attn.v_proj.weight`: `[256, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.self_attn.o_proj.weight`: `[2048, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.post_attention_layernorm.weight`: `[2048]` (`FP32`)
    - `model.layers.{i}.mlp.gate_proj.weight`: `[5632, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.mlp.up_proj.weight`: `[5632, 2048]` (`FP32` in M1, `Q4_0` in M2)
    - `model.layers.{i}.mlp.down_proj.weight`: `[2048, 5632]` (`FP32` in M1, `Q4_0` in M2)
  - `model.norm.weight`: `[2048]` (source `BF16`, offsets `[2200092672, 2200096768]`; `FP32` in `.ie`)
  - **Total Tensors:** $3 + (22 \times 9) = 201$ tensors (`1,100,048,384` parameters = `4,400,193,536` payload bytes in `FP32`).

---

## 3. `.ie` Binary File Format (Version 1)

- **Status:** **VERIFIED IN CODE** ([`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp), [`src/loader/loader.cpp`](../../src/loader/loader.cpp), [`src/common/tensor.cpp`](../../src/common/tensor.cpp)).
- **Header (64 bytes, little-endian):**
  - Bytes `0..3`: `magic = 0x464D4549` (`"IEMF"`)
  - Bytes `4..7`: `version = 1`
  - Bytes `8..11`: `tensor_count` ($\le 4096$)
  - Bytes `12..15`: `weight_dtype` (`0`=FP32, `1`=FP16, `2`=Q4_0, `3`=I32)
  - Bytes `16..47`: 8 $\times$ `u32` hyperparameters (`num_layers`, `hidden_dim`, `intermediate_dim`, `num_query_heads`, `num_kv_heads`, `head_dim`, `vocab_size`, `max_seq_len`)
  - Bytes `48..55`: 2 $\times$ `f32` (`rms_norm_eps`, `rope_theta`)
  - Bytes `56..63`: `total_file_bytes` (`u64`)
- **Directory Entry (variable length: `4 + name_len + 56` bytes):**
  - `name_len` (`u32`, $1..256$), `name` (`UTF-8`), `dtype` (`u32`), `ndim` (`u32`, $1..4$), `shape[4]` ($4 \times \text{u64}$, unused trailing dims $= 0$), `payload_offset` (`u64`), `payload_nbytes` (`u64`).
- **Validation Rules Enforced:**
  - `payload_offset % 64 == 0` (`kIePayloadAlignment = 64`).
  - `payload_nbytes` matches exact expected byte count for `shape` and `dtype`.
  - For `Q4_0`, `shape[ndim - 1] % 32 == 0` AND total elements $\% 32 == 0$.
  - No payload overlaps the `[0, align_up(directory_end, 64))` region or any other tensor's payload range.
  - `payload_ranges.back().end == total_file_bytes` (no unreferenced trailing bytes).
- **Mixed-Precision Policy in `.ie` (`ADR-008`):**
  - 1D RMSNorm tensors (`45` tensors: `*.input_layernorm.weight`, `*.post_attention_layernorm.weight`, `model.norm.weight`) are **always stored as `DataType::FP32`** (`368,640` bytes total).
  - In `--dtype q4_0` mode (`header.weight_dtype = DataType::Q4_0`), all 154 2D layer projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are stored as `DataType::Q4_0` (`544,997,376` bytes). `model.embed_tokens.weight` and `lm_head.weight` (`[32000, 2048]`) default to `DataType::Q4_0` (`36,864,000` bytes each, yielding a **`619,113,472`-byte / `619.11 MB`** `.ie` file), or remain `DataType::FP32` when exported with `--keep-embeddings-fp32` (`1,069,673,472` bytes).
  - `TensorView` (`include/inference_engine/tensor.hpp`) provides typed `data_f32()` (`const float*`) and `data_q4_0()` (`const BlockQ4_0*`) accessors along with `dtype()`, `ndim()`, `shape()`, `payload_nbytes()`, `num_elements()`, and `num_blocks()`.

---

## 4. `Q4_0` Quantisation Format

- **Status:** **VERIFIED IN CODE** ([`include/inference_engine/quant.hpp`](../../include/inference_engine/quant.hpp), [`src/kernels/quant_q4_0.cpp`](../../src/kernels/quant_q4_0.cpp)).
- **Block Structure (`BlockQ4_0`, 18 bytes packed):**
  - `uint16_t d`: IEEE 754 binary16 scale factor.
  - `uint8_t qs[16]`: 32 4-bit unsigned values (`0..15`).
  - **Low nibble** (`qs[j] & 0x0F`): element `j` (`0..15`).
  - **High nibble** (`(qs[j] >> 4) & 0x0F`): element `j + 16` (`16..31`).
- **Quantisation Math:**
  - Find `signed_max` = first element `v` in block maximizing `std::fabs(v)`.
  - `d = signed_max / -8.0f`, `id = (d != 0.0f) ? (1.0f / d) : 0.0f`.
  - `q = std::clamp(static_cast<int>(std::floor(x * id + 8.5f)), 0, 15)`.
- **Dequantisation Math:**
  - `x_j = (static_cast<int>(qs[j] & 0x0F) - 8) * fp16_to_fp32(d)`
  - `x_{j+16} = (static_cast<int>((qs[j] >> 4) & 0x0F) - 8) * fp16_to_fp32(d)`

---

## 5. `KvCache` Semantics

- **Status:** **VERIFIED IN CODE** ([`include/inference_engine/kv_cache.hpp`](../../include/inference_engine/kv_cache.hpp), [`src/core/kv_cache.cpp`](../../src/core/kv_cache.cpp)).
- **Storage Layout:** Flat `std::vector<float>` for keys and values indexed by:
  $$\text{offset}(\text{layer\_idx}, \text{pos}) = (\text{layer\_idx} \times \text{max\_seq\_len} + \text{pos}) \times \text{kv\_dim}$$
- **Contract:**
  - `append(layer_idx, pos, k_data, v_data)` copies `kv_dim` (`256`) floats into `k_storage_` and `v_storage_` and **does NOT modify `committed_seq_len_`**.
  - `commit_token()` (or `commit_positions(count)`) increments `committed_seq_len_` after all 22 layers finish.

---

## 6. Mathematical Kernels (RMSNorm, RoPE, GQA, SwiGLU, MatMul)

- **RMSNorm (`VERIFIED` / `TESTED` — `src/kernels/scalar/kernels_scalar.cpp`):**
  - Accumulates sum of squares in `float` (`sum_sq`), computes `inv_rms = 1.0f / std::sqrt((sum_sq / dim) + eps)`, and outputs `(x[i] * inv_rms) * weight[i]`.
- **RoPE (`VERIFIED` / `TESTED` — `src/kernels/scalar/kernels_scalar.cpp`):**
  - Uses Hugging Face Llama **half-dim split** rotation (`rotate_half`, pairs `(i, i + head_dim / 2)` for `i = 0 .. head_dim/2 - 1`), with frequency $\theta^{-2i / d_{\text{head}}}$ and angle $\text{pos} \cdot \text{freq}$:
    $$y_i = x_i \cos\alpha - x_{i + d/2} \sin\alpha, \qquad y_{i + d/2} = x_{i + d/2} \cos\alpha + x_i \sin\alpha$$
- **SwiGLU (`VERIFIED` / `TESTED` — `src/kernels/scalar/kernels_scalar.cpp`):**
  - `out[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i]`.
- **Causal GQA (`VERIFIED` / `TESTED` — `src/kernels/scalar/kernels_scalar.cpp`):**
  - Query head $h_q \in [0, 31]$ maps to KV head $h_{kv} = \lfloor h_q / 8 \rfloor \in [0, 3]$.
  - Scores for $t \in [0, \text{pos}]$: $s_t = \text{dot}(q_{h_q}, k_{h_{kv}, t}) / \sqrt{64}$, followed by `softmax_inplace(s, pos + 1)` and weighted accumulation of $v_{h_{kv}, t}$.

---

## 7. SIMD Dispatch & Persistent ThreadPool

- **Status:** **`VERIFIED` / `TESTED`** ([`src/kernels/dispatch.cpp`](../../src/kernels/dispatch.cpp), [`src/kernels/threading/thread_pool.cpp`](../../src/kernels/threading/thread_pool.cpp), [`src/kernels/avx2/kernels_avx2.cpp`](../../src/kernels/avx2/kernels_avx2.cpp), [`src/kernels/neon/kernels_neon.cpp`](../../src/kernels/neon/kernels_neon.cpp)).
- **CPU Detection:** `detect_best_backend()` returns `SimdBackend::Neon` on ARM64, `SimdBackend::Avx2` on x86-64 when runtime CPUID + OS XSAVE (`xgetbv`) confirm `AVX2` + `FMA`, and `SimdBackend::Scalar` otherwise.
- **ThreadPool:** Spawns `num_threads - 1` persistent background `std::thread` workers in the constructor; thread `0` participates in dynamic chunked execution (`std::max<size_t>(1, num_tasks / (num_threads * 4))`) during `parallel_for()`, with `std::exception_ptr` capture and rethrow on the calling thread (`ADR-010`).
- **SIMD Coverage:**
  - `dot_f32` & `matmul_f32`: Vectorized in both `neon` (`vfmaq_f32`) and `avx2` (`_mm256_fmadd_ps`, with 2-row register tiling on `origin/feature/cpu-kernel-optimisation` `b05c4e4`).
  - `dot_q4_0_f32` & `matmul_q4_0_f32`: Vectorized with fused nibble unpack + signed-offset subtraction (`q - 8`) + FMA accumulation in both `neon` (`vld1q_u8`, `vcvt_f32_f16`, `vfmaq_f32`) and `avx2` (`_mm256_cvtepi8_epi32`, `_mm256_fmadd_ps`). On `origin/feature/cpu-kernel-optimisation` (`b05c4e4`), `avx2::dot_q4_0_f32` uses vectorized per-block scale accumulation (`compute_block_unscaled_dot_vec`) with a single end-of-row `hsum256_ps`, and `avx2::matmul_q4_0_f32` + `dispatch.cpp` use 2-row register tiling (`ADR-012`). Porting the 2-row tiling and deferred reduction to `neon` is tracked as `P1` for Member I.

---

## 8. Tokenizer, Chat Template & Autoregressive Sampling

- **Chat Template (`REAL-CHECKPOINT VALIDATED` against `models/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json`):**
  ```text
  <|system|>
  {system_message}</s>
  <|user|>
  {user_message}</s>
  <|assistant|>
  ```
- **Special Token IDs:** `<unk> = 0`, `<s> = 1`, `</s> = 2` (`vocab_size = 32000`).
- **SentencePiece C++ Binding (`REAL-CHECKPOINT VALIDATED` — `src/tokenizer/tokenizer.cpp`):** CMake discovers SentencePiece `>= 0.2.0` or builds `sentencepiece-static` via `FetchContent`. `Tokenizer` uses dual processors (`processor_` and `processor_no_prefix_` with `add_dummy_prefix = false`) and explicit special-token splitting, passing `15/15` prompts in `tools/verify_tokenizer_parity.py --self-test`.
- **Sampler (`VERIFIED` / `TESTED` — `src/sampler/sampler.cpp`):** Greedy argmax breaks ties toward the lowest ID; temperature/top-p sampling uses a seeded 64-bit `xorshift*` generator (`0x2545F4914F6CDD1DULL`) and sanitizes `NaN`/`-Inf` logits.
- **CLI (`SYNTHETICALLY VALIDATED` — `src/app/main.cpp`):** Supports both `FP32` and `Q4_0` `.ie` models, routes projections through `TransformerEngine`'s persistent `ThreadPool` (`--threads`), prefills prompt token positions `0 .. N-1`, samples the first generated token from `pos = N-1` logits, feeds subsequent generated tokens at `pos = N, N+1, ...`, and reports TTFT, prefill/decode tok/s, CPU utilization %, and Peak RSS MB under `--benchmark`.
- **Autoregressive Position Indexing (`VERIFIED`):**
  - A 12-token prompt occupies `pos = 0..11`; `pos = 11` logits predict the first generated token, which is then processed at `pos = 12`.
