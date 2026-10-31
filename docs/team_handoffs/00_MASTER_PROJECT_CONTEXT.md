# `00_MASTER_PROJECT_CONTEXT.md` — Master Project & Architectural Context

**Document Role:** Primary System & Competition Reference (`docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`)
**Consolidated Integration Branch:** `integration-ps08-team-sync-2026-10-31`
**Target Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## 1. Competition & Project Identity

- **Project / Repository Name:** `inference-engine` (`https://github.com/shreyaas5/inference-engine`)
- **Competition Context:** BIOS V2 — Problem Statement **PS08: Own Inference Engine**
- **C++ Namespace:** `inference_engine`
- **Primary Targets:**
  - Static Library: `inference_engine_core`
  - CLI Executable: `inference-engine`
  - Unit Test Runner: `inference_engine_tests`
- **Model File Extension:** `.ie` (Magic `0x464D4549` / `"IEMF"`, version `1`)

> **Strict Naming Constraint:** `BIOS V2` is the competition name only. Never use `BIOS` as the project name, C++ namespace, binary name, or file prefix.

### Core Competition Rules
1. **No External Inference Runtimes:** Every forward-pass operation (token embedding lookup, RMSNorm, QKV/O/MLP linear projections, RoPE, causal Grouped Query Attention, KV cache management, SwiGLU, final classifier projection, and token sampling) executes in custom C++17 inside `src/`. PyTorch, LibTorch, ONNX Runtime, TensorFlow, `llama.cpp`, GGML, and MLX are strictly forbidden at runtime.
2. **Offline Python Tooling Boundary:** Python scripts in `tools/` are restricted to offline `.safetensors` inspection, `.ie` model conversion/quantisation, golden activation/logit fixture generation, and tokenizer parity testing.
3. **CPU-Only Official Execution:** Target hardware is commodity laptops running ARM64 NEON and x86-64 AVX2/FMA CPUs. No GPU, Metal, CUDA, or NPU execution is used.

---

## 2. Frozen Target Model Specification (`TinyLlama-1.1B-Chat-v1.0`)

All architectural dimensions are frozen in [`include/inference_engine/config.hpp`](../../include/inference_engine/config.hpp) and verified against the live Hugging Face `TinyLlama/TinyLlama-1.1B-Chat-v1.0` `config.json` and 23,088-byte `model.safetensors` header (`201` `BF16` tensors, `1,100,048,384` parameters):

| Parameter | Symbol / Constant | Value | Notes |
|---|---|---|---|
| Architecture | `LlamaForCausalLM` | Decoder-only | Pre-norm RMSNorm, RoPE, GQA, SwiGLU |
| Decoder Layers | `kTinyLlamaNumLayers` | `22` | `model.layers.{0..21}.*` |
| Hidden Size | `kTinyLlamaHiddenDim` | `2048` | Residual stream width |
| Intermediate Size | `kTinyLlamaIntermediateDim` | `5632` | SwiGLU MLP hidden dimension |
| Query Heads | `kTinyLlamaNumHeads` | `32` | `32 * 64 = 2048` |
| Key/Value Heads | `kTinyLlamaNumKvHeads` | `4` | `4 * 64 = 256` (`8` Q heads per KV head) |
| Head Dimension | `kTinyLlamaHeadDim` | `64` | RoPE applied across `64` dims (`rotate_half` pairs `i` and `i + 32`) |
| Vocabulary Size | `kTinyLlamaVocabSize` | `32000` | SentencePiece BPE (`<unk>=0`, `<s>=1`, `</s>=2`) |
| Max Sequence Length | `kTinyLlamaMaxSeqLen` | `2048` | Context window (`0 .. 2047`) |
| RMSNorm $\epsilon$ | `kTinyLlamaRmsNormEps` | `1e-5f` | `1.0e-5` |
| RoPE $\theta$ | `kTinyLlamaRopeTheta` | `10000.0f` | $\text{freq}_i = 10000^{-2i / 64}$ for $i \in [0, 31]$ |
| Untied Embeddings | `tie_word_embeddings` | `false` | Separate `model.embed_tokens.weight` and `lm_head.weight` |
| Attention/MLP Bias | `attention_bias`, `mlp_bias` | `false` | Zero bias tensors in checkpoint |

### Per-Layer & Global Tensor Inventory (`201` Tensors Total)

Every valid TinyLlama `.ie` model contains exactly $3 + 9 \times \text{num\_layers}$ tensors (`201` for `22` layers):

| Tensor Name Pattern | Shape | Elements | Allowed `.ie` Dtypes |
|---|---|---|---|
| `model.embed_tokens.weight` | `[32000, 2048]` | `65,536,000` | `FP32` or `Q4_0` |
| `model.layers.{i}.input_layernorm.weight` | `[2048]` | `2,048` | `FP32` always |
| `model.layers.{i}.self_attn.q_proj.weight` | `[2048, 2048]` | `4,194,304` | `FP32` or `Q4_0` |
| `model.layers.{i}.self_attn.k_proj.weight` | `[256, 2048]` | `524,288` | `FP32` or `Q4_0` |
| `model.layers.{i}.self_attn.v_proj.weight` | `[256, 2048]` | `524,288` | `FP32` or `Q4_0` |
| `model.layers.{i}.self_attn.o_proj.weight` | `[2048, 2048]` | `4,194,304` | `FP32` or `Q4_0` |
| `model.layers.{i}.post_attention_layernorm.weight` | `[2048]` | `2,048` | `FP32` always |
| `model.layers.{i}.mlp.gate_proj.weight` | `[5632, 2048]` | `11,534,336` | `FP32` or `Q4_0` |
| `model.layers.{i}.mlp.up_proj.weight` | `[5632, 2048]` | `11,534,336` | `FP32` or `Q4_0` |
| `model.layers.{i}.mlp.down_proj.weight` | `[2048, 5632]` | `11,534,336` | `FP32` or `Q4_0` |
| `model.norm.weight` | `[2048]` | `2,048` | `FP32` always |
| `lm_head.weight` | `[32000, 2048]` | `65,536,000` | `FP32` or `Q4_0` |

### Exact `.ie` File Sizes for 22-Layer TinyLlama-1.1B

- **Pure `FP32` (`--dtype fp32`):** `4,400,213,568` bytes (`4.40 GB` / `4.10 GiB`)
- **Pure `Q4_0` (`--dtype q4_0`, 1D RMSNorm weights kept `FP32`):** `619,113,472` bytes (`619.11 MB` / `590.43 MiB` — **7.11x compression** over `FP32`)
- **Mixed-Precision `Q4_0` (`--dtype q4_0 --keep-embeddings-fp32`):** `1,069,673,472` bytes (`1.07 GB` / `1,020.12 MiB`)

---

## 3. End-to-End Inference Dataflow

```
User Prompt (UTF-8 string)
  │
  ▼  [Tokenizer::format_chat_prompt] (optional --chat Zephyr template)
  ▼  [Tokenizer::encode] (SentencePiece C++ -> prompt_ids[0 .. N-1], prepends BOS=1)
  │
  ├─► PREFILL LOOP: for pos = 0 .. N-1:
  │     TransformerEngine::forward_token(prompt_ids[pos], pos, logits)
  │       ├─ 1. Embedding lookup: FP32 copy or single-row dequantize_row_q4_0 -> x_[2048]
  │       ├─ 2. For layer_idx = 0 .. num_layers-1:
  │       │     ├─ rmsnorm_f32(x_, input_layernorm) -> xb_[2048]
  │       │     ├─ matmul(q_proj, xb_) -> q_[2048]; matmul(k_proj, xb_) -> k_[256]; matmul(v_proj, xb_) -> v_[256]
  │       │     ├─ rope_inplace_f32(q_, pos, 32, 64); rope_inplace_f32(k_, pos, 4, 64)
  │       │     ├─ kv_cache_.append(layer_idx, pos, k_, v_)   // Does NOT increment seq_len()
  │       │     ├─ gqa_attention_f32(q_, kv_cache_, pos, 32, 4, 64) -> att_out_[2048]
  │       │     ├─ matmul(o_proj, att_out_) -> xb_[2048]; x_ += xb_
  │       │     ├─ rmsnorm_f32(x_, post_attention_layernorm) -> xb_[2048]
  │       │     ├─ matmul(gate_proj, xb_) -> gate_[5632]; matmul(up_proj, xb_) -> up_[5632]
  │       │     ├─ swiglu_f32(gate_, up_) -> hb_[5632]
  │       │     └─ matmul(down_proj, hb_) -> xb_[2048]; x_ += xb_
  │       ├─ 3. kv_cache_.commit_token()                      // Increments seq_len() once after all layers
  │       ├─ 4. rmsnorm_f32(x_, final_norm) -> xb_[2048]
  │       └─ 5. matmul(lm_head, xb_) -> logits[32000]
  │
  ├─► FIRST GENERATED TOKEN (sampled from logits at pos = N-1):
  │     next_token = Sampler::sample(logits, 32000)           // TTFT measured here
  │
  └─► AUTOREGRESSIVE DECODE LOOP: for step = 1 .. max_new_tokens-1 (while next_token != EOS=2):
        pos = N + step - 1
        TransformerEngine::forward_token(next_token, pos, logits)
        next_token = Sampler::sample(logits, 32000)
  │
  ▼  [Tokenizer::decode] (SentencePiece C++ detokenization of generated_ids)
Generated Response + Benchmark Telemetry (Load ms, TTFT ms, Prompt tok/s, Decode tok/s, Peak RSS MiB, CPU %)
```

---

## 4. Team Ownership & Consolidated Branch Topology

| Member | Primary Ownership | Feature Branches Merged into `integration-ps08-team-sync-2026-10-31` | Hand-Off Document |
|---|---|---|---|
| **S** (Lead) | `.ie` v1 format, `ModelLoader` (`mmap`), `TensorView`, `.safetensors` converter (`FP32`/`Q4_0`), real checkpoint quant error evaluation, cross-team integration | `origin/feature/model-loader` (`aa40368`), `origin/feature/model-format-q4` (`cb176b3`), `origin/feature/real-checkpoint-validation` (`4515579`) | [`01_S_INTEGRATION_AND_MODEL_SYSTEM.md`](01_S_INTEGRATION_AND_MODEL_SYSTEM.md) |
| **U** | Scalar reference kernels (`rmsnorm`, `rope`, `swiglu`, `softmax`, `gqa_attention`), `KvCache`, 16-step `TransformerEngine` (`FP32` + `Q4_0` + `ThreadPool`), golden reference generator & `Q4_0` NumPy verifier | `origin/feature/transformer` (`5115467`), `origin/feature/transformer-bindings` (`bce66ae`), `origin/feature/transformer-q4` (`ce1026a`) | [`02_U_TRANSFORMER_ENGINE.md`](02_U_TRANSFORMER_ENGINE.md) |
| **I** | `BlockQ4_0` packing/dequantisation, ARM64 NEON kernels, x86-64 AVX2/FMA kernels, CPUID/XGETBV detection, `ThreadPool` task chunking & exception propagation, kernel dispatch | `origin/feature/quantisation-simd` (`f76ab55`), `origin/feature/quantisation-simd-m3` (`58c604d`) | [`03_I_QUANTISATION_AND_CPU_KERNELS.md`](03_I_QUANTISATION_AND_CPU_KERNELS.md) |
| **P** | SentencePiece C++ `Tokenizer`, Zephyr chat template, `Sampler` (greedy, temp, top-p), CLI application & `--benchmark` harness, Hugging Face parity script, GitHub Actions CI | `origin/feature/runtime-tokenizer` (`ce769c8`) | [`04_P_TOKENIZER_AND_RUNTIME.md`](04_P_TOKENIZER_AND_RUNTIME.md) |

Additional cross-cutting documents in this folder:
- [`05_SHARED_INTERFACES_AND_DEPENDENCIES.md`](05_SHARED_INTERFACES_AND_DEPENDENCIES.md) — Exact header contracts (`tensor.hpp`, `loader.hpp`, `quant.hpp`, `kernels.hpp`, `thread_pool.hpp`, `kv_cache.hpp`, `transformer.hpp`, `tokenizer.hpp`, `sampler.hpp`).
- [`06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`](06_CURRENT_STATUS_AND_NEXT_ACTIONS.md) — Honest implementation matrix, verification evidence, and exact commands to complete real 22-layer TinyLlama inference and 4-laptop benchmarking.
