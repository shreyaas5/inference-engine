# System Architecture (`inference-engine`)

## 1. Overview

`inference-engine` is a custom C++17 CPU inference runtime engineered specifically for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1 billion parameters) on commodity laptop processors across ARM64 (Apple Silicon NEON) and x86-64 (AMD/Intel AVX2).

The runtime executes inference entirely in native C++17 without delegating execution to PyTorch, TensorFlow, ONNX Runtime, `llama.cpp`, or any external deep-learning runtime. Python is restricted strictly to offline checkpoint inspection, `.ie` model conversion, tokenizer parity verification, and generating trusted golden reference outputs.

## 2. Frozen Model Hyperparameters (`TinyLlama-1.1B-Chat-v1.0`)

| Hyperparameter | Value | Notes |
|---|---|---|
| Architecture | LlamaForCausalLM | Pre-norm decoder-only Transformer |
| Layers (`num_layers`) | `22` | Transformer decoder blocks |
| Hidden dimension (`hidden_dim`) | `2048` | Residual stream width |
| Intermediate dimension (`intermediate_dim`) | `5632` | SwiGLU MLP expansion width |
| Query heads (`num_query_heads`) | `32` | `32 * 64 = 2048` |
| Key/Value heads (`num_kv_heads`) | `4` | Grouped Query Attention (GQA), 8 Q heads per KV head |
| Head dimension (`head_dim`) | `64` | Rotary Position Embedding (RoPE) applied per head |
| Vocabulary size (`vocab_size`) | `32000` | SentencePiece BPE vocabulary |
| Max sequence length (`max_seq_len`) | `2048` | Context window capacity |
| Normalization | RMSNorm | $\epsilon = 10^{-5}$ (`1e-5f`) |
| Positional encoding | RoPE | $\theta = 10000.0$ |
| MLP activation | SwiGLU | $\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})$ |

## 3. Subsystem Boundaries

```text
+-------------------------------------------------------------------+
|                        CLI & Benchmark App                        |
|                         (src/app/main.cpp)                        |
+-----------------+-------------------------------+-----------------+
                  |                               |
                  v                               v
+-----------------------------------+   +---------------------------+
|      Tokenizer & Chat Format      |   |          Sampler          |
|          (src/tokenizer/)         |   |       (src/sampler/)      |
+-----------------------------------+   +---------------------------+
                  |                               ^
                  | token IDs                     | logits [32000]
                  v                               |
+-------------------------------------------------------------------+
|                         TransformerEngine                         |
|                         (src/transformer/)                        |
+-----------------+-------------------------------+-----------------+
                  |                               |
                  v                               v
+-----------------------------------+   +---------------------------+
|      ModelLoader (.ie v1 mmap)    |   |    KvCache (FP32 state)   |
|            (src/loader/)          |   |    (src/core/kv_cache)    |
+-----------------------------------+   +---------------------------+
                  |                               |
                  +---------------+---------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|              Kernel Dispatch & Persistent ThreadPool              |
|          (src/kernels/dispatch.cpp, src/kernels/threading/)       |
+-----------------+-----------------------+-------------------------+
                  |                       |
      +-----------+-----------+           +-------------+
      |                       |                         |
      v                       v                         v
+-------------+       +---------------+         +---------------+
|   Scalar    |       |  ARM64 NEON   |         |  x86-64 AVX2  |
|  Reference  |       | SIMD Kernels  |         | SIMD Kernels  |
+-------------+       +---------------+         +---------------+
```

## 4. Precision & Memory Model

- **Activations:** Always `FP32` (`float`) in the residual stream, attention logits/probabilities, and MLP buffers.
- **KV Cache:** `FP32` initially (`[num_layers=22, max_seq_len=2048, kv_dim=256]`), requiring:
  $$2 \times 22 \times 2048 \times 256 \times 4\text{ bytes} = 92,274,688\text{ bytes} \approx 88\text{ MiB}$$
- **Weights:**
  - **M1 Baseline:** `FP32` weights across all tensors (~4.4 GB model file).
  - **M2/M3 Quantised Runtime:** Mixed-precision `.ie` file where 1D RMSNorm weights remain `FP32` and large 2D matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and optionally `lm_head` / `embed_tokens`) use `Q4_0` block quantisation (~630 MB model file), fitting comfortably within P's 8 GB RAM budget.
