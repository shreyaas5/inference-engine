# BIOS V2 — Master Technical Guide: Engineering Architecture & Defense Manual
## Project: `inference-engine` | Problem Statement: PS08 (Own Inference Engine)
**Document Classification:** Internal Technical Defense & Pitch Mastery Guide  
**Target Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Decoder-Only Transformer, 22 Layers, GQA 8:1)  
**Numerical Precision:** `Q4_0` Block Quantised Weights (18 bytes / 32 weights), `FP32` Activations & KV Cache  
**Runtime Architecture:** Zero-Framework C++17, POSIX `mmap`, ARM64 NEON & x86-64 AVX2/FMA SIMD, Persistent ThreadPool, JSON-Lines Telemetry Subprocess, Node.js Gateway, React 19 Observability Studio

---

## Executive Pitch Summary

`inference-engine` is an independent, zero-framework, hardware-optimised Large Language Model (LLM) inference runtime written from scratch in C++17. It was designed, engineered, verified, and benchmarked for **BIOS V2 Problem Statement PS08 ("Own Inference Engine")**.

The engine executes real 1.1-billion-parameter TinyLlama checkpoints (`TinyLlama-1.1B-Chat-v1.0`) on commodity laptop CPUs. It achieves **~35.5 tokens/sec autoregressive decode throughput** and a **~600 ms Time-To-First-Token (TTFT)** within a **~658 MiB peak RAM footprint** on consumer hardware without linking, embedding, wrapping, or calling any external machine learning framework (no PyTorch, LibTorch, ONNX Runtime, GGML, `llama.cpp`, Ollama, or Apple MLX).

---

# Chapter 1: What Exactly Did We Build?

### 1.1 What is an Inference Engine?
A Large Language Model consists of two fundamentally distinct phases of computation:
1. **Model Architecture & Parameters:** A frozen directed acyclic graph of mathematical tensor operations (linear projections, normalizations, rotary positional embeddings, causal softmax self-attention, non-linear activation functions) parameterized by millions or billions of numerical weights.
2. **Execution Engine (Runtime):** The software layer that allocates memory, loads and binds weights, schedules execution across CPU registers and cache hierarchies, evaluates matrix mathematics using SIMD vector instructions, maintains autoregressive state across token iterations, and samples output tokens from vocabulary probability distributions.

An **inference engine** is a specialized runtime focused exclusively on the execution of trained neural networks in forward-evaluation mode. Unlike training runtimes, an inference engine requires no backward pass, no gradient tracking, no optimizer state (Adam/SGD), no loss calculation, and no distributed backward communication. Instead, it is hyper-optimized for **memory footprint, cache locality, memory bandwidth saturation, and low latency**.

### 1.2 What Problem Does BIOS V2 / PS08 Solve?
Modern generative AI is overwhelmingly trapped in heavy, opaque cloud ecosystems or monolithic third-party runtimes:
- **Cloud API Limitations:** Cloud-hosted LLMs introduce network latency, ongoing token costs, privacy hazards for proprietary data, and total dependence on external internet connectivity.
- **The "Wrapper" Trap:** Many local LLM desktop apps are shallow wrappers: an Electron shell that shells out to a precompiled `llama.cpp` binary or connects to a local Ollama daemon. When memory leaks, race conditions, or numerical drift occur, the developers cannot debug, profile, or defend the underlying execution.
- **PS08 Core Mandate:** Design and build our **own** inference engine from first principles in C++17, demonstrating complete ownership over:
  - Custom binary file format (`.ie`) with strict byte alignment.
  - Zero-copy virtual memory architecture (`mmap`).
  - Custom 4-bit block quantization format (`Q4_0`) and packing routines.
  - Architecture-specific vector math kernels (ARM NEON and x86 AVX2/FMA).
  - Contiguous Key-Value caching with position commit and rewind semantics.
  - Tokenization, sampling, and live system telemetry.

### 1.3 What is the Difference Between Training and Inference?
| Dimension | Training (e.g. PyTorch / Megatron-LM) | Inference (`inference-engine`) |
| :--- | :--- | :--- |
| **Computation Graph** | Bidirectional (Forward pass + Backward gradient backpropagation) | Unidirectional (Forward pass only) |
| **Memory Allocation** | Massive: Activations for all tokens retained for backprop + Adam optimizer states (8–16 bytes/param) | Minimal: Activations reused across layers in a static ~393 KB workspace; KV cache bounds context |
| **Batching Regime** | Large batch sizes ($B = 32 \dots 1024$) to saturate massively parallel GPUs | Latency-critical single-batch ($B = 1$) interactive streaming decode |
| **Bottleneck Profile** | Compute-bound (Tensor core arithmetic throughput, FLOPS) | Prefill: Compute-bound; Decode: **Memory-bandwidth bound** |
| **Weight Modification** | Dynamic in-place gradient updates ($W \leftarrow W - \eta \nabla W$) | **Read-only, strictly immutable** |
| **Target Hardware** | Datacenter GPU clusters (NVIDIA H100/A100) with NVLink | **Commodity laptop CPUs (ARM64 & x86-64)** |

### 1.4 Why Did We Choose `TinyLlama-1.1B`?
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` was deliberately locked as our target architecture (`ADR-002`) because it represents the optimal intersection of **modern architectural complexity** and **commodity CPU feasibility**:
1. **Modern Architecture:** Unlike early architectures (GPT-2, BERT) which lack modern techniques, TinyLlama is an exact scaled-down replica of **Llama 2 / Llama 3**. It features **RMSNorm** (Zhang & Sennrich, 2019), **Rotary Positional Embeddings (RoPE)** (Su et al., 2021), **SwiGLU feed-forward networks** (Shazeer, 2020), and **Grouped Query Attention (GQA)** (Ainslie et al., 2023) with an $8:1$ head ratio.
2. **Commodity Memory Fit:** In full `FP32`, 1.1 billion parameters require $4.40\text{ GB}$ of RAM. When quantized to `Q4_0`, the entire model payload shrinks to **$619.11\text{ MB}$**. This leaves ample headroom to run comfortably on low-spec laptops (e.g., an 8 GB Intel Core i5 laptop) without swapping.
3. **Legitimate Reasoning & Dialogue:** Pretrained on 3 trillion tokens from SlimPajama and Starcoder, TinyLlama produces coherent, syntactically valid English and code, enabling judges to observe real-world interactive chat responses rather than toy synthetic tokens.

### 1.5 What Does "Owning Our Inference Engine" Mean?
Owning our engine means that from the moment raw bytes are read from disk to the moment an ASCII character renders on screen, **every single byte and clock cycle is governed by code written inside this repository**:
- We designed the binary format specification (`docs/FORMAT_SPECIFICATION.md`).
- We write the binary parsing, pointer validation, and memory-mapping routines (`src/loader/loader.cpp`).
- We pack FP32 weights into 4-bit nibbles and IEEE 754 half-precision scales (`src/kernels/quant_q4_0.cpp`).
- We author the ARM64 NEON (`src/kernels/neon/kernels_neon.cpp`) and x86 AVX2 (`src/kernels/avx2/kernels_avx2.cpp`) assembly intrinsics that unpack nibbles, subtract offsets, and compute fused multiply-accumulate operations in CPU vector registers.
- We allocate and manage the contiguous ring-buffer KV cache (`src/core/kv_cache.cpp`).
- We schedule multi-core work using our persistent, chunked thread pool (`src/kernels/threading/thread_pool.cpp`).
- We drive the interactive workstation and live hardware telemetry (`src/app/studio_worker.hpp`).

### 1.6 What Did We Implement Ourselves vs. What Did We Reuse?
Transparency is mandatory for technical defense:
- **Implemented 100% in Custom Repository Code:**
  - Complete tensor model loader and memory-mapped file parser (`ModelLoader`, `TensorView`).
  - `.ie` format specification and offline safetensors-to-`.ie` converter (`tools/convert_to_inference_engine.py`).
  - `Q4_0` quantization, block packing, and software IEEE 754 FP16 bitwise conversion.
  - Scalar numerical reference library (`rmsnorm_f32`, `rope_inplace_cached_f32`, `gqa_attention_f32`, `swiglu_f32`, `softmax_f32`).
  - SIMD kernels: 2-row register-tiled NEON and AVX2+FMA vector matrix-vector multiplication with vectorized scale accumulation.
  - Runtime CPU detection (`is_backend_available`, `detect_best_backend`) using `__builtin_cpu_supports` and `cpuid` + `_xgetbv`.
  - Contiguous float KV cache with token commit, prefix-matching, and rewind truncation (`KvCache`).
  - Full 16-step Transformer execution pipeline (`TransformerEngine`).
  - Seeded 64-bit `xorshift*` PRNG, greedy argmax, temperature scaling, top-p nucleus filtering, and top-k filtering (`Sampler`).
  - CLI application, interactive chat REPL, and multi-run benchmark statistical harness (`src/app/main.cpp`).
  - Persistent JSON-lines telemetry worker (`src/app/studio_worker.hpp`) instrumented with OS-level `mach_task_basic_info` and `/proc/self/statm` RSS readers.
  - Node.js API Gateway, SSE event dispatcher, and React 19 / Tailwind CSS workstation frontend (`studio/`).
- **Carefully Reused External Component:**
  - **SentencePiece C++ Core (`sentencepiece-static` v0.2.0):** Used strictly for BPE subword segmentation (`ADR-007` & `ADR-011`). We do not write subword regex splitting from scratch because SentencePiece normalization (`▁` whitespace tags, byte-fallback tokens) is an established tokenization utility, not neural network matrix inference. However, we wrote our own dual-processor C++ wrapper (`src/tokenizer/tokenizer.cpp`) to handle TinyLlama's Zephyr special tokens (`<|user|>`, `<|assistant|>`, `<s>`, `</s>`) and special-token dummy prefix alignment.

### 1.7 Why is this Meaningfully Different from Wrapping `llama.cpp`, Ollama, or MLX?
1. **Zero External Runtime Binary Dependency:** If you compile our binary with `cmake --build build`, you produce a standalone static binary `inference-engine`. It links to `libc++`, `libm`, `pthread`, and `sentencepiece-static`. It does not require `libllama.so`, `libtorch.dylib`, Python runtimes, or external daemons.
2. **Transparent Architectural Guarantees:** Third-party runtimes hide details behind layers of abstraction, dynamic dispatch, and generic operator graphs. Our engine is tailored specifically to TinyLlama/Llama-class decoders. Every tensor dimension, buffer offset, and register layout is statically predictable and verifiable in code.
3. **No Black-Box Telemetry:** Studio worker telemetry does not read synthetic estimations from an API. It queries kernel Mach task ports and OS procfs tables to prove exact physical memory allocation to judges in real time.

---

# Chapter 2: End-to-End Inference Pipeline

Below is the complete end-to-end execution path traced from the user's raw prompt string to the generated output token.

```
[ User Input: "Explain entropy." ]
               │
               ▼
┌──────────────────────────────────────────────┐
│ 1. Chat Template & Special Token Formatting  │  src/tokenizer/tokenizer.cpp
│    format_chat_prompt()                      │  Adds <|user|>\n...</s>\n<|assistant|>\n
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. SentencePiece Tokenization                │  src/tokenizer/tokenizer.cpp
│    Tokenizer::encode()                       │  Produces std::vector<int32_t> Token IDs
└──────────────────────┬───────────────────────┘
                       │  e.g., [1, 529, 29989, 5205, ...]
                       ▼
┌──────────────────────────────────────────────┐
│ 3. KV-Cache Prefix Matching & Truncation     │  src/app/main.cpp / studio_worker.hpp
│    KvCache::truncate()                       │  Reuses common tokens from prior turns
└──────────────────────┬───────────────────────┘
                       │
         ══════════════╪══════════════════════════════════════════════
         PHASE A: PROMPT PREFILL (Positions 0 .. N-1)
         ══════════════╪══════════════════════════════════════════════
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. Token Embedding Lookup                    │  src/transformer/transformer.cpp:153
│    x = embed_tokens.weight[token_id]         │  Copies FP32 row or dequantizes Q4_0 block
└──────────────────────┬───────────────────────┘
                       │  Shape: [2048] float
                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 5. 22-Layer Decoder Loop (layer_idx = 0 .. 21)                         │
 │                                                                        │
 │  5a. Pre-Attention RMSNorm:                                            │
 │      u = x * inv_rms(x) * input_layernorm.weight                       │
 │      (src/kernels/scalar/kernels_scalar.cpp:9)                         │
 │                                                                        │
 │  5b. Q, K, V Linear Projections (NEON / AVX2 / Scalar MatMul):         │
 │      q = W_q · u   [2048]                                              │
 │      k = W_k · u   [256]                                               │
 │      v = W_v · u   [256]                                               │
 │      (src/transformer/transformer.cpp:186, dispatch.cpp:99)            │
 │                                                                        │
 │  5c. Rotary Position Embedding (RoPE):                                 │
 │      q = rope(q, pos), k = rope(k, pos) via precomputed cos/sin        │
 │      (src/kernels/scalar/kernels_scalar.cpp:50)                        │
 │                                                                        │
 │  5d. KV Cache Append (Commit Contract):                                │
 │      KvCache::append(layer_idx, pos, k, v)                             │
 │      Writes to flat contiguous buffer; does NOT advance seq_len        │
 │      (src/core/kv_cache.cpp:38)                                        │
 │                                                                        │
 │  5e. Causal Grouped Query Attention (GQA 8:1):                         │
 │      Query head h_q (0..31) reads KV head h_kv = h_q / 8 (0..3)       │
 │      Score_t = (q_hq · k_hkv,t) / sqrt(64) for t in 0..pos             │
 │      Softmax(Score_0..pos)                                             │
 │      AttnOut_hq = sum_t (Score_t * v_hkv,t)                            │
 │      (src/kernels/scalar/kernels_scalar.cpp:148)                       │
 │                                                                        │
 │  5f. Attention Output Projection & Residual Add:                       │
 │      proj = W_o · AttnOut                                              │
 │      x = x + proj                                                      │
 │      (src/transformer/transformer.cpp:215)                             │
 │                                                                        │
 │  5g. Post-Attention RMSNorm:                                           │
 │      m = x * inv_rms(x) * post_attention_layernorm.weight              │
 │      (src/transformer/transformer.cpp:221)                             │
 │                                                                        │
 │  5h. SwiGLU Feed-Forward Network & Residual Add:                       │
 │      gate = W_gate · m    [5632]                                       │
 │      up   = W_up · m      [5632]                                       │
 │      ffn_act = SiLU(gate) * up                                         │
 │      proj = W_down · ffn_act [2048]                                    │
 │      x = x + proj                                                      │
 │      (src/transformer/transformer.cpp:225, kernels_scalar.cpp:121)     │
 └─────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────┐
│ 6. Advance KV Cache Position                 │  src/core/kv_cache.cpp:64
│    KvCache::commit_token()                   │  Increments committed_seq_len by 1
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 7. Final RMSNorm & LM Head Projection        │  src/transformer/transformer.cpp:241
│    norm_out = RMSNorm(x, model.norm.weight)  │
│    logits = W_lm_head · norm_out             │  Produces [32000] unnormalized floats
└──────────────────────┬───────────────────────┘
                       │
         ══════════════╪══════════════════════════════════════════════
         PHASE B: AUTOREGRESSIVE TOKEN GENERATION & SAMPLING
         ══════════════╪══════════════════════════════════════════════
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 8. Sampler Pipeline                          │  src/sampler/sampler.cpp:30
│    - Sanitize NaNs & +/- Infinities          │
│    - Temperature scaling: logits / T         │
│    - Softmax candidate probability sorting   │
│    - Top-K truncation (e.g. top 40)          │
│    - Top-P nucleus filtering (CDF <= 0.90)   │
│    - 64-bit xorshift* PRNG random draw       │  Produces next_token ID (e.g., 3450)
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 9. Incremental Prefix Detokenization         │  src/app/main.cpp:673
│    Tokenizer::decode()                       │  Flushes new characters immediately
└──────────────────────┬───────────────────────┘  over stdout / SSE event hub
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 10. Next Decode Iteration                    │  Feed sampled token back into
│     forward_token(next_token, pos = N)       │  Step 4 at pos = N. Repeat until
└──────────────────────────────────────────────┘  EOS (2) or max_tokens reached.
```

### Detailed Trace of Every Subsystem:

#### 1. Prompt Input & Chat Formatting
- **Intuition:** Conversational models require structured role demarcation tags so the network distinguishes user instructions from system prompts and its own prior answers.
- **Mathematics / Syntax:** TinyLlama uses the Zephyr chat template:
  $$\text{Formatted} = \texttt{<|system|>}\backslash\text{n}\{\text{system}\}\texttt{</s>}\backslash\text{n}\texttt{<|user|>}\backslash\text{n}\{\text{user}\}\texttt{</s>}\backslash\text{n}\texttt{<|assistant|>}\backslash\text{n}$$
- **File & Function:** `src/tokenizer/tokenizer.cpp`, `Tokenizer::format_chat_prompt()`.
- **Inputs:** `const std::vector<ChatMessage>& messages`, `bool add_generation_prompt`.
- **Outputs:** Formatted UTF-8 string.

#### 2. SentencePiece Tokenization
- **Intuition:** Neural networks operate on fixed-size numerical vectors, not arbitrary-length variable strings. Subword tokenization breaks words into frequent character n-grams, handling punctuation and out-of-vocabulary words cleanly.
- **File & Function:** `src/tokenizer/tokenizer.cpp`, `Tokenizer::encode()`.
- **Inputs:** `const std::string& text`, `bool add_bos = true`, `bool add_eos = false`.
- **Outputs:** `std::vector<int32_t>* out_ids`.
- **Implementation Detail:** Standard SentencePiece prepends a dummy whitespace prefix `\xe2\x96\x81` to the first word. In chat templates, special tokens (`<|user|>`) sit at segment boundaries. Our implementation maintains two internal processors (`processor_` and `processor_no_prefix_`) to prevent spurious space insertion after control tokens.

#### 3. Token IDs
- **Data Structure:** Vector of signed 32-bit integers (`int32_t`), constrained to vocabulary range $[0, 31999]$. Special IDs: `<unk> = 0`, `<s>` (BOS) $= 1$, `</s>` (EOS) $= 2$.

#### 4. Embedding Lookup
- **Intuition:** Map a scalar discrete token index to a rich continuous semantic vector in hidden space.
- **Mathematics:** Select row $t$ from weight matrix $E \in \mathbb{R}^{32000 \times 2048}$:
  $$x = E[t, :] \in \mathbb{R}^{2048}$$
- **File & Function:** `src/transformer/transformer.cpp:153`, `TransformerEngine::forward_token()`.
- **Data Structure:** `x_` is a contiguous pre-allocated buffer: `std::vector<float> x_` of length 2048.
- **Handling Q4_0 Embeddings:** When embeddings are stored in `Q4_0` (`ADR-008`), the single row $t$ (consisting of $2048 / 32 = 64$ consecutive `BlockQ4_0` structs, 1152 bytes) is dequantized directly into `x_` via `dequantize_row_q4_0()`. When stored in `FP32` (`--keep-embeddings-fp32`), it is a zero-overhead `std::memcpy` of $2048 \times 4 = 8192$ bytes.

#### 5. Transformer Forward Pass (22 Layers)
- **Intuition:** Sequentially refine the token's representation through 22 identical stacked decoder layers containing attention (contextual information mixing across tokens) and feed-forward networks (factual knowledge retrieval and transformation).
- **File & Function:** `src/transformer/transformer.cpp:178`, outer loop over `layer_idx \in [0, 21]`.

#### 6. RMSNorm (Root Mean Square Normalization)
- **Intuition:** LayerNorm centers activations (subtracts mean) and scales by variance. Zhang & Sennrich (2019) demonstrated that the mean-centering operation is computationally superfluous; scaling solely by the root-mean-square achieves equivalent numerical stability with higher execution speed.
- **Mathematics:**
  $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \qquad \text{inv\_rms} = \frac{1}{\text{RMS}(x)}, \qquad \text{out}_i = (x_i \cdot \text{inv\_rms}) \cdot w_i$$
  where $d = 2048$, $\epsilon = 10^{-5}$, and $w \in \mathbb{R}^{2048}$ is the learned FP32 scaling vector (`input_layernorm.weight` or `post_attention_layernorm.weight`).
- **File & Function:** `src/kernels/scalar/kernels_scalar.cpp:9`, `scalar::rmsnorm_f32()`.
- **Inputs:** `float* out`, `const float* x`, `const float* weight`, `size_t dim = 2048`, `float eps = 1e-5f`.
- **Outputs:** Normalized activations written to `norm_buf_` (2048 floats). Sum of squares is accumulated in double precision (`double sum_sq`) to prevent underflow/overflow.

#### 7. Rotary Positional Embeddings (RoPE)
- **Intuition:** Attention dot products are naturally permutation-invariant. Traditional absolute positional embeddings (like GPT-2's learned vectors) add a position vector directly to the token embedding, degrading long-context generalization. RoPE rotates the Query and Key vectors in complex 2D planes by angles proportional to their sequence position $\text{pos}$, causing the inner product $\langle R_m q, R_n k \rangle$ to depend purely on the relative distance $m - n$.
- **Mathematics:** Split each 64-dimensional head into 32 pairs $(x_i, x_{i + 32})$ for $i \in [0, 31]$ (the Hugging Face "rotate-half" convention):
  $$\text{freq}_i = \theta^{-2i / d_{\text{head}}} = 10000^{-2i / 64}, \qquad \alpha_i = \text{pos} \cdot \text{freq}_i$$
  $$\begin{bmatrix} y_i \\ y_{i + 32} \end{bmatrix} = \begin{bmatrix} \cos\alpha_i & -\sin\alpha_i \\ \sin\alpha_i & \cos\alpha_i \end{bmatrix} \begin{bmatrix} x_i \\ x_{i + 32} \end{bmatrix} = \begin{bmatrix} x_i \cos\alpha_i - x_{i + 32} \sin\alpha_i \\ x_{i + 32} \cos\alpha_i + x_i \sin\alpha_i \end{bmatrix}$$
- **File & Function:** `src/kernels/scalar/kernels_scalar.cpp:50`, `scalar::rope_inplace_cached_f32()`.
- **Optimization (`KI-006`):** Computing `std::cos` and `std::sin` via transcendental runtime calls for $32 + 4 = 36$ heads $\times 32$ frequencies on every token is a major bottleneck. During `TransformerEngine::init()`, we precalculate all $2048 \times 32$ `cos` and `sin` values into contiguous float tables (`rope_cos_`, `rope_sin_`). At inference, RoPE is a zero-transcendental memory-streaming kernel.

#### 8. Grouped Query Attention (GQA)
- **Intuition:** In Multi-Head Attention (MHA), 32 query heads attend to 32 independent key and value heads, consuming $32 \times 64 = 2048$ floats of KV cache per token per layer. Grouped Query Attention (Ainslie et al., 2023) shares one Key/Value head across multiple Query heads (in TinyLlama, 8 Query heads share 1 KV head). This reduces KV cache size and memory traffic by an entire **factor of 8** while preserving 99%+ of full MHA model quality.
- **Mathematics:**
  - Query heads: $h_q \in [0, 31]$. Key/Value heads: $h_{kv} \in [0, 3]$.
  - Head group size: $G = 32 / 4 = 8$. Mapping: $h_{kv} = \lfloor h_q / 8 \rfloor$.
  - Query projection: $W_q \in \mathbb{R}^{2048 \times 2048} \implies q \in \mathbb{R}^{2048}$ (32 heads $\times 64$).
  - Key projection: $W_k \in \mathbb{R}^{256 \times 2048} \implies k \in \mathbb{R}^{256}$ (4 heads $\times 64$).
  - Value projection: $W_v \in \mathbb{R}^{256 \times 2048} \implies v \in \mathbb{R}^{256}$ (4 heads $\times 64$).
- **File & Function:** `src/kernels/scalar/kernels_scalar.cpp:148`, `scalar::gqa_attention_f32()`.

#### 9. Attention Score Computation & Softmax
- **Intuition:** For each query head $h_q$, measure its compatibility with all past and present key vectors $k_{h_{kv}, t}$ up to the current position $\text{pos}$, normalize into a valid probability distribution via numerically stable softmax, and compute the expectation over value vectors $v_{h_{kv}, t}$.
- **Mathematics:**
  $$S_t = \frac{q_{h_q} \cdot k_{h_{kv}, t}}{\sqrt{d_{\text{head}}}} = \frac{1}{8} \sum_{d=0}^{63} q_{h_q, d} \cdot k_{h_{kv}, t, d} \quad \text{for } t \in [0, \text{pos}]$$
  $$\text{Softmax: } m = \max_{0 \le j \le \text{pos}} S_j, \qquad P_t = \frac{\exp(S_t - m)}{\sum_{j=0}^{\text{pos}} \exp(S_j - m)}$$
  $$\text{AttnOut}_{h_q} = \sum_{t=0}^{\text{pos}} P_t \cdot v_{h_{kv}, t} \in \mathbb{R}^{64}$$
- **Data Structure:** `scores_` scratchpad of size 2048 floats.
- **File & Function:** `src/kernels/scalar/kernels_scalar.cpp:129`, `scalar::softmax_f32()` and `gqa_attention_f32()`.

#### 10. KV Cache Append & Commit
- **Storage Layout:** Flat contiguous `std::vector<float>` allocated once during engine initialization.
  $$\text{Index}(\text{layer}, \text{pos}) = (\text{layer} \times 2048 + \text{pos}) \times 256$$
- **Commit Contract (`ADR-005`):** `append()` writes the 256 floats of $k$ and 256 floats of $v$ for layer $i$ at position $\text{pos}$ and **never** advances `seq_len()`. Only after all 22 layers complete does `commit_token()` increment `committed_seq_len_` by 1.

#### 11. Feed-Forward Network (FFN) & SwiGLU Activation
- **Intuition:** Attention only mixes existing token representations. The feed-forward network acts as a key-value associative memory where learned factual knowledge is retrieved and non-linearly combined. Llama uses SwiGLU (Swish-Gated Linear Unit) rather than standard ReLU or GeLU.
- **Mathematics:**
  $$\text{gate} = W_{\text{gate}} \cdot m \in \mathbb{R}^{5632}, \qquad \text{up} = W_{\text{up}} \cdot m \in \mathbb{R}^{5632}$$
  $$\text{SiLU}(g) = \frac{g}{1 + \exp(-g)}$$
  $$\text{ffn\_act} = \text{SiLU}(\text{gate}) \odot \text{up} \in \mathbb{R}^{5632} \qquad (\odot = \text{element-wise Hadamard product})$$
  $$\text{proj} = W_{\text{down}} \cdot \text{ffn\_act} \in \mathbb{R}^{2048}, \qquad x \leftarrow x + \text{proj}$$
- **File & Function:** `src/kernels/scalar/kernels_scalar.cpp:121`, `scalar::swiglu_f32()`, invoked in `src/transformer/transformer.cpp:227`.

#### 12. Final Logits
- **Mathematics:** After 22 layers, $x$ passes through the final root-mean-square normalization and is projected onto the vocabulary:
  $$x_{\text{final}} = \text{RMSNorm}(x, \texttt{model.norm.weight}, \epsilon)$$
  $$\text{logits} = W_{\text{lm\_head}} \cdot x_{\text{final}} \in \mathbb{R}^{32000}$$
  where $W_{\text{lm\_head}} \in \mathbb{R}^{32000 \times 2048}$.
- **File & Function:** `src/transformer/transformer.cpp:241`.

#### 13. Sampling
- **Intuition:** Convert raw unbounded logits into a concrete token choice according to temperature, top-k candidate truncation, and top-p nucleus cumulative probability filtering.
- **Mathematics:**
  1. Filter out $\text{NaN}$ and $-\infty$.
  2. Scale by temperature $T$: $z_i = \frac{\text{logits}_i - \max(\text{logits})}{T}$, $w_i = \exp(z_i)$.
  3. Sort candidates descending by weight $w_i$.
  4. Top-K: Truncate candidate list to at most $K$ items (e.g. $K = 40$).
  5. Top-P (Nucleus): Retain smallest subset $V^{(p)}$ such that $\sum_{i \in V^{(p)}} w_i \ge p \cdot \sum_{\text{all}} w_i$ (e.g. $p = 0.90$).
  6. Sample from $V^{(p)}$ using uniform random draw $u \in [0, \sum_{i \in V^{(p)}} w_i)$ generated by seeded 64-bit `xorshift*` PRNG.
- **File & Function:** `src/sampler/sampler.cpp:30`, `Sampler::sample()`.

#### 14. Autoregressive Generation Loop
- **Intuition:** The Transformer decoder generates text one token at a time. The prompt is prefilled across positions $0 \dots N-1$. The logits at position $N-1$ yield the first generated token $T_{N}$. This token is fed into the engine at position $N$, producing $T_{N+1}$, and so on, until the End-Of-Sequence token (`</s>` / ID 2) or `max_new_tokens` is reached.
- **Position Contract (`ADR-006`):**
  - Prompt tokens ($N = 12$): processed at $\text{pos} = 0, 1, \dots, 11$.
  - First generated token: predicted by logits at $\text{pos} = 11$, then fed into `forward_token(T_N, pos = 12)`.
  - Second generated token: predicted by logits at $\text{pos} = 12$, then fed into `forward_token(T_{N+1}, pos = 13)`.
- **File & Function:** `src/app/main.cpp:687` and `src/app/studio_worker.hpp:892`.

#### 15. Incremental Prefix Detokenization
- **Intuition:** In UTF-8 and BPE tokenization, a single token does not always correspond to a single complete character. For instance, multi-byte emojis or split UTF-8 sequences take multiple tokens. Naively decoding individual token IDs in isolation produces replacement characters (``) or broken spacing.
- **Algorithm:** Maintain the full list of generated token IDs in `turn_generated_ids`. Decode the entire accumulated list into `current_output`. Find the length of the longest common prefix between `previous_output` and `current_output` (`common`). Print only the suffix `current_output.substr(common)`, then flush stdout immediately.
- **File & Function:** `src/app/main.cpp:673-685` and `src/app/studio_worker.hpp:912-925`.

---

# Chapter 3: TinyLlama Architecture

### 3.1 Model Hyperparameter Inventory
All values are frozen in `include/inference_engine/config.hpp` and verified against Hugging Face `TinyLlama/TinyLlama-1.1B-Chat-v1.0`:

| Hyperparameter Constant | Value | Role in Architecture |
| :--- | :--- | :--- |
| `kTinyLlamaNumLayers` | **22** | Number of stacked Transformer decoder layers |
| `kTinyLlamaHiddenDim` | **2048** | Activation vector dimensionality ($d_{\text{model}}$) |
| `kTinyLlamaIntermediateDim` | **5632** | SwiGLU hidden layer expansion ($d_{\text{ffn}}$) |
| `kTinyLlamaNumQueryHeads` | **32** | Number of query attention heads ($H_q$) |
| `kTinyLlamaNumKvHeads` | **4** | Number of key/value attention heads ($H_{kv}$) |
| `kTinyLlamaHeadDim` | **64** | Dimensionality per head ($d_{\text{head}} = 2048 / 32$) |
| `kTinyLlamaVocabSize` | **32000** | Number of tokens in SentencePiece BPE vocabulary |
| `kTinyLlamaMaxSeqLen` | **2048** | Maximum supported sequence length context window |
| `kTinyLlamaRmsNormEps` | **$10^{-5}$** (`1e-5f`) | RMSNorm numerical stabilization constant $\epsilon$ |
| `kTinyLlamaRopeTheta` | **$10000.0$** | Base angular frequency for Rotary Position Embeddings |

### 3.2 What Parameters Represent
Every parameter in the model is a single learned floating-point weight stored as a 4-bit integer index + block scale (`Q4_0`) or IEEE 754 32-bit float (`FP32`). The parameters represent the weights of linear transformations ($y = W x$) and normalizations ($y = x \odot w$) that map token vectors through semantic activation space:

$$\text{Total Parameters} = \underbrace{32000 \times 2048}_{\text{embed\_tokens}} + \underbrace{22 \times \text{LayerParams}}_{\text{22 decoder layers}} + \underbrace{2048}_{\text{final norm}} + \underbrace{32000 \times 2048}_{\text{lm\_head}}$$

$$\text{LayerParams} = \underbrace{2048}_{\text{input\_layernorm}} + \underbrace{2048 \times 2048}_{W_q} + \underbrace{256 \times 2048}_{W_k} + \underbrace{256 \times 2048}_{W_v} + \underbrace{2048 \times 2048}_{W_o} + \underbrace{2048}_{\text{post\_attn\_norm}} + \underbrace{5632 \times 2048}_{W_{\text{gate}}} + \underbrace{5632 \times 2048}_{W_{\text{up}}} + \underbrace{2048 \times 5632}_{W_{\text{down}}}$$

$$\text{LayerParams} = 2048 + 4,194,304 + 524,288 + 524,288 + 4,194,304 + 2048 + 11,534,336 + 11,534,336 + 11,534,336 = 43,444,240 \text{ params}$$

$$\text{Total Model Parameters} = 65,536,000 + (22 \times 43,444,240) + 2048 + 65,536,000 = \mathbf{1,100,048,384 \text{ parameters}}$$
This matches the exact 201-tensor count and 1.100 billion parameter specification of TinyLlama.

### 3.3 Why 32 Query Heads but Only 4 KV Heads? (GQA Intuition)
In autoregressive text generation, the query vector is computed **only for the current token being processed** at position $\text{pos}$ (shape $[1, 2048]$). In contrast, the key and value vectors must be **stored for every token in the entire prompt and previous history** in the KV cache.

If TinyLlama used standard Multi-Head Attention (32 Q heads, 32 KV heads):
$$\text{KV Cache Size per Token per Layer} = 2 \times 32 \times 64 \times 4\text{ bytes} = 16,384\text{ bytes}$$
$$\text{Full 22-Layer KV Cache at 2048 Context} = 22 \times 2048 \times 16,384\text{ bytes} \approx \mathbf{738.2\text{ MB}}$$

By using Grouped Query Attention with 4 KV heads ($8:1$ head sharing):
$$\text{KV Cache Size per Token per Layer} = 2 \times 4 \times 64 \times 4\text{ bytes} = 2,048\text{ bytes}$$
$$\text{Full 22-Layer KV Cache at 2048 Context} = 22 \times 2048 \times 2,048\text{ bytes} = \mathbf{92,274,688\text{ bytes}} \approx \mathbf{88.0\text{ MB}}$$
*(In our engine's contiguous float buffer, this is stored as two $22 \times 2048 \times 256$ arrays = $45.09\text{ MB}$ for K and $45.09\text{ MB}$ for V = **$90.18\text{ MB}$ total**).*

GQA slashes the memory bandwidth demanded to stream the KV cache from RAM on every decode step by **87.5%**, directly unlocking interactive decoding speeds on memory-constrained laptop CPUs.

### 3.4 Tensor Shapes Throughout the Forward Pass
Here is the exact progression of tensor dimensions during a single token evaluation `forward_token(token_id, pos, out_logits)`:

| Step | Operation | Input Tensor(s) & Shape | Output Tensor & Shape | FLOPs ($2 \times M \times K$) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Embedding Lookup | Scalar `token_id` | `x`: $[2048]$ | 0 (Memory read/dequant) |
| **2a** | Pre-Attn RMSNorm | `x`: $[2048]$, `w`: $[2048]$ | `norm_buf`: $[2048]$ | $\approx 3 \times 2048 \approx 6.1\text{ KFLOPs}$ |
| **2b** | $W_q$ Projection | `norm_buf`: $[2048]$, $W_q$: $[2048, 2048]$ | `q`: $[2048]$ | $2 \times 2048 \times 2048 \approx 8.39\text{ MFLOPs}$ |
| **2b** | $W_k$ Projection | `norm_buf`: $[2048]$, $W_k$: $[256, 2048]$ | `k`: $[256]$ | $2 \times 256 \times 2048 \approx 1.05\text{ MFLOPs}$ |
| **2b** | $W_v$ Projection | `norm_buf`: $[2048]$, $W_v$: $[256, 2048]$ | `v`: $[256]$ | $2 \times 256 \times 2048 \approx 1.05\text{ MFLOPs}$ |
| **2c** | RoPE on Q & K | `q`: $[32, 64]$, `k`: $[4, 64]$ | `q`: $[32, 64]$, `k`: $[4, 64]$ | $36 \times 64 \times 4 \approx 9.2\text{ KFLOPs}$ |
| **2d** | KV Cache Write | `k`: $[256]$, `v`: $[256]$ | `kv_cache[layer, pos]` | 0 (Direct memory copy) |
| **2e** | Attention Dot Product | `q`: $[32, 64]$, `K_cache`: $[\text{pos}+1, 4, 64]$ | `scores`: $[32, \text{pos}+1]$ | $32 \times (\text{pos}+1) \times 128\text{ FLOPs}$ |
| **2e** | Softmax | `scores`: $[32, \text{pos}+1]$ | `scores`: $[32, \text{pos}+1]$ | $32 \times (\text{pos}+1) \times 3\text{ FLOPs}$ |
| **2e** | Value Aggregation | `scores`: $[32, \text{pos}+1]$, `V_cache` | `attn_out`: $[32, 64] = [2048]$ | $32 \times (\text{pos}+1) \times 128\text{ FLOPs}$ |
| **2f** | $W_o$ Projection | `attn_out`: $[2048]$, $W_o$: $[2048, 2048]$ | `proj_out`: $[2048]$ | $2 \times 2048 \times 2048 \approx 8.39\text{ MFLOPs}$ |
| **2f** | Residual Add 1 | `x`: $[2048]$, `proj_out`: $[2048]$ | `x`: $[2048]$ | $2048\text{ FLOPs}$ |
| **2g** | Post-Attn RMSNorm | `x`: $[2048]$, `w`: $[2048]$ | `norm_buf`: $[2048]$ | $\approx 6.1\text{ KFLOPs}$ |
| **2h** | $W_{\text{gate}}$ Projection | `norm_buf`: $[2048]$, $W_g$: $[5632, 2048]$ | `gate`: $[5632]$ | $2 \times 5632 \times 2048 \approx 23.07\text{ MFLOPs}$ |
| **2h** | $W_{\text{up}}$ Projection | `norm_buf`: $[2048]$, $W_u$: $[5632, 2048]$ | `up`: $[5632]$ | $2 \times 5632 \times 2048 \approx 23.07\text{ MFLOPs}$ |
| **2h** | SwiGLU Activation | `gate`: $[5632]$, `up`: $[5632]$ | `ffn_act`: $[5632]$ | $\approx 5632 \times 4 \approx 22.5\text{ KFLOPs}$ |
| **2h** | $W_{\text{down}}$ Projection | `ffn_act`: $[5632]$, $W_d$: $[2048, 5632]$ | `proj_out`: $[2048]$ | $2 \times 2048 \times 5632 \approx 23.07\text{ MFLOPs}$ |
| **2h** | Residual Add 2 | `x`: $[2048]$, `proj_out`: $[2048]$ | `x`: $[2048]$ | $2048\text{ FLOPs}$ |
| **End**| **Total per Layer** | | | $\approx \mathbf{88.09\text{ MFLOPs}}$ |
| **3** | All 22 Layers | | `x`: $[2048]$ | $22 \times 88.09 \approx \mathbf{1.938\text{ GFLOPs}}$ |
| **4** | Final RMSNorm | `x`: $[2048]$, `w`: $[2048]$ | `norm_buf`: $[2048]$ | $6.1\text{ KFLOPs}$ |
| **4** | $W_{\text{lm\_head}}$ Projection| `norm_buf`: $[2048]$, $W_{\text{head}}$: $[32000, 2048]$| `logits`: $[32000]$ | $2 \times 32000 \times 2048 \approx \mathbf{131.07\text{ MFLOPs}}$ |
| **Total**| **Single Decode Token** | | | $\approx \mathbf{2.069\text{ GFLOPs}}$ |

---

# Chapter 4: Q4_0 Quantization

### 4.1 Why FP32 Uses 32 Bits per Weight
Standard single-precision IEEE 754 floating-point (`FP32`) represents real numbers with 1 sign bit, 8 exponent bits (bias 127), and 23 mantissa/fraction bits (yielding ~7 decimal digits of precision). In TinyLlama:
$$1.10\text{ billion weights} \times 4\text{ bytes} = \mathbf{4.40\text{ GB}}$$
On commodity laptop CPUs with dual-channel LPDDR4/LPDDR5 memory offering 30–60 GB/s practical bandwidth, streaming 4.40 GB across the memory bus to compute a single output token limits decode throughput to at most $40 / 4.40 \approx 9\text{ tokens/sec}$. Furthermore, on an 8 GB RAM laptop, a 4.40 GB model leaves almost no OS headroom, inducing memory paging.

### 4.2 Why Q4_0 Uses 4-Bit Values
Weights in deep neural networks are not uniformly distributed over $\mathbb{R}$; they are bell-shaped and centered near zero with small variance. Representing each weight with a 4-bit unsigned integer ($q \in [0, 15]$) mapped symmetrically around zero with a per-block scaling factor $d$ compresses 32 weights into just 18 bytes.

### 4.3 How Blocks of 32 Weights Are Represented
Weights along the innermost matrix row dimension are grouped into contiguous blocks of $B = 32$.
Each block is stored in the packed C++ struct `BlockQ4_0` (`include/inference_engine/quant.hpp:23`):
```cpp
#pragma pack(push, 1)
struct BlockQ4_0 {
    uint16_t d;                 // 2 bytes: IEEE 754 half-precision (binary16) scale factor
    uint8_t qs[16];             // 16 bytes: 32 packed 4-bit unsigned integers
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be exactly 18 bytes");
```
$$\text{Effective bits per weight} = \frac{18\text{ bytes} \times 8\text{ bits}}{32\text{ weights}} = \frac{144}{32} = \mathbf{4.5\text{ bits/weight}}$$

### 4.4 How the FP16 Scale is Stored
The scale $d$ is an IEEE 754 binary16 (half-precision) floating-point number stored in little-endian order. It uses 1 sign bit, 5 exponent bits (bias 15), and 10 mantissa bits.
- To maintain zero external dependencies and guaranteed portability across all compilers (even where native `_Float16` or `__fp16` is unsupported), we implement bitwise IEEE 754 half-precision encoders and decoders in `src/kernels/quant_q4_0.cpp`:
  - `fp32_to_fp16(float value)`: Extracts sign, exponent, and mantissa from 32-bit float; handles denormals, rounding bit accumulation, and infinity saturation.
  - `fp16_to_fp32(uint16_t bits)`: Bitwise unpacks the 16-bit word, re-biases the exponent by $+112$, normalizes denormals via bit-shifting, and emits a standard IEEE 754 32-bit float.

### 4.5 How Two 4-Bit Values Fit into One Byte (Split-Half Layout)
Many naive 4-bit formats store adjacent weights in the same byte: low nibble $= w_{2j}$, high nibble $= w_{2j+1}$.
`inference-engine` adopts the **split-half nibble layout** (`ADR-004`), which is upstream-compatible with GGML/`llama.cpp`:
- For each byte index $j \in [0, 15]$:
  - **Low Nibble** (`qs[j] & 0x0F`): Stores weight $j$ (indices $0 \dots 15$).
  - **High Nibble** (`(qs[j] >> 4) & 0x0F`): Stores weight $j + 16$ (indices $16 \dots 31$).

```
Byte index j (0..15):
┌───────────────────────────┬───────────────────────────┐
│  Bits 7..4 (High Nibble)   │   Bits 3..0 (Low Nibble)  │
│   Stores Weight j + 16    │      Stores Weight j      │
│     (Range: 16..31)       │       (Range: 0..15)      │
└───────────────────────────┴───────────────────────────┘
```
**Why Split-Half Layout Matters for SIMD:**
A single 16-byte unaligned vector load (`vld1q_u8` on ARM NEON or `_mm_loadu_si128` on AVX2) loads `qs[0..15]` into a vector register. With a single vector shift (`vshrq_n_u8` / `_mm_srli_epi16`) and vector mask (`vandq_u8` with `0x0F`), all 16 low weights and all 16 high weights are separated in two assembly instructions without costly element-by-element bit manipulation.

### 4.6 Exact Quantization and Dequantization Formulas

#### Quantization (`src/kernels/quant_q4_0.cpp:78`):
1. Find the element with maximum absolute magnitude in the 32-weight slice:
   $$v_{\max} = \arg\max_{v \in \text{block}} |v|, \qquad \text{signed\_max} = v_{\max}$$
2. Compute the scale factor $d$:
   $$d = \frac{\text{signed\_max}}{-8.0f}$$
   *(If $d = 0$, the block is all zeros; set $d = 0$ and $q = 8$).*
3. Compute reciprocal scale: $\text{id} = 1.0f / d$.
4. Quantize each element $x$ into 4-bit unsigned integer $q \in [0, 15]$:
   $$q = \text{clamp}\left(\left\lfloor x \cdot \text{id} + 8.5f \right\rfloor, 0, 15\right)$$
5. Store $d$ as `fp32_to_fp16(d)` and pack nibbles:
   $$\text{qs}[j] = (q_j \ \& \ \text{0x0F}) \ | \ ((q_{j+16} \ \& \ \text{0x0F}) \ll 4)$$

#### Dequantization (`src/kernels/quant_q4_0.cpp:121`):
$$w_j = \left((\text{qs}[j] \ \& \ \text{0x0F}) - 8\right) \times \text{fp16\_to\_fp32}(d) \quad \text{for } j \in [0, 15]$$
$$w_{j+16} = \left(((\text{qs}[j] \gg 4) \ \& \ \text{0x0F}) - 8\right) \times \text{fp16\_to\_fp32}(d) \quad \text{for } j \in [0, 15]$$

### 4.7 What Precision Do We Lose?
We evaluated real quantization error across all 201 weights of `model.safetensors` using `tools/inspect_checkpoint.py --eval-real-quant-error`:
- **Cosine Similarity:** Mean Cosine Similarity $\ge \mathbf{0.9888}$ across all functional tensor projections (Attention Q/K/V/O, MLP Gate/Up/Down).
- **Signal-to-Quantization-Noise Ratio (SQNR):** Averages **$16.51 \dots 16.82\text{ dB}$**.
- **Per-Element Errors:** Mean Absolute Error (MAE) $\approx 5.3 \times 10^{-4}$, Root Mean Square Error (RMSE) $\approx 8.6 \times 10^{-4}$.
- **End-to-End Forward Parity:** Full 22-layer forward output logits between pure FP32 and Q4_0 achieve a cosine similarity of $\mathbf{0.996359}$ (`tests/test_transformer_q4.cpp`).

### 4.8 Why is the Model ~7.1x Smaller Rather than Exactly 8x?
Judges frequently ask: *"If 32 bits divided by 4 bits is 8, why is your model 7.1x smaller instead of 8x?"*
Three concrete mathematical reasons from our binary file layout:
1. **The Scale Factor Overhead:** Each 32-weight block requires an 18-byte struct (16 bytes for weights + 2 bytes for FP16 scale). This means each weight takes 4.5 bits, not 4.0 bits ($32 / 4.5 = 7.11\times$).
2. **RMSNorm Weights Are Kept in FP32 (`ADR-008`):** All 45 1D normalization tensors (`input_layernorm`, `post_attention_layernorm`, `model.norm.weight`, total 368,640 bytes) are kept in full 32-bit floating point. Quantizing 1D normalization vectors introduces severe scale distortion while saving less than 300 KB.
3. **Format Metadata & Padding:** The 64-byte header, tensor directory entries, and 64-byte inter-tensor alignment padding add minor constant byte overhead.
- **Exact File Sizes:**
  - `tinyllama-1.1b-fp32.ie`: **4,400,213,312 bytes** (4.40 GB)
  - `tinyllama-1.1b-q4_0.ie`: **619,113,792 bytes** (619.11 MB)
  - Ratio: $4,400,213,312 / 619,113,792 = \mathbf{7.1073\times}$.

### 4.9 How the Converter Produces the `.ie` File
`tools/convert_to_inference_engine.py` converts Hugging Face `model.safetensors` to `.ie`:
1. Opens `model.safetensors` and parses the JSON metadata header to obtain tensor names, shapes, and byte offsets.
2. Casts source `BF16` weights to `FP32` NumPy arrays.
3. Quantizes weights in-place row-by-row into `BlockQ4_0` structs.
4. Serializes the `.ie` 64-byte header, variable-length directory, and payload bytes in a single streaming pass using checked byte alignments.

### 4.10 How the Runtime Consumes Quantized Values
The runtime **never** dequantizes entire weight matrices into separate heap memory. That would defeat the entire purpose of quantization by inflating RAM back to 4.4 GB.
Instead, dequantization is **fused directly into the SIMD inner product loop**:
- The vector registers load `BlockQ4_0` structs directly from the OS page cache via `mmap`.
- The nibbles are unpacked and converted to floats in CPU vector registers.
- Vectorized multiply-accumulate computes the inner product with activation vectors.
- Raw weights are discarded immediately from CPU registers. RAM allocation remains ~658 MiB throughout execution.

---

# Chapter 5: Custom `.ie` Model Format and Loader

### 5.1 Binary File Layout Specification
The `.ie` format (version 1) is a strictly little-endian, binary-aligned format defined in `include/inference_engine/loader.hpp`:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. FIXED FILE HEADER (64 Bytes)                                        │
│    Bytes 0..3:   Magic 0x464D4549 ("IEMF")                             │
│    Bytes 4..7:   Format Version (1)                                    │
│    Bytes 8..11:  Tensor Count (201 for TinyLlama)                      │
│    Bytes 12..15: Weight DataType (0=FP32, 1=FP16, 2=Q4_0, 3=I32)        │
│    Bytes 16..47: 8 x u32 Hyperparameters:                              │
│                  num_layers (22), hidden_dim (2048), intermediate (5632)│
│                  num_q_heads (32), num_kv_heads (4), head_dim (64)     │
│                  vocab_size (32000), max_seq_len (2048)                │
│    Bytes 48..55: 2 x f32: rms_norm_eps (1e-5), rope_theta (10000.0)    │
│    Bytes 56..63: u64 total_file_bytes                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. VARIABLE-LENGTH TENSOR DIRECTORY (201 Entries)                      │
│    For each tensor i in 0 .. tensor_count - 1:                         │
│      name_len:        u32 (1..256)                                     │
│      name:            UTF-8 string (name_len bytes, non-null-terminated)│
│      dtype:           u32 (0=FP32, 2=Q4_0)                             │
│      ndim:            u32 (1..4)                                       │
│      shape:           4 x u64 (dimensions; unused trailing dims = 0)   │
│      payload_offset:  u64 (MUST be 64-byte aligned: offset % 64 == 0)   │
│      payload_nbytes:  u64 (exact byte count of payload)                │
├────────────────────────────────────────────────────────────────────────┤
│ 3. PADDING TO 64-BYTE BOUNDARY                                         │
│    Aligns directory end to nearest multiple of 64 bytes                │
├────────────────────────────────────────────────────────────────────────┤
│ 4. 64-BYTE ALIGNED TENSOR PAYLOADS                                     │
│    Tensor 0:   model.embed_tokens.weight [32000, 2048]                 │
│                Offset: min_payload_offset (% 64 == 0)                  │
│    Tensor 1:   model.layers.0.input_layernorm.weight [2048] (FP32)     │
│                Offset: aligned (% 64 == 0)                             │
│    ...                                                                 │
│    Tensor 200: lm_head.weight [32000, 2048]                            │
│                End of payload == total_file_bytes                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Strict Validation and Error Handling
`ModelLoader::load_from_buffer()` (`src/loader/loader.cpp:133`) executes a multi-stage validation gauntlet before accepting any model image:
1. **Buffer Bounds:** Ensures buffer size $\ge 64$ bytes.
2. **Magic & Version:** Validates magic == `0x464D4549` and version == `1`.
3. **Hyperparameter Sanity:** Checks dimensions $> 0$, $H_q \% H_{kv} == 0$, $H_q \times d_{\text{head}} == d_{\text{model}}$, $d_{\text{head}} \% 2 == 0$, $\epsilon > 0$, $\theta > 0$. Rejects non-finite floats (`NaN`, $\infty$).
4. **File Length Integrity:** Verifies `header.total_file_bytes == actual_buffer_size`.
5. **Checked 64-Bit Arithmetic:** Uses `checked_add_u64()` and `checked_mul_u64()` on all pointer and offset calculations to prevent 64-bit integer overflow vulnerabilities.
6. **Name & Duplicate Validation:** Requires $1 \le \text{name\_len} \le 256$, rejects duplicate tensor names via hash set tracking.
7. **Payload Alignment:** Enforces `payload_offset % 64 == 0` on every tensor entry.
8. **Shape Divisibility:** For `Q4_0` tensors, enforces that the innermost dimension `shape[ndim - 1] % 32 == 0` and total elements $\% 32 == 0$.
9. **Sorted Interval Non-Overlap Check:** Collects `[payload_offset, payload_offset + payload_nbytes)` intervals for all tensors, sorts them, and verifies:
   - No interval starts before `min_payload_offset` (preventing payload corruption of the header or directory).
   - No two tensor intervals overlap: `range[i].begin >= range[i-1].end`.
   - The final interval ends exactly at `total_file_bytes` (preventing unreferenced trailing bytes or hidden payloads).
10. **Full Inventory Check:** `ModelLoader::bind_transformer_weights()` verifies that all $3 + 22 \times 9 = 201$ canonical Llama tensors are present with matching dimensions and expected data types.

### 5.3 Memory Mapping (`mmap`) & Zero-Copy Architecture
- `ModelLoader::load_from_file()` uses POSIX `::mmap()` (`PROT_READ`, `MAP_PRIVATE`, `fd`, 0) on macOS/Linux.
- **Zero-Copy Semantics:** `mmap` assigns the model file into the process's 64-bit virtual address space. **No physical heap memory is allocated** to load the model.
- **Demand Paging:** The operating system's virtual memory manager lazily loads 4 KB pages from disk or SSD into the OS page cache only when the CPU actually touches those addresses during matrix multiplication.
- **Model Loading Latency:** Because `mmap` merely sets up page table entries, model loading time is virtually instantaneous: **0.34 ms to 3.7 ms** for a 619 MB model file.

### 5.4 `TensorView`
`TensorView` (`include/inference_engine/tensor.hpp:33`) is a lightweight, non-owning view struct wrapping a `TensorDesc` metadata descriptor and a `const uint8_t*` pointer to the memory-mapped payload:
- It allocates zero dynamic heap memory.
- It exposes strongly typed accessor methods:
  - `data_f32()`: Returns `const float*` if `dtype == DataType::FP32`, else `nullptr`.
  - `data_q4_0()`: Returns `const BlockQ4_0*` if `dtype == DataType::Q4_0`, else `nullptr`.
  - `shape()`, `ndim()`, `num_elements()`, `num_blocks()`.
- **Lifetime Contract:** A `TensorView` is valid only for the lifetime of its backing `ModelLoader`.

---

# Chapter 6: CPU Optimisation

### 6.1 Kernel Architecture & Hierarchy
The engine organizes CPU compute into a clean 3-tier hierarchy:
1. **Scalar Reference Layer (`src/kernels/scalar/kernels_scalar.cpp`):** Platform-agnostic C++17 scalar loops. Implements numerical ground truth. Used on architectures without SIMD support or as fallback.
2. **SIMD Vector Layer (`src/kernels/neon/`, `src/kernels/avx2/`):** Platform-specific vector assembly intrinsics utilizing 128-bit NEON registers (ARM64) or 256-bit YMM registers (x86-64).
3. **Dispatch Layer (`src/kernels/dispatch.cpp`):** Dynamic runtime CPU feature detection (`detect_best_backend()`) and multi-core work scheduling via `ThreadPool`.

### 6.2 ARM NEON Optimizations (`src/kernels/neon/kernels_neon.cpp`)
Targeting Apple Silicon M-series and ARMv8-A/ARMv9-A processors:
- **Vector Registers:** 32 128-bit NEON registers (`v0` through `v31`).
- **Fused Dequantize-and-Dot Product:**
  1. `vld1q_u8(block.qs)` loads 16 packed bytes.
  2. `vandq_u8(raw, 0x0F)` isolates low nibbles; `vshrq_n_u8(raw, 4)` isolates high nibbles.
  3. `vsubq_s8(..., 8)` subtracts 8 to produce signed 8-bit integers $[-8, +7]$.
  4. Widens `int8` $\to$ `int16` (`vmovl_s8`) $\to$ `int32` (`vmovl_s16`) $\to$ `float32` (`vcvtq_f32_s32`).
  5. Computes vector multiply-accumulate `vfmaq_f32` against activation vector slices loaded with `vld1q_f32`.
- **Vectorized Scale Accumulation (`ADR-012`):** Rather than calling horizontal addition `vaddvq_f32` inside every 32-weight block (which incurs ~14 cycles of shuffle stall 64 times per row), unscaled block sums are accumulated into vector register `row_acc` across all blocks, and scaled by `vdupq_n_f32(fp16_to_fp32(d))`. A single `vaddvq_f32` reduction executes at the end of the entire row.
- **2-Row Register Tiling (`matmul_q4_0_f32`):**
  Output rows are processed in pairs ($m$ and $m+1$). The 32 floats of activation vector $x$ are loaded **once** into NEON registers (`x0` through `x7`) and reused to compute dot products against both Row $m$ and Row $m+1$. This cuts activation vector L1/L2 load port bandwidth by **50%**.

### 6.3 x86 AVX2 + FMA Optimizations (`src/kernels/avx2/kernels_avx2.cpp`)
Targeting modern Intel and AMD x86-64 laptops:
- **Vector Registers:** 16 256-bit YMM registers (`ymm0` through `ymm15`).
- **CPUID Runtime Detection:**
  On GCC/Clang: Calls `__builtin_cpu_init()`, checks `__builtin_cpu_supports("avx2")` and `__builtin_cpu_supports("fma")`.
  On MSVC: Executes Leaf 1 and Leaf 7 `__cpuid` checks and verifies OS YMM state saving via `_xgetbv(0)`.
- **Fast Unscaled Vector Dot (`compute_block_unscaled_dot_vec`):**
  1. `_mm_loadu_si128` loads 16 packed bytes.
  2. Low and high nibbles are masked, subtracted by 8, and sign-extended to 32-bit floats using `_mm256_cvtepi8_epi32` and `_mm256_cvtepi32_ps`.
  3. Fused multiply-add `_mm256_fmadd_ps` accumulates inner products with 8-wide float activations `_mm256_loadu_ps`.
  4. Scaled by `_mm256_set1_ps(scale)` and accumulated in YMM registers.
- **Horizontal Sum:** Single end-of-row `hsum256_ps()` extracts 128-bit halves, folds them with `_mm_add_ps`, shuffles, and emits scalar float in registers.
- **2-Row Register Tiling:** Pairs rows $m$ and $m+1$, holding 4 YMM registers of $x$ loaded once, utilizing 12 of the 16 available YMM registers with zero register spilling.

### 6.4 Multithreading & ThreadPool Architecture
`ThreadPool` (`include/inference_engine/thread_pool.hpp`, `src/kernels/threading/thread_pool.cpp`):
- **Zero Thread Creation Overhead:** Spawns `num_threads - 1` worker threads in its constructor that stay alive and idle on condition variable `cv_start_`.
- **Calling Thread Participation:** Thread 0 (the main thread) actively participates in running task chunks. It does not sit idle blocking on workers.
- **Dynamic Task Chunking (`ADR-010`):** Instead of synchronizing on every individual row (which causes massive mutex contention across 32,000 rows), chunk size is dynamically computed:
  $$\text{chunk\_size} = \max\left(1, \frac{\text{num\_tasks}}{\text{num\_threads} \times 4}\right)$$
  This eliminates 95%+ of atomic lock acquisitions while preserving balanced load distribution across heterogeneous cores (e.g. Apple Silicon Performance/Efficiency cores).
- **Contiguous 2-Row Chunk Slicing (`dispatch.cpp:37`):** Matrix rows are sliced in contiguous even-sized blocks (`rows_per_task`) aligned to 2-row boundaries, allowing each thread worker to execute 2-row tiled SIMD kernels directly.
- **Adaptive Small-Matrix SMT Cap (`ADR-012`):** For small projections (such as Attention K/V where $\text{out\_dim} \le 256$, executing in $<15\ \mu\text{s}$), overscheduling 16 SMT threads incurs thread wakeup jitter that exceeds compute time. The scheduler caps tasks to physical cores ($\le 8$) and enforces a minimum chunk of 16 rows.
- **Exception Safety (`ADR-010`):** Worker callbacks are wrapped in `try / catch`. If any thread throws, `std::exception_ptr` captures the first exception, remaining work halts safely, workers synchronize back to idle, and `std::rethrow_exception` rethrows the error on the calling thread.

### 6.5 Memory Locality & Cache Behaviour
During single-token autoregressive decoding ($B = 1$), matrix multiplication is strictly memory-bandwidth bound:
- Matrix weights $W$ are streamed linearly from RAM/page cache once per token.
- Hardware stream prefetchers on modern CPUs (Zen 4, Intel Raptor Lake/Meteor Lake, Apple M-series) reliably predict linear row-major strides. Manual software prefetching (`_mm_prefetch`) was rigorously evaluated on AMD Ryzen 7 7745HX (`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`) and proven to degrade throughput by 10–23% due to load queue pollution and front-end issue saturation. It was deliberately omitted (`ADR-012`).
- Intermediate activations (`x_`, `norm_buf_`, `gate_`, `up_`, `ffn_act_`) total ~393 KB, fitting entirely in L2 cache (1–2 MB per core). They never spill to system RAM.

---

# Chapter 7: KV Cache and Autoregressive Generation

### 7.1 Why Repeatedly Computing the Entire Prompt is Wasteful
In causal language models, token $i$ can only attend to tokens $j \le i$.
When generating token $N+1$, the key and value representations of tokens $0 \dots N$ do not change.
If an engine does not cache Keys and Values:
- Generating token 1 requires evaluating 1 token.
- Generating token 2 requires re-evaluating tokens $1 \dots 2$.
- Generating token $M$ requires evaluating tokens $1 \dots M$.
- Total compute scales quadratically: $O(M^2)$.
With Key-Value caching, past keys and values are retained in RAM. Each new token computes K and V **only for itself** and appends them to the cache. Total decode compute per token is strictly $O(1)$ with respect to model weights and $O(N)$ with respect to attention dot products.

### 7.2 KV Cache Storage & Growth
- Allocated once during initialization as a contiguous float buffer:
  $$2 \times \text{num\_layers} \times \text{max\_seq\_len} \times \text{kv\_dim} \times 4\text{ bytes}$$
  $$= 2 \times 22 \times 2048 \times 256 \times 4 = \mathbf{92,274,688\text{ bytes}} \approx \mathbf{88.0\text{ MB}}$$
- **Growth Semantics:** The cache does not resize or reallocate dynamically on the heap. It uses fixed pre-allocated capacity up to `max_seq_len` (2048). As tokens arrive, `committed_seq_len_` monotonically increases from $0 \to 2048$.

### 7.3 Multi-Turn Prefix Matching & Reuse (`ADR-013`)
In conversational interactions, the conversation history grows with each turn:
- **Turn 1:** `[System Prompt + User Question 1]` $\to$ Generate `[Answer 1]`
- **Turn 2:** `[System Prompt + User Question 1 + Answer 1 + User Question 2]` $\to$ Generate `[Answer 2]`
If Turn 2 re-evaluates all tokens from scratch, the user experiences increasing latency on every conversational turn.

#### Prefix Matching Algorithm (`src/app/main.cpp:622`):
1. The engine tracks `committed_tokens`, recording the exact token IDs currently cached in `KvCache`.
2. When a new turn begins, the entire formatted chat history is tokenized into `full_tokens`.
3. The engine finds the longest common prefix:
   ```cpp
   size_t match_len = 0;
   while (match_len < committed_tokens.size() &&
          match_len < full_tokens.size() &&
          committed_tokens[match_len] == full_tokens[match_len]) {
       ++match_len;
   }
   ```
4. If `match_len < committed_tokens.size()`, the cache is rewound via `KvCache::truncate(match_len)`.
5. Prefill executes **only for the new tokens** from index `match_len` to `full_tokens.size() - 1`.
- **Measured Benefit:** On Turn 2 of a real conversation, **30+ tokens are reused instantly**. Prefill time drops from ~550 ms to $<30\text{ ms}$, delivering instantaneous interactive responses.

### 7.4 Context Limit & `/clear` Semantics
- When sequence length reaches `max_seq_len` (2048), `forward_token()` returns `StatusCode::OutOfBounds`. The CLI and Studio display a warning preventing buffer overflows.
- Typing `/clear` or sending the `clear` JSON command resets conversational state:
  - Invokes `KvCache::clear()`, setting `committed_seq_len_ = 0` and zeroing memory.
  - Clears `committed_tokens` and chat history vectors.

---

# Chapter 8: Inference Studio Architecture

### 8.1 Complete Request Path
```
┌────────────────────────┐
│  React 19 Frontend     │  studio/src/components/ChatPlayground.tsx
│  (Tailwind CSS UI)     │  Submits prompt via fetch('/api/chat')
└───────────┬────────────┘
            │ HTTP POST JSON { messages, temperature, top_p, top_k }
            ▼
┌────────────────────────┐
│  Node.js API Gateway   │  studio/server/index.ts
│  (Native HTTP Server)  │  Receives POST, formats JSON-lines command
└───────────┬────────────┘
            │ Writes single-line JSON to stdin pipe: {"cmd":"chat", ...}\n
            ▼
┌────────────────────────┐
│  Persistent C++ Worker │  src/app/studio_worker.hpp (inference-engine --studio-worker)
│  (Mmap Model Resident) │  Background stdin_reader_loop parses JSON command
└───────────┬────────────┘
            │
            ├─► Executes KvCache prefix matching
            ├─► Prefills remaining tokens
            ├─► Autoregressively generates tokens
            │
            │ Streams line-delimited JSON events to stdout pipe:
            │ {"event":"generation_started", ...}\n
            │ {"event":"token","token_id":3450,"text":"Entropy", ...}\n
            │ {"event":"generation_completed", ...}\n
            ▼
┌────────────────────────┐
│  Node.js Process Supv  │  studio/server/engineManager.ts
│  (readline interface)  │  Parses JSON events, pushes to SSE event emitter
└───────────┬────────────┘
            │ HTTP Server-Sent Events (SSE) stream (/api/events)
            ▼
┌────────────────────────┐
│  React Observability   │  studio/src/components/ObservabilityPanel.tsx
│  Workstation           │  - Renders token in markdown chat stream
│                        │  - Updates real-time SVG rolling throughput chart
│                        │  - Updates live TTFT, TPS, and RSS gauges
└────────────────────────┘
```

### 8.2 Why a Persistent Worker? (Why Not Launch a New Process?)
Spawning `./build/inference-engine` anew for every user prompt causes severe overhead:
1. **Loss of Memory Residency:** Even with `mmap`, re-opening file descriptors, parsing the 201-tensor directory, allocating thread pools, and verifying checksums consumes 50–100 ms of CPU cold-start latency per prompt.
2. **Total Destruction of the KV Cache:** When a process exits, its address space is reclaimed by the operating system. Conversational KV cache prefix reuse becomes impossible; every follow-up turn would have to re-evaluate the entire multi-turn conversation from scratch ($O(N^2)$ prefill compute).
The persistent worker loads the model **once** at startup (taking ~0.34 ms), pre-allocates the KV cache, and remains resident indefinitely.

### 8.3 Stdin/Stdout Communication & Protocol Purity
- **Bidirectional Pipes:** Communication between Node.js and C++ uses standard OS file descriptor pipes (`stdin` for inbound commands, `stdout` for outbound telemetry events, `stderr` for process crash logs).
- **Strict Protocol Purity:** Standard output is the machine-to-machine protocol wire. **No raw text or uncontrolled `printf` statements are ever written to stdout.** Any log message, warning, or diagnostic inside the C++ engine is encapsulated into a JSON event:
  ```json
  {"event":"log","level":"INFO","message":"Model loaded successfully in 0.342 ms"}
  ```
  This guarantees that the Node.js JSON parser never encounters corrupt or unexpected stream tokens.

### 8.4 Event Schema
Every event emitted by the worker is a single-line JSON string:
- `engine_ready`: Emitted on startup with format (`Q4_0`), architecture, and hardware capabilities.
- `generation_started`: Emitted when prompt prefill finishes, including prompt token count and TTFT.
- `token`: Emitted on every generated token with `token_id`, `text`, `decode_tps`, and current OS `rss_bytes`.
- `generation_completed`: Emitted at EOS or max tokens with final statistics (total tokens, prefill TPS, decode TPS, peak RSS, CPU utilization).
- `generation_cancelled`: Emitted if the user clicks "Stop" or triggers `/api/chat/stop`.
- `benchmark_progress` & `benchmark_completed`: Emitted during multi-run statistical benchmarking.

### 8.5 Cancellation Architecture
When a user clicks "Stop Generation" in the UI:
1. React sends `POST /api/chat/stop`.
2. Node.js writes `{"cmd":"stop"}\n` directly to the C++ worker's stdin pipe.
3. In `studio_worker.hpp`, the background `stdin_reader_loop` intercepts `stop` immediately and sets an atomic flag:
   ```cpp
   stop_requested_.store(true, std::memory_order_relaxed);
   ```
4. The token generation loop checks `stop_requested_` on every single token iteration. Generation halts cleanly within $<1\text{ ms}$ without killing the process or corrupting the KV cache.

### 8.6 Real OS Telemetry Instrumentation
Telemetry numbers shown in Inference Studio are **real operating system metrics**, not approximations:
- **Resident Set Size (RSS):**
  - macOS (Darwin): Reads `mach_task_basic_info_data_t.resident_size` via `task_info(mach_task_self(), MACH_TASK_BASIC_INFO, ...)`.
  - Linux: Reads `/proc/self/statm` second column (resident pages) multiplied by `sysconf(_SC_PAGESIZE)`.
  - Windows: Reads `PROCESS_MEMORY_COUNTERS.WorkingSetSize` via `GetProcessMemoryInfo()`.
- **CPU Core Utilization:**
  - Measures user and system CPU time via `getrusage(RUSAGE_SELF, ...)` over the elapsed wall-clock interval:
    $$\text{CPU Utilization \%} = \frac{\Delta\text{user\_sec} + \Delta\text{sys\_sec}}{\Delta\text{wall\_sec}} \times 100\%$$
    *(On a 4-thread execution saturating 4 CPU cores, this reports $\approx 360 \dots 390\%$).*

---

# Chapter 9: Performance and Benchmarking

### 9.1 Metric Definitions
- **Time To First Token (TTFT):** Wall-clock milliseconds from prompt receipt until the first generated token is sampled. Includes tokenization and prompt prefill of positions $0 \dots N-1$.
- **Prefill Throughput:** Prompt tokens processed divided by prefill time:
  $$\text{Prefill TPS} = \frac{N_{\text{prompt}}}{\text{TTFT (seconds)}}$$
- **Decode Throughput:** Tokens generated divided by decode generation time:
  $$\text{Decode TPS} = \frac{N_{\text{generated}} - 1}{\text{Decode Wall Time (seconds)}}$$
- **Peak RSS (Resident Set Size):** Maximum physical RAM pages committed by the OS kernel to the process.
- **CPU Utilization %:** Total active core utilization across all worker threads.
- **Model Loading Time:** Time required to `mmap` the file, parse the 201-tensor directory, and validate alignment.

### 9.2 The Reported M8 Benchmark Sweep
The following empirical measurements were recorded during our M8 benchmark sweep running real `TinyLlama-1.1B-Chat-v1.0` weights on commodity Apple Silicon hardware:

| Benchmark Metric | Measured Result | Significance |
| :--- | :--- | :--- |
| **Mean Decode Throughput** | **35.503 tokens/sec** | Faster than human reading speed (~5–8 tok/s); smooth streaming UX |
| **Mean Time to First Token (TTFT)** | **600.335 ms** | Sub-second response initiation on a 1.1B model running on a laptop CPU |
| **Peak Resident Set Size (RSS)** | **658.31 MiB** | Less than 1/12th the RAM of an 8 GB laptop; zero disk swapping |
| **Model Load Time** | **0.342 ms** | Zero-copy `mmap` instantaneous startup |
| **CPU Utilization (4 Threads)** | **~355–368%** | Excellent multi-core scaling efficiency across 4 CPU cores |

### 9.3 Statistical Aggregation Methodology
A single benchmark run can be skewed by background OS tasks, thermal throttling, or scheduler jitter. The engine includes `--bench-runs <N>` and Benchmark Lab which execute $N$ iterations (typically $N = 3$ or $N = 5$) and report:
- **Mean:** $\mu = \frac{1}{N} \sum x_i$
- **Median:** Middle value of sorted runs (robust against single-run outlier spikes)
- **Min / Max:** Extreme bounds capturing best-case and worst-case execution
- **Standard Deviation:** $\sigma = \sqrt{\frac{1}{N-1} \sum (x_i - \mu)^2}$ (demonstrates execution stability, typically $\sigma < 0.8\text{ tok/s}$)

### 9.4 Historical Baselines Comparison
| Engine Milestone | Model Format | Backend | Threads | Mean TTFT | Mean Decode TPS | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Milestone M5 (FP32)** | `FP32` | NEON | 4 | 6096.8 ms | 16.10 tok/s | 3956.6 MiB |
| **Milestone M6 (Q4_0)** | `Q4_0` | NEON | 4 | 561.26 ms | 35.25 tok/s | 658.20 MiB |
| **Milestone M7 (Clean)**| `Q4_0` | NEON | 4 | 657.79 ms | 34.03 tok/s | 658.16 MiB |
| **Milestone M8 (Studio)**| `Q4_0` | NEON | 4 | **600.34 ms** | **35.50 tok/s** | **658.31 MiB** |

### 9.5 Why Q4_0 Delivers Speedup on Hardware
Judges often ask: *"Quantized 4-bit weights require instructions to unpack nibbles and convert to float. Why is it faster than FP32 which requires no unpacking?"*
**The Memory-Bandwidth Wall:**
In single-token decoding, every token requires reading all 1.1 billion weights from memory.
- `FP32`: Must transfer **4.40 GB** across the memory bus per token. At 40 GB/s bandwidth, physical memory transfer takes $4.40 / 40 = \mathbf{110\text{ ms}}$ per token ($\le 9\text{ tok/s}$). The CPU ALUs sit stalled waiting for cache lines to arrive from RAM.
- `Q4_0`: Transfers only **0.62 GB** across the memory bus per token. Physical memory transfer takes $0.62 / 40 = \mathbf{15.5\text{ ms}}$ per token. The CPU has ample spare ALU cycles to unpack nibbles in vector registers while waiting for memory, resulting in a **$2.2\times$ decode throughput increase** (from 16.1 to 35.5 tok/s) and a **$10\times$ prefill speedup**.

---

# Chapter 10: Engineering Decisions and Trade-Offs (ADR Consolidated Defense)

Every major architectural decision in `inference-engine` was deliberate. Below is the technical defense and trade-off ledger:

### 1. C++17 Standard (`ADR-001`)
- **Alternative:** C++20 or C++23.
- **Why Chosen:** Universal compiler support across AppleClang 15+, GCC 9+, and MSVC 2019+. Avoids experimental module ecosystems and compiler-specific standard library inconsistencies.
- **Gains:** Rock-solid portability across macOS, Linux, and Windows; zero warning builds under `-Wall -Wextra -Wpedantic`.
- **Sacrifice:** Inability to use C++20 concepts or ranges; implemented lightweight template SFINAE and explicit validators instead.

### 2. Custom `.ie` v1 Format (`ADR-003`)
- **Alternative:** Loading `.safetensors` directly in C++, or using GGUF.
- **Why Chosen:** `.safetensors` contains a variable-length JSON string header without 64-byte payload alignment, forcing heap copies to align Q4_0 blocks. GGUF is tightly coupled to the GGML runtime codebase.
- **Gains:** Exact 64-byte payload alignment, checked 64-bit integer arithmetic, strict non-overlap validation, zero external JSON/binary parsing libraries at runtime.
- **Sacrifice:** Requires an offline one-time conversion step via Python (`convert_to_inference_engine.py`).

### 3. Symmetric Block `Q4_0` Quantization (`ADR-004`)
- **Alternative:** Asymmetric `Q4_1` (adds per-block minimum value $m$) or 8-bit `Q8_0`.
- **Why Chosen:** `Q4_0` achieves maximum compression (7.1x smaller) with symmetric scale $d$, requiring only 18 bytes per 32 weights. Split-half nibble layout maps cleanly to 128-bit SIMD vector instructions.
- **Gains:** 619 MB model file, $16.5+\text{ dB}$ SQNR, upstream compatibility with standard 4-bit quantization benchmarks.
- **Sacrifice:** Slight perplexity loss compared to `Q8_0` or `FP16` (compensated by high cosine similarity $>0.996$).

### 4. Zero-Copy `mmap` Model Loading (`ADR-009`)
- **Alternative:** `fread()` / `std::ifstream::read()` into heap buffers (`malloc` / `std::vector`).
- **Why Chosen:** Heap loading doubles physical memory requirements during startup and requires several seconds to allocate and copy 4.4 GB.
- **Gains:** Model loading in $<4\text{ ms}$, shared OS page cache, zero dynamic memory duplication, instant process startup.
- **Sacrifice:** Pointers returned by `TensorView` are strictly non-owning and tied to the `ModelLoader` virtual memory lifetime.

### 5. Persistent `ThreadPool` with Dynamic Chunking (`ADR-010`)
- **Alternative:** OpenMP (`#pragma omp parallel for`) or creating new `std::thread`s per matrix multiplication.
- **Why Chosen:** OpenMP introduces compiler runtime dependencies (`libgomp`, `libomp`). Creating threads per call causes catastrophic thread spawn latency.
- **Gains:** Zero runtime dependencies; worker threads spawned once. Dynamic chunking ($\text{num\_tasks} / (\text{threads} \times 4)$) eliminates lock contention across 32,000 matrix rows.
- **Sacrifice:** Requires custom synchronization primitives (`std::condition_variable`, generation counters, and atomic task indices).

### 6. 2-Row Register Tiling & Vectorized Scale Accumulation (`ADR-012`)
- **Alternative:** Naive 1-row matrix-vector multiplication with per-block horizontal sum reductions.
- **Why Chosen:** Calling horizontal reductions inside every block stalled CPU vector pipelines. Streaming activation vectors repeatedly saturated cache read ports.
- **Gains:** Activation vector loads cut by 50%; horizontal reduction deferred to a single end-of-row instruction. Delivers **+30.1% latency reduction** on AVX2 (Ryzen 7 7745HX) and up to **114.2 GFLOPS** on NEON (Apple Silicon).
- **Sacrifice:** Added code complexity with dedicated odd-row and tail fallbacks.

### 7. KV Cache Prefix Matching (`ADR-013`)
- **Alternative:** Stateless generation re-evaluating prompt history from scratch on each turn.
- **Why Chosen:** In multi-turn chat, re-evaluating prompts leads to increasing latency on every turn.
- **Gains:** Sub-30 ms prompt prefill on follow-up turns by reusing committed KV cache vectors.
- **Sacrifice:** Requires strict cache rewind truncation (`KvCache::truncate`) when history branches or is cleared.

### 8. Persistent Worker Architecture for Inference Studio (`ADR-014`)
- **Alternative:** Embedding a C++ HTTP server library (`cpp-httplib`, `boost::beast`) directly into `inference_engine_core`.
- **Why Chosen:** Embedding networking into the core tensor library contaminates a pure mathematical engine with socket handling, HTTP parsing, and SSL dependencies.
- **Gains:** Clear architectural separation: C++ engine focuses purely on high-performance tensor math; Node.js handles HTTP routing, Server-Sent Events, and client management.
- **Sacrifice:** Inter-process communication across stdio pipes introduces ~0.1 ms serialization overhead per event.

---

# Chapter 11: Limitations and Honest Engineering Claims

A core tenet of our technical defense is **absolute honesty with the judges**. We never fabricate numbers, claim unsupported features, or disguise stubs.

### 11.1 What is Fully Implemented and Verified
- **100% Implemented & Tested:**
  - Complete `.ie` binary loader and validator (`ModelLoader`, `TensorView`).
  - Python conversion tooling for `FP32`, pure `Q4_0`, and mixed `Q4_0` (`--keep-embeddings-fp32`).
  - Bitwise IEEE 754 half-precision conversion (`fp16_to_fp32`, `fp32_to_fp16`).
  - Complete scalar math reference library (`rmsnorm`, `rope`, `attention_gqa`, `swiglu`, `softmax`, `dot`, `matmul`).
  - ARM64 NEON SIMD kernels with 2-row register tiling and vectorized scale accumulation.
  - x86-64 AVX2 + FMA SIMD kernels with 2-row register tiling and vectorized scale accumulation.
  - Runtime CPU feature detection via `__builtin_cpu_supports` and `cpuid` + `_xgetbv`.
  - Contiguous float `KvCache` with commit contract and `truncate()`.
  - Full 22-layer `TransformerEngine` forward pass for prompt prefill and token decode.
  - SentencePiece C++ integration with dual-processor dummy-prefix handling and Zephyr chat template.
  - Sampler with greedy, temperature, top-p, top-k, and deterministic `xorshift*` PRNG.
  - Interactive chat REPL with KV-cache prefix matching and streaming detokenization.
  - Multi-run benchmark harness with statistical aggregation (mean, median, min, max, stddev).
  - Persistent worker mode (`--studio-worker`) with real OS RSS and CPU telemetry.
  - Node.js API Gateway and React 19 Observability Studio workstation.
  - **All 9/9 CTest test suites pass 100% with zero warnings.**

### 11.2 What Has Been Verified on Hardware vs. CI
- **Apple Silicon ARM64 (MacBook Air M4, 24 GB):** Fully verified on native hardware across all 9 CTest suites, Python tools, live TinyLlama streaming, and Inference Studio workstation.
- **AMD Ryzen 7 7745HX (16 GB, AVX2 + FMA):** Kernel microbenchmarks (`benchmarks/bench_kernels.cpp`) verified on native hardware across AVX2 and scalar paths (`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`). Full end-to-end TinyLlama inference is verified via GitHub Actions CI (`ubuntu-22.04` x86_64).
- **Intel Core Ultra 7 155H & Core i5-10210U:** Covered by portable standard C++17 build and AVX2 kernel paths; live end-to-end full-model hardware runs are scheduled as post-M8 priority items.

### 11.3 What is NOT Supported (Explicit Boundaries)
- **No GPU or NPU Acceleration:** We do not support CUDA, Metal, Vulkan, OpenCL, or Apple Neural Engine. All computation is strictly CPU-bound.
- **No Multi-Batch Inference:** The engine is optimized strictly for single-batch ($B = 1$) interactive chat and local evaluation. It does not implement continuous batching or PagedAttention.
- **No Model Architectures other than Llama-class Decoders:** The engine is built for TinyLlama / Llama-2 / Llama-3 style architectures (RMSNorm, RoPE, GQA, SwiGLU). It does not load GPT-2, Falcon, or MoE (Mixtral) architectures.
- **No In-Flight Context Paging:** Once context length reaches `max_seq_len` (2048), generation stops. We do not implement sliding-window attention or context compression.

### 11.4 What We Must NEVER Claim to Judges
1. **Never claim we support GPUs or Metal:** Our engine is 100% CPU-only.
2. **Never claim end-to-end AVX2 TinyLlama benchmarks match NEON numbers until physically executed on hardware:** AVX2 kernel microbenchmarks are verified on Ryzen 7, but full-model E2E numbers reported in the pitch deck are from our native ARM64 NEON runs.
3. **Never claim we wrote SentencePiece from scratch:** We integrated Google's upstream `sentencepiece-static` C++ library for tokenization and wrote custom wrapper logic for chat templates and prefix spaces.
4. **Never claim Q4_0 is completely lossless:** Quantization reduces file size by 7.1x with an SQNR of ~16.6 dB and cosine similarity of 0.996, but represents a minor, measurable approximation of full FP32 weights.
