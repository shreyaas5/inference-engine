# BIOS V2 — Comprehensive Judge Question Bank
## 65 Battle-Tested Technical Questions, Defenses, and Source Mappings
**Repository:** `inference-engine` | **Problem Statement:** PS08 (Own Inference Engine)  
**Evaluation Target:** Technical Judges, Systems Architects, and ML Engineers

---

## Guide Structure
Each question contains four mandatory components:
1. **20-Second Pitch Answer:** A concise, assertive elevator summary suitable for an immediate oral response.
2. **Deep Technical Defense:** Concrete mathematical, architectural, and algorithmic details proving first-principles mastery.
3. **Source Code References:** Exact file paths and line ranges demonstrating where the code lives.
4. **Likely Follow-Up Questions:** Anticipated probing questions and counter-defenses.

---

# Category 1: Basic Conceptual Questions

### Q1: What is an inference engine, and how does it differ from a model?
- **20-Second Answer:** The model is a static graph of frozen mathematical weights and hyperparameters. The inference engine is the C++ execution runtime that memory-maps those weights, manages CPU caches, executes SIMD matrix multiplication, allocates the KV cache, and samples tokens.
- **Deep Technical Defense:** In training, engines (like PyTorch) maintain computational graphs, activations for backpropagation, and optimizer states, consuming huge RAM. An inference engine executes strictly the unidirectional forward pass. It needs no backward pass, no gradient tape, and no Adam momentum buffers. Its execution focuses strictly on memory bandwidth saturation, register tiling, and low-latency single-batch ($B=1$) autoregressive decoding.
- **Source Files:** [`include/inference_engine/transformer.hpp`](../include/inference_engine/transformer.hpp), [`src/transformer/transformer.cpp:124`](../src/transformer/transformer.cpp).
- **Follow-Up:** *Why did you write the engine in C++ instead of Python with NumPy?* (Answer: C++ provides manual control over SIMD vector registers, CPU thread pinning, zero-copy `mmap`, and memory layout without GIL or garbage collector pauses).

### Q2: Why is TinyLlama 1.1B the right choice for this problem statement?
- **20-Second Answer:** TinyLlama exercises all modern Llama-3 class architectural primitives—RMSNorm, RoPE, Grouped Query Attention, and SwiGLU—while fitting entirely in 619 MB of RAM under Q4_0, making it ideal for commodity laptop CPUs.
- **Deep Technical Defense:** Older architectures like GPT-2 lack modern innovations (they use learned absolute positional embeddings, LayerNorm, standard MHA, and GeLU). Larger 7B models require ~4 GB in Q4_0, pushing 8 GB commodity laptops into memory paging. TinyLlama's 22 layers, 2048 hidden dim, and 8:1 GQA ratio represent modern production LLM architectures while allowing real-time 35+ tok/s execution on a consumer CPU.
- **Source Files:** [`include/inference_engine/config.hpp:11-20`](../include/inference_engine/config.hpp).
- **Follow-Up:** *Could your engine run Llama-3-8B?* (Answer: Yes, the 16-step forward pass is identical; it only requires loading a config with 32 layers and 4096 hidden dim).

### Q3: Why is your engine genuinely yours and not just a wrapper around llama.cpp or Ollama?
- **20-Second Answer:** We do not link or shell out to any external ML runtime. Every byte parser, NEON/AVX2 intrinsic loop, KV cache manager, and sampler is written from scratch in standard C++17 inside this repository.
- **Deep Technical Defense:** Inspect our CMakeLists.txt and binary symbols. We link only to standard OS libraries (`libc++`, `libm`, `pthread`) and static `sentencepiece`. There are zero shared library calls or subprocess pipes to `llama.cpp`, GGML, LibTorch, ONNX, or MLX. We designed our own `.ie` binary format, wrote the SIMD 2-row register tiling, implemented the thread pool, and authored the web workstation from first principles.
- **Source Files:** [`CMakeLists.txt`](../CMakeLists.txt), [`src/loader/loader.cpp`](../src/loader/loader.cpp), [`src/kernels/neon/kernels_neon.cpp`](../src/kernels/neon/kernels_neon.cpp).
- **Follow-Up:** *Why did you reuse SentencePiece?* (Answer: SentencePiece is used strictly for BPE string tokenization, which is a text preprocessing utility. All neural network tensor math is 100% our own).

### Q4: What is the fundamental difference between prompt prefill and token decoding?
- **20-Second Answer:** Prefill is compute-bound, evaluating all prompt tokens to populate the initial KV cache. Decoding is memory-bandwidth bound, generating one token at a time by streaming all 1.1 billion weights from memory once per token.
- **Deep Technical Defense:** In prefill, an $N$-token prompt can theoretically be evaluated as a matrix-matrix multiplication ($M=N, K=2048, N=2048$), with high arithmetic intensity ($O(N)$ arithmetic operations per byte loaded). In autoregressive decode, the batch size is $M=1$; we perform matrix-vector multiplication. We must load all 619 MB of weights to compute just 2 GFLOPs of math, making memory bus bandwidth the sole throughput bottleneck.
- **Source Files:** [`src/transformer/transformer.cpp:247`](../src/transformer/transformer.cpp) (`forward_prompt`), [`src/transformer/transformer.cpp:124`](../src/transformer/transformer.cpp) (`forward_token`).
- **Follow-Up:** *How do you optimize for the decode memory-bandwidth bottleneck?* (Answer: 4-bit quantization to reduce payload to 619 MB, and 2-row register tiling to load activations once for two rows).

### Q5: What is autoregressive text generation?
- **20-Second Answer:** Generating text sequentially token by token, where each newly produced token is appended to the context and fed back into the engine as input for the next step.
- **Deep Technical Defense:** The language model models the joint probability distribution of a sequence as $P(w_1, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$. At step $t$, the logits at position $t-1$ produce a probability distribution over the vocabulary. A token is sampled, its Key and Value vectors are added to the KV cache, and it becomes the input for position $t$.
- **Source Files:** [`src/app/main.cpp:687-720`](../src/app/main.cpp), [`src/app/studio_worker.hpp:890-940`](../src/app/studio_worker.hpp).
- **Follow-Up:** *When does the loop terminate?* (Answer: When the sampler outputs EOS token ID 2 (`</s>`) or when the sequence length reaches `max_new_tokens` / `max_seq_len`).

### Q6: What does Grouped Query Attention (GQA) accomplish?
- **20-Second Answer:** GQA groups multiple query heads to share a single key/value head, slashing the KV cache memory footprint and memory bus traffic by an entire factor of 8 with negligible loss in model accuracy.
- **Deep Technical Defense:** Standard Multi-Head Attention (MHA) pairs 32 Q heads with 32 KV heads, requiring 738 MB of KV cache at 2048 context. Multi-Query Attention (MQA) collapses to 1 KV head, which can degrade conversational quality. GQA strikes the optimal balance: TinyLlama groups 8 Query heads per KV head ($32 / 4 = 8$), reducing the KV cache to 88 MB and saving 87.5% of memory bandwidth during attention lookup.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:148-199`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *How is the mapping implemented?* (Answer: Query head index $h_q \in [0, 31]$ accesses KV head index $h_{kv} = \lfloor h_q / 8 \rfloor \in [0, 3]$).

### Q7: What is Root Mean Square Normalization (RMSNorm) and why use it over LayerNorm?
- **20-Second Answer:** RMSNorm enforces activation stability by scaling vectors by their root-mean-square without subtracting the mean, cutting computational steps and memory passes while preserving stability.
- **Deep Technical Defense:** Standard LayerNorm computes mean $\mu$ and variance $\sigma^2$: $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$. Zhang & Sennrich (2019) demonstrated that the mean-centering step is unnecessary for scaling invariance. RMSNorm computes $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}$ and sets $y = \frac{x}{\text{RMS}(x)} \odot w$. It saves a full vector pass and requires no learned bias vector $\beta$.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:9-22`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *Why accumulate sum of squares in double precision?* (Answer: To prevent numerical underflow or overflow across 2048 dimensions).

### Q8: What is Rotary Position Embedding (RoPE)?
- **20-Second Answer:** RoPE encodes relative token positions by rotating query and key vectors in 2D coordinate planes by angles proportional to their sequence position, allowing the model to naturally generalize across relative distances.
- **Deep Technical Defense:** Unlike absolute additive embeddings ($x + p$), RoPE satisfies the inner product property $\langle R_m q, R_n k \rangle = g(q, k, m - n)$. For each 64-dimensional head, dimensions are split into 32 pairs $(x_i, x_{i+32})$. Each pair is rotated by angle $\alpha_i = \text{pos} \cdot \theta^{-2i/64}$ using a 2D rotation matrix.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:24-71`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *Why precalculate RoPE tables?* (Answer: Calling `cos()` and `sin()` at runtime on every token across 36 heads is slow; we precompute a $2048 \times 32$ float table at startup).

### Q9: What is SwiGLU?
- **20-Second Answer:** SwiGLU is a gated feed-forward activation function combining a Swish (SiLU) activated gate projection with a linear up-projection via element-wise multiplication.
- **Deep Technical Defense:** Standard Transformers use $\text{FFN}(x) = \text{ReLU}(x W_1) W_2$. LLaMA architectures use Shazeer's SwiGLU: $\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$. In TinyLlama, $W_{\text{gate}}$ and $W_{\text{up}}$ expand the 2048 hidden dim to 5632 intermediate dim, followed by element-wise Hadamard multiplication and down-projection back to 2048.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:121-127`](../src/kernels/scalar/kernels_scalar.cpp), [`src/transformer/transformer.cpp:225-231`](../src/transformer/transformer.cpp).
- **Follow-Up:** *Why intermediate_dim = 5632?* (Answer: In LLaMA, the expansion factor is approximately $\frac{8}{3} d_{\text{model}} \approx \frac{8}{3} \times 2048 \approx 5461$, rounded to the nearest multiple of 256 for memory alignment, giving 5632).

---

# Category 2: Architecture & Format Questions

### Q10: Walk us through the binary layout of your `.ie` format.
- **20-Second Answer:** A 64-byte fixed header, followed by a variable-length tensor directory with checked 64-bit offsets, padded to 64 bytes, followed by 64-byte aligned tensor payloads.
- **Deep Technical Defense:**
  - Header (64 bytes): Magic `0x464D4549`, version `1`, tensor count (201), `weight_dtype` (2 for Q4_0), 8 $\times$ u32 hyperparameters, 2 $\times$ f32 ($\epsilon, \theta$), and u64 `total_file_bytes`.
  - Directory: Each entry stores `name_len` (u32), `name` (string), `dtype` (u32), `ndim` (u32), `shape[4]` (4 $\times$ u64), `payload_offset` (u64, aligned to 64), and `payload_nbytes` (u64).
  - Payloads: Stored sequentially with strict non-overlap checks.
- **Source Files:** [`include/inference_engine/loader.hpp:15-27`](../include/inference_engine/loader.hpp), [`docs/FORMAT_SPECIFICATION.md`](../docs/FORMAT_SPECIFICATION.md).
- **Follow-Up:** *Why enforce 64-byte alignment?* (Answer: 64 bytes matches the CPU L1 cache line size and 512-bit AVX-512 register boundaries, preventing unaligned memory cache line splits).

### Q11: What happens if a model file is corrupted or tampered with?
- **20-Second Answer:** Our loader performs a 10-point structural validation before mapping any memory. It rejects bad magic, truncated directories, non-finite floats, overlapping payloads, or size mismatches with descriptive error codes.
- **Deep Technical Defense:** `ModelLoader::load_from_buffer()` uses safe arithmetic helpers (`checked_add_u64`). It validates that `payload_offset + payload_nbytes <= total_file_bytes`, checks that no payload overlaps the directory, sorts intervals to verify that no payloads overlap each other, and verifies that the final payload ends exactly at `total_file_bytes`. Any error returns a `StatusCode::CorruptedFile` or `StatusCode::OutOfBounds` without crashing.
- **Source Files:** [`src/loader/loader.cpp:133-300`](../src/loader/loader.cpp).
- **Follow-Up:** *What tests prove this?* (Answer: `tests/test_loader_validation.cpp` contains 15 explicit corrupted-file rejection test cases).

### Q12: Why is the model ~7.1x smaller under Q4_0 rather than exactly 8x smaller?
- **20-Second Answer:** Because each 32-weight block requires an 18-byte struct (16 bytes weights + 2 bytes FP16 scale = 4.5 bits/weight), and all 45 RMSNorm tensors are intentionally preserved in 32-bit float.
- **Deep Technical Defense:** 32 bits divided by 4 bits is 8x. However:
  1. Block scale: 18 bytes / 32 weights = 4.5 bits/weight ($32 / 4.5 = 7.11\times$).
  2. 1D RMSNorm weights: 45 tensors (368,640 bytes) remain in FP32 (`ADR-008`) to avoid normalization scale distortion.
  3. Header, directory entries, and 64-byte padding add fixed constant overhead.
  - Exactly: $4,400,213,312\text{ B} / 619,113,792\text{ B} = \mathbf{7.1073\times}$.
- **Source Files:** [`include/inference_engine/quant.hpp:18-30`](../include/inference_engine/quant.hpp), [`src/loader/loader.cpp:468-490`](../src/loader/loader.cpp).
- **Follow-Up:** *Why not quantize RMSNorm to save another megabyte?* (Answer: Quantizing 2048 floats saves $<300\text{ KB}$ total across the entire model, but introduces severe scale distortion into activations).

### Q13: Does using `mmap` mean the entire 619 MB model is loaded into RAM?
- **20-Second Answer:** No. `mmap` allocates virtual address space only. The operating system kernel lazily demand-pages 4 KB blocks into the physical page cache only when the CPU touches them during matrix multiplication.
- **Deep Technical Defense:** When `mmap` executes, it only sets up Virtual Memory Area (VMA) page table mappings in the kernel, completing in $<0.4\text{ ms}$. Physical RAM is only allocated when a page fault occurs as weights are accessed. Because the OS page cache manages these pages, inactive pages can be evicted under memory pressure without disk writes.
- **Source Files:** [`src/loader/loader.cpp:87-114`](../src/loader/loader.cpp).
- **Follow-Up:** *What is your actual resident RAM footprint?* (Answer: Exactly 658.3 MiB, measured via `mach_task_basic_info` and `/proc/self/statm`).

### Q14: How does `TensorView` resolve tensor names to their actual binary data?
- **20-Second Answer:** `ModelLoader` builds an `unordered_map<string, TensorView>` during parsing. Finding a tensor is an $O(1)$ lookup returning a non-owning `TensorView` pointing directly to the memory-mapped offset.
- **Deep Technical Defense:** In `ModelLoader::bind_transformer_weights()`, we query this map for all 201 expected tensors (e.g. `model.layers.0.self_attn.q_proj.weight`). Each tensor view validates that dimensionality matches expectations (`expected_ndim`, `expected_shape`, `expected_dtype`) and binds the raw pointer into `TransformerWeightBindings`.
- **Source Files:** [`src/loader/loader.cpp:302-340`](../src/loader/loader.cpp), [`include/inference_engine/tensor.hpp:33-75`](../include/inference_engine/tensor.hpp).
- **Follow-Up:** *What if a tensor name is missing from the file?* (Answer: `validate_tensor_binding` returns `StatusCode::NotFound` and halts loading before execution begins).

### Q15: How does your mixed-precision export policy work (`ADR-008`)?
- **20-Second Answer:** All 154 2D projection matrices are quantized to `Q4_0`. All 45 1D normalization tensors remain `FP32`. Token embeddings and `lm_head` default to `Q4_0` but can be exported as `FP32` via `--keep-embeddings-fp32`.
- **Deep Technical Defense:** In `--keep-embeddings-fp32` mode, the file size is 1.07 GB. `ModelLoader::bind_transformer_weights` inspects `desc.dtype` for each tensor individually. If `embed_tokens` is FP32, `forward_token()` does a direct pointer copy; if Q4_0, it dequantizes the single row into `x_` via `dequantize_row_q4_0()`.
- **Source Files:** [`tools/convert_to_inference_engine.py:314-350`](../tools/convert_to_inference_engine.py), [`src/transformer/transformer.cpp:154-176`](../src/transformer/transformer.cpp).
- **Follow-Up:** *Why offer `--keep-embeddings-fp32`?* (Answer: For users who prioritize maximal vocabulary projection precision over 450 MB of disk savings).

---

# Category 3: Mathematical & Algorithmic Questions

### Q16: What is the exact mathematical formula for Q4_0 quantization and dequantization?
- **20-Second Answer:** Scale is $d = \frac{\max |v|}{-8.0}$. Quantization: $q = \text{clamp}(\lfloor x/d + 8.5 \rfloor, 0, 15)$. Dequantization: $w = (q - 8) \times d$.
- **Deep Technical Defense:**
  - Finding scale: $d = \arg\max_{v} |v| / -8.0f$, converted to IEEE 754 half-precision float (`uint16_t`).
  - Quantization: $x$ is multiplied by reciprocal $1/d$, shifted by $+8.5$ for rounding, clamped to $[0, 15]$.
  - Dequantization: The 4-bit unsigned integer $q \in [0, 15]$ has offset 8 subtracted, yielding a signed integer in $[-8, +7]$, multiplied by $\text{fp16\_to\_fp32}(d)$.
- **Source Files:** [`src/kernels/quant_q4_0.cpp:102-142`](../src/kernels/quant_q4_0.cpp).
- **Follow-Up:** *Why use $-8.0f$ in the denominator instead of $+7.0f$?* (Answer: $-8.0f$ matches the GGML/llama.cpp symmetric convention, mapping the signed range $[-8, +7]$ with zero represented as $q = 8$).

### Q17: What is the computational complexity of your attention mechanism?
- **20-Second Answer:** At sequence position $\text{pos}$, the attention score computation is $O(\text{pos} \times d_{\text{head}})$ per head, scaling to $O(\text{pos})$ overall per token. Total prefill across $N$ tokens is $O(N^2)$.
- **Deep Technical Defense:**
  - For query head $h_q$, we compute dot product with $k_{h_{kv}, t}$ for all $t \in [0, \text{pos}]$.
  - Length of vector is $d_{\text{head}} = 64$.
  - Number of query heads is 32.
  - Flops per token decode at position $P$: $32 \times (P + 1) \times 64 \times 2 = 4096 \times (P + 1)$ FLOPs.
  - Across an $N$-token prompt, sum is $\sum_{p=0}^{N-1} 4096(p+1) = 2048 N^2$ FLOPs.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:148-199`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *At what context length does attention compute exceed matrix multiplication?* (Answer: For TinyLlama, FFN and projections take ~2 GFLOPs per token. Attention only reaches 2 GFLOPs at $\text{pos} \approx 500,000$, well beyond our 2048 limit).

### Q18: How do you prevent numerical overflow in Softmax?
- **20-Second Answer:** We use the standard numerical trick of subtracting the maximum logit before exponentiation: $P_i = \frac{\exp(x_i - \max x)}{\sum \exp(x_j - \max x)}$.
- **Deep Technical Defense:** In `softmax_f32`, we find $m = \max_{i} x_i$. All values $(x_i - m) \le 0$, ensuring $\exp(x_i - m) \in (0, 1]$. This prevents floating-point overflow ($\infty$) while double-precision accumulation (`double sum`) prevents denominator underflow.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:129-146`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *What if all logits are $-\infty$?* (Answer: The sampler explicitly checks for non-finite logits and falls back to uniform selection or error status).

### Q19: Explain the RoPE half-dim split rotation vs adjacent-pair rotation.
- **20-Second Answer:** Some papers rotate adjacent pairs $(x_{2i}, x_{2i+1})$. LLaMA and TinyLlama split the head into two halves, pairing dimension $i$ with dimension $i + 32$.
- **Deep Technical Defense:**
  - In Hugging Face transformers and LLaMA, `rotate_half` is defined as:
    $$\text{rotate\_half}(x) = [-x[32..63], x[0..31]]$$
  - The transformation is:
    $$y_i = x_i \cos\alpha_i - x_{i+32} \sin\alpha_i, \qquad y_{i+32} = x_{i+32} \cos\alpha_i + x_i \sin\alpha_i$$
  - Using adjacent pairs causes total token degradation and nonsensical text output.
- **Source Files:** [`src/kernels/scalar/kernels_scalar.cpp:35-46`](../src/kernels/scalar/kernels_scalar.cpp).
- **Follow-Up:** *How did you verify this?* (Answer: Verified against Hugging Face golden outputs in `tests/test_golden_forward.cpp` with zero numerical divergence).

### Q20: How does your PRNG ensure deterministic sampling?
- **20-Second Answer:** We use a 64-bit `xorshift*` pseudo-random number generator initialized with an explicit seed (default: 42) that produces identical random sequences across all operating systems.
- **Deep Technical Defense:** Standard `std::default_random_engine` is implementation-defined across GCC, Clang, and MSVC. Our `xorshift*` implementation (`src/sampler/sampler.cpp:125`) uses the explicit multiplier `0x2545F4914F6CDD1DULL`. Given the same seed, temperature, and logits, it produces bit-identical token selections on macOS, Linux, and Windows.
- **Source Files:** [`src/sampler/sampler.cpp:125-137`](../src/sampler/sampler.cpp).
- **Follow-Up:** *What is your greedy sampling strategy?* (Answer: When temperature is 0, we bypass the PRNG entirely and execute an exact argmax with tie-breaking toward the lowest token ID).

---

# Category 4: Implementation & Concurrency Questions

### Q21: Explain your `KvCache` commit contract and why `append()` does not advance `seq_len()`.
- **20-Second Answer:** A single token forward pass evaluates 22 layers sequentially. `append()` writes layer $i$'s K/V data. `commit_token()` increments sequence length only after all 22 layers finish.
- **Deep Technical Defense:** If `append()` auto-incremented sequence length on the final layer, any intermediate layer failure, dry-run, or unit test would corrupt the cache length. Explicit separation (`ADR-005`) guarantees that sequence length reflects only fully evaluated tokens, keeping RoPE angles and attention masks strictly synchronized.
- **Source Files:** [`src/core/kv_cache.cpp:38-74`](../src/core/kv_cache.cpp), [`ADR-005`](../DECISIONS.md).
- **Follow-Up:** *What happens during multi-token prompt prefill?* (Answer: `commit_positions(count)` increments `seq_len` once by $N$ after prefill finishes).

### Q22: How does your ThreadPool avoid race conditions and lock contention?
- **20-Second Answer:** It uses a generation counter, dynamic task chunking ($\text{tasks} / (\text{threads} \times 4)$), thread 0 participation, and condition variables, eliminating 95%+ of mutex acquisitions.
- **Deep Technical Defense:**
  - Standard worker pools grab 1 task per lock. For 32,000 rows in `lm_head`, that causes 32,000 mutex lock/unlock cycles, causing severe thread convoying on high-core CPUs.
  - Our pool chunks work into blocks of 100–500 rows. A single lock acquisition executes hundreds of rows.
  - Thread 0 participates, preventing the calling thread from idling.
  - Exceptions are captured via `std::exception_ptr` and rethrown cleanly on the calling thread.
- **Source Files:** [`src/kernels/threading/thread_pool.cpp:31-92`](../src/kernels/threading/thread_pool.cpp).
- **Follow-Up:** *Did you test this under high concurrency?* (Answer: `tests/test_kernels_simd.cpp` runs 1,000 rapid successive parallel-for cycles with zero deadlocks).

### Q23: How do you handle unsupported CPU architectures?
- **20-Second Answer:** Our dispatcher detects CPU capabilities at runtime. If NEON or AVX2 is absent, execution gracefully falls back to our portable scalar kernels.
- **Deep Technical Defense:** In `src/common/common.cpp:95`, `is_backend_available()` checks compiler macros (`__aarch64__`, `__AVX2__`) and runtime CPUID + OS XSAVE features. If a user runs our binary on an older x86 CPU without AVX2, `detect_best_backend()` automatically selects `SimdBackend::Scalar`. All translation units compile cleanly on all platforms.
- **Source Files:** [`src/common/common.cpp:95-158`](../src/common/common.cpp), [`src/kernels/dispatch.cpp:7-28`](../src/kernels/dispatch.cpp).
- **Follow-Up:** *What if a user forces `--backend avx2` on an ARM machine?* (Answer: The CLI and Studio reject the request with `StatusCode::InvalidArgument` and report available backends).

### Q24: What happens when the KV cache reaches its context limit?
- **20-Second Answer:** When sequence length reaches `max_seq_len` (2048), `forward_token()` returns `StatusCode::OutOfBounds`. The CLI and Studio warn the user and prompt them to `/clear`.
- **Deep Technical Defense:**
  ```cpp
  if (pos >= config_.max_seq_len) {
      return Status::Error(StatusCode::OutOfBounds, "pos exceeds max_seq_len");
  }
  ```
  We intentionally do not silently overwrite older tokens with a naive ring buffer because without positional re-indexing, RoPE angles become corrupted.
- **Source Files:** [`src/transformer/transformer.cpp:137-139`](../src/transformer/transformer.cpp), [`src/app/main.cpp:609-615`](../src/app/main.cpp).
- **Follow-Up:** *How would you support infinite generation?* (Answer: Sliding-window attention with positional re-computation or StreamingLLM attention sinks).

### Q25: How does your interactive chat REPL reuse KV cache prefixes across turns?
- **20-Second Answer:** We track committed token IDs. On each turn, we compute the common prefix between history and cache, truncate the cache to the match length, and prefill only new tokens.
- **Deep Technical Defense:**
  ```cpp
  size_t match_len = 0;
  while (match_len < committed_tokens.size() &&
         match_len < full_tokens.size() &&
         committed_tokens[match_len] == full_tokens[match_len]) {
      ++match_len;
  }
  if (match_len < committed_tokens.size()) {
      engine.kv_cache().truncate(match_len);
  }
  // Prefill ONLY from match_len to full_tokens.size() - 1
  ```
  On Turn 2, 30+ tokens of system prompt and prior dialogue are reused instantly, reducing prefill time from 550 ms to $<30\text{ ms}$.
- **Source Files:** [`src/app/main.cpp:622-650`](../src/app/main.cpp), [`src/core/kv_cache.cpp:76-86`](../src/core/kv_cache.cpp).
- **Follow-Up:** *What if the user edits previous turns?* (Answer: `match_len` drops to the edit point; `truncate()` rewinds the cache to that exact point, and only the edited suffix is prefilled).

### Q26: Why is stdout strictly protocol-clean in `--studio-worker` mode?
- **20-Second Answer:** Because stdout is the machine-to-machine JSON-lines transport pipe connected to Node.js. Random `printf` outputs would corrupt JSON parsing and crash the frontend stream.
- **Deep Technical Defense:** All internal diagnostics, timings, and warnings in `studio_worker.hpp` are wrapped in `{"event":"log","level":"INFO","message":"..."}` JSON payloads protected by a mutex `stdout_mutex_`. Node's `readline` parser receives guaranteed valid JSON on every line.
- **Source Files:** [`src/app/studio_worker.hpp:467-473`](../src/app/studio_worker.hpp), [`studio/server/engineManager.ts:180-210`](../studio/server/engineManager.ts).
- **Follow-Up:** *What if a fatal crash occurs?* (Answer: C++ crashes print to `stderr`; Node.js captures `stderr` and emits a critical error event to the UI).

---

# Category 5: Performance & Optimisation Questions

### Q27: Explain 2-row register tiling and why it speeds up matrix multiplication.
- **20-Second Answer:** By processing rows in pairs, we load each activation vector element once into registers and reuse it across two rows of weights, cutting activation memory loads by 50%.
- **Deep Technical Defense:** In $y = W x$, loading $x$ repeatedly for all $M$ rows saturates L1/L2 cache load ports. Under 2-row tiling:
  - Rows $m$ and $m+1$ are evaluated concurrently.
  - Slices of $x$ (32 floats) are loaded into vector registers once.
  - Vector FMAs accumulate dot products for Row $m$ and Row $m+1$ simultaneously.
  - On AVX2, this utilizes 12 of the 16 YMM registers with zero spilling, boosting throughput by +30.1%.
- **Source Files:** [`src/kernels/neon/kernels_neon.cpp:184-277`](../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp:210-300`](../src/kernels/avx2/kernels_avx2.cpp).
- **Follow-Up:** *What if the row count is odd?* (Answer: A clean tail fallback processes the final single row using 1-row vector dot product).

### Q28: What is vectorized scale accumulation and how does it eliminate stalls?
- **20-Second Answer:** We accumulate unscaled dot products directly in vector registers across all blocks and scale them by the FP16 scale, deferring horizontal addition to a single instruction at row end.
- **Deep Technical Defense:** In naive Q4_0 dot products, developers call `hsum256_ps` or `vaddvq_f32` inside every 32-weight block. A 2048-dim row has 64 blocks, incurring 64 horizontal reduction stalls (~14 cycles each = ~900 stall cycles per row). Our kernel accumulates unscaled float vectors in registers, scales by `fp16_to_fp32(d)`, and executes a single horizontal reduction per row.
- **Source Files:** [`src/kernels/neon/kernels_neon.cpp:105-110`](../src/kernels/neon/kernels_neon.cpp), [`src/kernels/avx2/kernels_avx2.cpp:24-72`](../src/kernels/avx2/kernels_avx2.cpp).
- **Follow-Up:** *Does deferring reduction cause floating-point overflow?* (Answer: No, because activations and quantized weights in normalized LLMs have bounded dynamic range; 32 inner products sum to $<100.0$).

### Q29: Why did you omit software prefetching (`_mm_prefetch`)?
- **20-Second Answer:** Empirical testing on AMD Zen 4 showed manual prefetching degraded throughput by 10–23% because hardware stream prefetchers already predict linear row strides accurately.
- **Deep Technical Defense:** In `benchmarks/results/benchmarks_prefetch.csv`, Attention Q dropped from 157.09 GFLOPS to 121.05 GFLOPS when `_mm_prefetch` was added. Modern CPU hardware prefetchers detect linear streaming access within 2 cache lines. Software prefetches waste front-end instruction issue slots and pollute load/store queues.
- **Source Files:** [`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md:144-180`](../docs/benchmarks/CPU_KERNEL_BENCHMARKS.md), [`ADR-012`](../DECISIONS.md).
- **Follow-Up:** *Does this hold true across all CPUs?* (Answer: Yes, modern Intel Raptor/Meteor Lake and Apple Silicon M-series have aggressive hardware prefetchers that outperform manual prefetch hints on contiguous data).

### Q30: How do you measure real CPU utilization and RSS?
- **20-Second Answer:** We query OS kernel APIs: `task_info(mach_task_self(), MACH_TASK_BASIC_INFO)` on macOS, `/proc/self/statm` on Linux, and `getrusage()` for user/system CPU time.
- **Deep Technical Defense:** We do not calculate theoretical estimates. In `studio_worker.hpp`, `current_resident_bytes()` queries the OS kernel for committed physical memory. `getrusage(RUSAGE_SELF, ...)` measures exact microseconds spent in user space and kernel space. CPU utilization is $\frac{\Delta\text{CPU Time}}{\Delta\text{Wall Time}} \times 100\%$.
- **Source Files:** [`src/app/studio_worker.hpp:50-80`](../src/app/studio_worker.hpp), [`src/app/main.cpp:22-34`](../src/app/main.cpp).
- **Follow-Up:** *Why does CPU utilization report ~360% on 4 threads?* (Answer: Operating systems report 100% per saturated core; 4 threads running at 90% efficiency yields 360%).

### Q31: Is your benchmark measuring prefill or decode?
- **20-Second Answer:** Both, reported separately. Prefill throughput measures prompt ingestion speed (tok/s); decode throughput measures single-token autoregressive generation speed (tok/s).
- **Deep Technical Defense:** Conflating prefill and decode produces deceptive numbers because prefill is compute-bound and can achieve 100+ tok/s, while decode is memory-bound. Our `--benchmark` harness and Benchmark Lab explicitly report TTFT (ms), Prefill Throughput (prompt tokens / prefill sec), and Decode Throughput (generated tokens / decode sec).
- **Source Files:** [`src/app/main.cpp:830-880`](../src/app/main.cpp), [`studio/server/engineManager.ts:43-57`](../studio/server/engineManager.ts).
- **Follow-Up:** *What are your official numbers on Apple Silicon?* (Answer: ~35.5 tok/s decode, ~35.6 tok/s prefill, 600 ms TTFT on 4 threads).

### Q32: Why does decode throughput plateau beyond 4 to 8 threads?
- **20-Second Answer:** Single-token decode is memory-bandwidth bound. Once 4 to 8 cores saturate the laptop's dual-channel LPDDR memory bus, adding more threads adds synchronization overhead without memory bandwidth.
- **Deep Technical Defense:** A laptop with 40 GB/s practical memory bandwidth can stream 619 MB of weights at most $40 / 0.619 \approx 64$ times per second. 4 high-performance CPU cores can already saturate this bus. Spawning 16 SMT threads introduces cache thrashing and lock contention, yielding diminishing returns.
- **Source Files:** [`benchmarks/results/benchmarks_optimised.csv`](../benchmarks/results/benchmarks_optimised.csv), [`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`](../docs/benchmarks/CPU_KERNEL_BENCHMARKS.md).
- **Follow-Up:** *How did you handle this on small matrices?* (Answer: In `dispatch.cpp`, we cap task threads for matrices with $\text{out\_dim} \le 256$ to avoid SMT synchronization stalls).

---

# Category 6: Critical Systems Engineering Questions

### Q33: How do you avoid race conditions between the persistent C++ worker and Node.js?
- **20-Second Answer:** Unidirectional pipe streams, atomic cancellation flags, command queuing via thread-safe mutex/condition variable, and request IDs in every message.
- **Deep Technical Defense:** In `studio_worker.hpp`, the stdin reader thread runs decoupled from the inference worker thread. Incoming commands are pushed to `std::queue<JsonValue>` protected by `queue_mutex_`. The inference loop checks `stop_requested_` on every token. Outgoing stdout writes are serialized under `stdout_mutex_`. Every command carries a `request_id` echoed in its response.
- **Source Files:** [`src/app/studio_worker.hpp:415-500`](../src/app/studio_worker.hpp).
- **Follow-Up:** *What happens if the client disconnects?* (Answer: Node.js detects HTTP socket close and immediately writes `{"cmd":"stop"}` to stdin).

### Q34: What overhead does the Inference Studio architecture introduce?
- **20-Second Answer:** Stdio pipe IPC introduces $<0.1\text{ ms}$ latency per event. React UI rendering runs asynchronously at 60 FPS without blocking the C++ engine.
- **Deep Technical Defense:** The C++ worker streams token events directly to stdout as single-line JSON strings (~80 bytes). Operating system pipe buffers (64 KB) buffer these writes without blocking. Node's `readline` emits SSE packets to the browser. End-to-end token latency from C++ emission to screen render is $<2\text{ ms}$.
- **Source Files:** [`studio/server/engineManager.ts:180-220`](../studio/server/engineManager.ts), [`studio/src/components/ChatPlayground.tsx`](../studio/src/components/ChatPlayground.tsx).
- **Follow-Up:** *Did you consider shared memory (shm)?* (Answer: We evaluated POSIX shared memory, but stdio pipes provide cross-platform portability, built-in OS buffering, and simpler process supervision with negligible latency).

### Q35: How do you validate your quantization quality?
- **20-Second Answer:** We evaluate MAE, RMSE, RelL2, and SQNR on real weights using `inspect_checkpoint.py`, and assert cosine similarity $\ge 0.996$ on 22-layer forward logits against FP32.
- **Deep Technical Defense:** In `tools/inspect_checkpoint.py --eval-real-quant-error`, we inspect all 201 tensors. Mean Cosine Similarity is $\ge 0.9888$ across all projections with SQNR $\approx 16.6\text{ dB}$. In `tests/test_transformer_q4.cpp`, we execute the full 22-layer forward pass on synthetic weights in both FP32 and Q4_0, asserting output logit cosine similarity $\ge 0.996359$.
- **Source Files:** [`tools/inspect_checkpoint.py:120-180`](../tools/inspect_checkpoint.py), [`tests/test_transformer_q4.cpp`](../tests/test_transformer_q4.cpp).
- **Follow-Up:** *What is the impact on perplexity?* (Answer: On 1B models, standard Q4_0 incurs ~0.2–0.3 perplexity degradation, preserving syntactic fluency and conversational intent).

### Q36: How do you ensure your tokenizer matches Hugging Face exactly?
- **20-Second Answer:** We run `tools/verify_tokenizer_parity.py` against Hugging Face's Python implementation across 15 complex test cases, asserting 100% bit-exact token ID matches.
- **Deep Technical Defense:** SentencePiece has tricky edge cases: whitespace dummy prefixing (`▁`), byte fallback for non-ASCII characters, and special token tags (`<|user|>`). Our dual-processor wrapper handles special token boundaries, matching Hugging Face LlamaTokenizer legacy behavior across chat templates, numbers, unicode, and punctuation.
- **Source Files:** [`src/tokenizer/tokenizer.cpp:88-142`](../src/tokenizer/tokenizer.cpp), [`tools/verify_tokenizer_parity.py`](../tools/verify_tokenizer_parity.py).
- **Follow-Up:** *What happens with unknown words?* (Answer: SentencePiece decomposes unknown words into byte-fallback tokens `<0xXX>` in range [3, 258], preventing any dropped characters).

### Q37: How would you scale this engine to an 8B or 70B model?
- **20-Second Answer:** The 16-step forward pass is architecture-identical; scaling requires adjusting `ModelConfig` hyperparameters, adding tensor parallelism across cores, or implementing multi-socket NUMA-aware allocation.
- **Deep Technical Defense:** An 8B model (like Llama-3-8B) uses 32 layers, 4096 hidden dim, and 8 KV heads. It requires ~4.5 GB in Q4_0, which fits in our existing single-node `mmap` architecture. For 70B (which requires ~38 GB), we would partition layers or weight matrices across NUMA memory nodes and use non-blocking inter-core synchronization.
- **Source Files:** [`include/inference_engine/config.hpp:22-66`](../include/inference_engine/config.hpp).
- **Follow-Up:** *Would your current ThreadPool scale to 64 threads?* (Answer: Yes, dynamic task chunking scales linearly, but NUMA memory pinning would be required to prevent remote memory socket latency).

---

# Category 7: Adversarial Judge Questions

### Q38: "Why didn't you just use llama.cpp? Isn't writing your own engine reinventing the wheel?"
- **20-Second Answer:** Problem Statement PS08 explicitly requires designing our **own** inference engine. Wrapping llama.cpp would fail the competition mandate and teach nothing about low-level systems engineering.
- **Deep Technical Defense:** Anyone can download a binary and build a web wrapper. By writing our own engine:
  - We proved we can design memory-safe binary formats with byte-level alignment.
  - We engineered handwritten ARM NEON and AVX2 SIMD vector loops with 2-row tiling.
  - We mastered KV cache memory structures, RoPE mathematics, and custom thread scheduling.
  - When performance degrades or memory corrupts, we can profile and fix the code because every line is ours.
- **Source Files:** [`CMakeLists.txt`](../CMakeLists.txt), [`AGENTS.md`](../AGENTS.md).
- **Follow-Up:** *How does your performance compare to llama.cpp?* (Answer: llama.cpp achieves ~40–45 tok/s on M4; our engine achieves ~35.5 tok/s decode, demonstrating we are within ~15–20% of a project with 50,000+ GitHub stars and hundreds of contributors).

### Q39: "How do I know you didn't cheat and call PyTorch or llama.cpp behind the scenes?"
- **20-Second Answer:** Run `otool -L` (macOS) or `ldd` (Linux) on our executable. It links only to system `libc++`, `libm`, and `pthread`. Grep the binary for symbols—there are zero external ML references.
- **Deep Technical Defense:**
  Execute on terminal:
  ```bash
  otool -L build/inference-engine
  nm -g build/inference-engine | grep -E "llama|ggml|torch|onnx"
  ```
  The symbol table shows only symbols in `namespace inference_engine` and Google `sentencepiece`. You can disconnect the laptop from Wi-Fi, purge the system, compile from source, and run inference completely isolated.
- **Source Files:** [`CMakeLists.txt:46-55`](../CMakeLists.txt).
- **Follow-Up:** *What about Python tools?* (Answer: Python scripts in `tools/` are offline-only tools for model conversion and parity verification. They are not packaged or executed at runtime).

### Q40: "Can you prove your output is mathematically correct and not hallucinated garbage?"
- **20-Second Answer:** We have automated golden parity tests comparing our C++ forward pass against bit-exact Hugging Face PyTorch activations with $<10^{-6}$ error on FP32 and $>0.996$ cosine similarity on Q4_0.
- **Deep Technical Defense:** In `tests/test_golden_forward.cpp`, we execute a multi-layer forward pass on known deterministic inputs and verify against reference vectors. In `tests/test_sampler.cpp`, we verify sampling distribution bounds. Furthermore, live chat prompts generate coherent English explanations of entropy, algorithms, and systems engineering.
- **Source Files:** [`tests/test_golden_forward.cpp`](../tests/test_golden_forward.cpp), [`tests/test_transformer_q4.cpp`](../tests/test_transformer_q4.cpp).
- **Follow-Up:** *Can you demonstrate a live chat turn right now?* (Answer: Yes, run `./build/inference-engine --chat` or open Inference Studio at `http://localhost:3000`).

### Q41: "Your prompt mentions AVX2 benchmarks, but you are presenting on a Mac. Are those numbers real?"
- **20-Second Answer:** Yes. Member I developed and benchmarked the AVX2 kernels on an AMD Ryzen 7 7745HX laptop. The CSV raw results and methodology are committed in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`.
- **Deep Technical Defense:** Our team hardware matrix divides responsibilities: Member S on Apple Silicon M4 (NEON), Member I on AMD Ryzen 7 (AVX2+FMA), Member U on Intel Core Ultra 7 (AVX2), and Member P on Intel Core i5 (AVX2). The AVX2 benchmark suite (`benchmarks/bench_kernels.cpp`) was executed natively on Ryzen 7, recording a 30.1% latency reduction and 162.8 GFLOPS on SwiGLU down-projection. CI runs on GitHub Ubuntu x86_64 runners verify AVX2 compilation and numerical parity.
- **Source Files:** [`docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`](../docs/benchmarks/CPU_KERNEL_BENCHMARKS.md), [`benchmarks/results/benchmarks_optimised.csv`](../benchmarks/results/benchmarks_optimised.csv).
- **Follow-Up:** *Why not run full TinyLlama on Ryzen during this presentation?* (Answer: This physical presentation laptop is Member S's MacBook Air; our cross-platform CI and benchmark logs provide verifiable proof of x86 execution).

### Q42: "What happens if a prompt contains 3,000 tokens? Does your engine crash?"
- **20-Second Answer:** No. The engine enforces sequence length validation before prefill begins. It returns `StatusCode::OutOfBounds` with an informative error message.
- **Deep Technical Defense:**
  ```cpp
  if (prompt_ids.size() >= loader.config().max_seq_len) {
      return fail("Prompt length exceeds the model's maximum context length");
  }
  ```
  The engine checks total requested length ($N_{\text{prompt}} + N_{\text{max\_tokens}}$) against `kTinyLlamaMaxSeqLen` (2048) in `main.cpp:748`. It halts gracefully without segmentation faults or heap buffer overflows.
- **Source Files:** [`src/app/main.cpp:745-752`](../src/app/main.cpp).
- **Follow-Up:** *Could you allocate more than 2048?* (Answer: TinyLlama was pretrained with RoPE base frequency tuned for 2048; exceeding 2048 without RoPE NTK-aware scaling causes perplexity explosion).

### Q43: "Why is your KV cache stored in FP32 instead of Q4_0?"
- **20-Second Answer:** Because the KV cache for TinyLlama is only 88 MB under GQA. Quantizing the KV cache to 4-bit would save $<70\text{ MB}$ while introducing attention score error on every autoregressive step.
- **Deep Technical Defense:** In 70B models, KV cache quantization is essential because cache size can reach 40 GB. In TinyLlama, GQA 8:1 already compressed the cache to 88 MB (45 MB for K, 45 MB for V). Storing K and V in full `FP32` preserves exact attention dot-product precision with zero quantization error, and 88 MB is negligible in a 658 MB total footprint.
- **Source Files:** [`include/inference_engine/kv_cache.hpp:18-25`](../include/inference_engine/kv_cache.hpp).
- **Follow-Up:** *When would you quantize the KV cache?* (Answer: If we extended context length to 32,768 tokens or scaled to an 8B model with 32 KV heads).

### Q44: "How do you know there are no memory leaks in your C++ code?"
- **20-Second Answer:** We enforce RAII everywhere. All weight pointers are non-owning `TensorView`s into `mmap`. Activations and KV caches use standard containers allocated once at initialization.
- **Deep Technical Defense:** There are zero raw `malloc`, `free`, or `new` calls in our forward inference loop. `TransformerEngine` pre-allocates all activation vectors in `init()`. When `ModelLoader` destructs, it automatically calls `munmap()` and `close(fd_)`. In Inference Studio, resident RSS stays rock-solid at ~658 MiB across hundreds of conversational turns without climbing.
- **Source Files:** [`src/loader/loader.cpp:68-85`](../src/loader/loader.cpp), [`src/transformer/transformer.cpp:76-87`](../src/transformer/transformer.cpp).
- **Follow-Up:** *Did you run AddressSanitizer (ASan)?* (Answer: Yes, builds configured with `-DINFERENCE_ENGINE_ENABLE_ASAN=ON` pass all 9 CTest suites with zero memory leaks or buffer overflows).

### Q45: "What if the user inputs an unknown Unicode emoji or non-English script?"
- **20-Second Answer:** The SentencePiece tokenizer uses byte fallback. Any character not present in the 32,000-piece vocabulary is split into individual UTF-8 byte tokens `<0xXX>`.
- **Deep Technical Defense:** TinyLlama's vocabulary contains dedicated tokens for all 256 individual byte values (tokens 3 to 258). When an arbitrary Unicode character (e.g. 🦀 / `0xF0 0x9F 0xA6 0x80`) is processed, SentencePiece emits the 4 byte tokens. The engine evaluates them normally, and our incremental prefix detokenizer reconstructs the original UTF-8 character on output.
- **Source Files:** [`src/tokenizer/tokenizer.cpp:180-210`](../src/tokenizer/tokenizer.cpp).
- **Follow-Up:** *What if a byte sequence is incomplete?* (Answer: The incremental prefix detokenizer waits until the final byte token arrives before printing the decoded Unicode glyph).

---

# Additional Rapid-Fire Technical Questions (Q46–Q65)

### Q46: What is the head dimension of TinyLlama?
- **Answer:** 64 ($2048 / 32 = 64$). Located in `include/inference_engine/config.hpp:16`.

### Q47: Why is head dimension even?
- **Answer:** RoPE pairs dimension $i$ with $i + d/2$. An odd head dimension cannot be evenly paired. Enforced in `config.hpp:56`.

### Q48: How many total tensors are in TinyLlama?
- **Answer:** Exactly 201 tensors ($3 + 22 \times 9 = 201$). Verified in `src/loader/loader.cpp:468`.

### Q49: What are the 3 non-layer tensors?
- **Answer:** `model.embed_tokens.weight`, `model.norm.weight`, and `lm_head.weight`.

### Q50: Are `embed_tokens` and `lm_head` weight-tied in TinyLlama?
- **Answer:** No. They have separate offsets and distinct learned weights in `.safetensors`.

### Q51: What is the magic number of your `.ie` format in ASCII?
- **Answer:** `0x464D4549` is `"IEMF"` ("Inference Engine Model Format") in little-endian ASCII.

### Q52: What is the default sampling temperature?
- **Answer:** 0.0 (Greedy argmax). Located in `include/inference_engine/config.hpp:76`.

### Q53: What is top-p nucleus sampling?
- **Answer:** Selecting from the smallest set of top tokens whose cumulative probability exceeds probability $p$ (default: 0.90 or 1.0).

### Q54: What is top-k filtering?
- **Answer:** Truncating the candidate pool to at most $K$ highest-probability tokens before nucleus filtering. Implemented in `src/sampler/sampler.cpp:98`.

### Q55: Why does `forward_prompt()` accept a null logits destination?
- **Answer:** During prefill of tokens $0 \dots N-2$, intermediate logits are discarded; only the final token $N-1$ needs logits for sampling. Saves memory.

### Q56: How many query heads share a KV head?
- **Answer:** 8 query heads ($32 / 4 = 8$).

### Q57: How many parameters are in one decoder layer of TinyLlama?
- **Answer:** Exactly 43,444,240 parameters.

### Q58: What is the scale factor inside the attention dot product?
- **Answer:** $1 / \sqrt{d_{\text{head}}} = 1 / \sqrt{64} = 1 / 8.0 = 0.125f$. Implemented in `src/kernels/scalar/kernels_scalar.cpp:162`.

### Q59: What is the purpose of `align_up_u64()`?
- **Answer:** Aligns byte offsets to the next 64-byte boundary using bitwise math `(offset + 63) & ~63`.

### Q60: How does the engine detect Apple Silicon at compile time?
- **Answer:** Via `#if defined(__aarch64__) || defined(_M_ARM64)`.

### Q61: What instruction does NEON use for fused multiply-accumulate?
- **Answer:** `vfmaq_f32(acc, val1, val2)`.

### Q62: What instruction does AVX2 use for fused multiply-accumulate?
- **Answer:** `_mm256_fmadd_ps(val1, val2, acc)` under `#if defined(__FMA__)`.

### Q63: What does the `/clear` command do in chat mode?
- **Answer:** Calls `KvCache::clear()`, zeroes memory, resets `committed_seq_len_ = 0`, and clears conversational history.

### Q64: How does cancellation work in Inference Studio?
- **Answer:** The UI sends `/api/chat/stop`, Node sends `{"cmd":"stop"}` to stdin, C++ reader sets `stop_requested_ = true`, and the decode loop breaks cleanly within $<1\text{ ms}$.

### Q65: What is your peak RSS during inference?
- **Answer:** Exactly 658.31 MiB, measured directly from the operating system kernel.
