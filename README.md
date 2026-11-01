# `inference-engine`

A custom C++17 CPU inference engine for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B parameters), engineered from scratch for commodity laptop processors using `Q4_0` block quantisation, ARM64 NEON, x86-64 AVX2, and a persistent C++17 thread pool.

Developed for BIOS V2 — Problem Statement **PS08 (Own Inference Engine)**.

---

## 1. Highlights & Key Innovations

- **Zero External Inference Runtimes:** Executes 100% in custom C++17. Zero runtime dependencies on PyTorch, LibTorch, TensorFlow, ONNX Runtime, `llama.cpp`, GGML, or MLX.
- **Self-Describing `.ie` Format (v1):** Memory-mapped (`mmap`) zero-copy binary format featuring a 64-byte little-endian header, variable tensor directory, and 64-byte cache-line aligned tensor payloads (`payload_offset % 64 == 0`).
- **SIMD Micro-Kernels:** Handwritten ARM64 NEON and x86-64 AVX2/FMA kernels with 2-row register tiling, vectorized scale accumulation, and scalar fallbacks. Achieves **2.7x – 11.6x speedup** over scalar execution.
- **Quantisation:** `Q4_0` block quantisation (32 weights per block, 18 bytes: 2-byte IEEE 754 FP16 scale + 16 packed nibble bytes), compressing the 1.1B model from **4.4 GB to 619 MB** (~7.1x memory reduction).
- **Persistent Thread Pool:** Low-overhead cache-line aligned work-stealing worker pool designed for low core counts (2–8 threads) without runtime heap allocation during generation.
- **Interactive Multi-Turn Chat:** Built-in interactive REPL (`--chat`) with automatic model/tokenizer discovery, streaming token output, prefix-matching KV cache reuse across conversational turns, and context limit enforcement.
- **Inference Studio Workstation:** Full-stack interactive terminal workstation with real-time SVG throughput charting, live KV-cache monitor, complete 201-tensor inspector, benchmark lab, and streaming runtime logs—powered 100% by the custom C++ engine over persistent JSON-lines stdio.
- **Flexible Sampling:** Deterministic greedy, nucleus ($top\text{-}p$), and candidate-filtered ($top\text{-}k$) sampling.

---

## 2. Target Model Architecture (`TinyLlama-1.1B-Chat-v1.0`)

| Parameter | Value | Description |
|---|---|---|
| **Layers** (`num_layers`) | `22` | Transformer decoder layers |
| **Hidden dimension** (`hidden_dim`) | `2048` | Activation embedding width |
| **Intermediate dimension** (`intermediate_dim`) | `5632` | SwiGLU MLP feed-forward hidden dimension |
| **Attention Heads** | `32` query / `4` KV | Grouped Query Attention (GQA 8:1 ratio) |
| **Head Dimension** (`head_dim`) | `64` | Dimension per head ($2048 / 32 = 64$) |
| **Vocabulary Size** (`vocab_size`) | `32000` | Pinned SentencePiece BPE vocabulary |
| **Max Sequence Length** (`max_seq_len`) | `2048` | Maximum causal context window |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Applied pre-attention, pre-FFN, and final |
| **Position Encoding** | RoPE ($\theta = 10000.0$) | Precomputed $\cos/\sin$ table lookup cached kernel |
| **Weight Formats** | Pure `Q4_0` & `FP32` | 619 MB (`Q4_0`) or 4.4 GB (`FP32`) |

---

## 3. High-Level Architecture

```text
                                  +---------------------------------------+
                                  |   Hugging Face TinyLlama Checkpoint   |
                                  |        (model.safetensors, BF16)      |
                                  +---------------------------------------+
                                                      |
                                                      | tools/convert_to_inference_engine.py
                                                      v
                                  +---------------------------------------+
                                  |    Custom .ie Format (v1 Binary)      |
                                  |    - 64-byte Header (Magic: IEMF)     |
                                  |    - 64-byte Aligned Tensors          |
                                  |    - Q4_0 (619 MB) or FP32 (4.4 GB)   |
                                  +---------------------------------------+
                                                      |
                                                      | mmap() zero-copy
                                                      v
+------------------------+        +---------------------------------------+
|  SentencePiece C++     |        |              ModelLoader              |
|  Tokenizer (BPE 32K)   |        |   Validates tensors & memory-maps     |
+------------------------+        +---------------------------------------+
            |                                         |
            | prompt tokens                           | tensor bindings
            v                                         v
+-------------------------------------------------------------------------+
|                           TransformerEngine                             |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  | Causal Loop: For each layer (0 .. 21):                            |  |
|  |   1. RMSNorm -> 2. QKV Proj (dispatch_matmul_q4_0_f32)            |  |
|  |   3. Cached RoPE -> 4. GQA Attention (8:1) via KvCache            |  |
|  |   5. O Proj -> Residual -> 6. RMSNorm                             |  |
|  |   7. Gate + Up Proj -> SwiGLU Activation -> Down Proj -> Residual |  |
|  +-------------------------------------------------------------------+  |
|  | Final RMSNorm -> LM Head Matmul -> Unnormalized Logits            |  |
+-------------------------------------------------------------------------+
       |                                             ^
       | logits [32000]                              | append / commit_token
       v                                             |
+------------------------+        +---------------------------------------+
|   Sampler (Top-K,      |        |                KvCache                |
|    Top-P, Greedy)      |------->|  Stores K/V vectors (22 layers)       |
|  -> Selected Token     | token  |  Prefix-matching multi-turn reuse     |
+------------------------+        +---------------------------------------+
            |
            v
+------------------------+
|  Streaming Detokenizer |
|  & Interactive REPL    | ===> stdout (BIOS>)
+------------------------+
```

---

## 4. Quick Start: One-Command Interactive Chat

To start an interactive chat session with auto-detection of the converted `Q4_0` model and tokenizer:

```bash
# 1. Build release binary
cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON
cmake --build build --parallel 4

# 2. Launch interactive multi-turn chat (auto-detects model and tokenizer in models/)
./build/inference-engine --chat --threads 4
```

Example session:

```text
BIOS V2 — Own Inference Engine

Model: TinyLlama 1.1B Chat
Format: Q4_0
Backend: NEON
Threads: 4

Type /help for available commands, /clear to reset context, /exit to quit.

BIOS> Explain how a CPU works.

BIOS:
A Central Processing Unit (CPU) is the primary component of a computer that performs
most of the processing inside the computer. It executes instructions of a computer
program, such as arithmetic, logical, control, and input/output (I/O) operations...

BIOS> What are its main components?

BIOS:
The main components of a CPU include:
1. Arithmetic Logic Unit (ALU): Performs basic mathematical and logical calculations.
2. Control Unit (CU): Directs the operation of the processor.
3. Registers: High-speed internal memory cells used for temporary storage.
4. Cache: Fast memory providing high-speed data access to the processor cores...

BIOS> /exit
```

### In-Chat Commands

- `/help` — Display list of available interactive commands.
- `/clear` — Reset conversation history and clear the KV cache.
- `/exit` or `/quit` — Cleanly exit the session.

---

## 5. Command-Line Options & Batch Inference

```text
inference-engine (TinyLlama-1.1B-Chat CPU inference runtime)
Usage: inference-engine --model <file.ie> --tokenizer <tokenizer.model> [--prompt <text> | --chat] [options]

Options:
  --model <path>       Path to FP32 or Q4_0 .ie model file (auto-detected if omitted)
  --tokenizer <path>   Path to TinyLlama SentencePiece tokenizer.model (auto-detected if omitted)
  --prompt <text>      Input prompt text (omit with --chat for interactive REPL)
  --chat               Format with Zephyr chat template (interactive if --prompt omitted)
  --threads <n>        Requested worker thread count (default: 1)
  --max-tokens <n>     Maximum new tokens (default: 64; 0 disables generation)
  --temp <value>       Sampling temperature (default: 0, greedy deterministic)
  --top-p <value>      Nucleus probability in (0, 1] (default: 1.0)
  --top-k <n>          Top-k candidate filtering (default: 0, disabled)
  --seed <n>           Reproducible sampling seed (default: 42)
  --benchmark          Print measured load, TTFT, throughput, RSS, and CPU metrics
  --bench-runs <n>     Run benchmark across <n> iterations and report statistical summary
  --encode-only        Print SentencePiece token IDs and exit
  --decode-only        Decode IDs supplied with --token-ids and exit
  --token-ids <ids>    Space-separated IDs for --decode-only
  --add-bos            Add BOS in --encode-only mode
  --add-eos            Add EOS in --encode-only mode
  --format-chat        Print the formatted user chat prompt and exit
  --info               Print detected CPU backend and frozen model config
  --help, -h           Show this help message
```

### Examples

```bash
# Single-turn generation with streaming output
./build/inference-engine \
  --model models/tinyllama-1.1b-q4_0.ie \
  --tokenizer models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
  --chat \
  --prompt "Write a short poem about compiler optimization." \
  --threads 4 \
  --max-tokens 128 \
  --temp 0.7 --top-p 0.9 --top-k 40

# Multi-run benchmark statistical summary (mean, median, min, max across 5 runs)
./build/inference-engine \
  --chat \
  --prompt "Explain entropy." \
  --threads 4 \
  --max-tokens 64 \
  --bench-runs 5
```

---

## 6. Inference Studio (Interactive Workstation Frontend)

Transform the engine into a full-stack local workstation with a high-density terminal interface:

- **Three-Panel Layout:** Navigation Sidebar, Centre View (Inference Playground, Benchmark Lab, Model Inspector, Runtime Logs), and Right Observability Panel.
- **Real-Time Telemetry:** Live decode throughput gauge with target baseline comparison (35 tok/s), TTFT latency in ms, prompt prefill TPS, resident memory (RSS in MiB), and rolling SVG throughput curve.
- **Model Inspector:** Detailed architecture breakdown (22 layers, GQA 8:1, SwiGLU, RoPE theta 10000.0, zero-copy `mmap`) and searchable directory of all 201 model tensors.
- **Dedicated Benchmark Lab:** Multi-iteration benchmark execution with statistical aggregation (mean, median, min, max, stddev) and pass/fail baseline comparison.
- **Zero External Inference Runtimes:** The web workstation communicates exclusively with `./build/inference-engine --studio-worker` via persistent standard input/output JSON-lines streams and Server-Sent Events (SSE).

### Quickstart

```bash
# 1. Build C++ engine (if not already built)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON
cmake --build build --parallel 4

# 2. Launch Inference Studio
cd studio
npm install
npm run build
npm start

# 3. Open workstation in browser
# http://localhost:3000
```

---

## 7. Performance Benchmarks

Measured on an Apple Silicon M-series laptop CPU (ARM64 NEON, 4 threads, `TinyLlama-1.1B-Chat`):

| Metric | `Q4_0` Quantised | Bit-Identical `FP32` |
|---|---|---|
| **On-Disk File Size** | **619 MB** | 4.4 GB |
| **Peak Resident Memory (RSS)** | **658.2 MiB** | 3,956.6 MiB |
| **Decode Throughput** | **35.25 tokens/s** | 16.10 tokens/s |
| **Prefill Throughput** | **35.65 tokens/s** | 3.40 tokens/s |
| **Time to First Token (TTFT)** | **556 ms** (20 prompt tokens) | 6,096 ms |
| **Kernel Acceleration** | 2-row NEON tiled matmul | 1-row NEON dot product |

---

## 8. Model Conversion Instructions

Use the included offline conversion utility (`tools/convert_to_inference_engine.py`) to convert any official TinyLlama checkpoint:

```bash
# 1. Download checkpoint from Hugging Face
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/TinyLlama-1.1B-Chat-v1.0

# 2. Convert to custom Q4_0 .ie format
python3 tools/convert_to_inference_engine.py \
  --input-dir models/TinyLlama-1.1B-Chat-v1.0 \
  --output models/tinyllama-1.1b-q4_0.ie \
  --dtype q4_0

# 3. (Optional) Convert to FP32 .ie format
python3 tools/convert_to_inference_engine.py \
  --input-dir models/TinyLlama-1.1B-Chat-v1.0 \
  --output models/tinyllama-1.1b-fp32.ie \
  --dtype fp32
```

---

## 9. Verification & Test Suite

All 9 deterministic CTest suites and Python parity validations pass:

```bash
# Run all 9 unit and integration test suites
ctest --test-dir build --output-on-failure
```

```text
1/9 Test #1: loader_validation ................   Passed
2/9 Test #2: quant_packing ....................   Passed
3/9 Test #3: kernels_scalar ...................   Passed
4/9 Test #4: kernels_simd .....................   Passed
5/9 Test #5: kv_cache .........................   Passed
6/9 Test #6: tokenizer ........................   Passed
7/9 Test #7: sampler ..........................   Passed
8/9 Test #8: golden_forward ...................   Passed
9/9 Test #9: cli_integration ..................   Passed

100% tests passed out of 9
```

Python Parity & Tool Self-Tests:
```bash
python3 tools/inspect_checkpoint.py --print-frozen-contract
python3 tools/convert_to_inference_engine.py --self-test
python3 tools/verify_tokenizer_parity.py \
  --tokenizer-model models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
  --tokenizer-config models/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json \
  --engine ./build/inference-engine
```

---

## 9. License & Competition Attribution

Developed for BIOS V2 — PS08 (Own Inference Engine). Licensed under the MIT License.
