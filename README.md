# `inference-engine`

A custom C++17 CPU inference engine for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B parameters), engineered for commodity laptop processors using `Q4_0` block quantisation, ARM64 NEON, x86-64 AVX2, and a persistent C++17 thread pool.

## 1. What `inference-engine` Does

`inference-engine` loads a custom, self-describing `.ie` (version 1) binary model file via memory mapping (`mmap`) and executes autoregressive Transformer inference on the CPU without PyTorch, TensorFlow, ONNX Runtime, `llama.cpp`, or any external inference framework.

## 2. Why a Custom Inference Engine

- Full control over memory layout, 64-byte cache-line alignment, and `mmap` zero-copy weight access.
- Deterministic, transparent implementations of RMSNorm, RoPE, Grouped Query Attention (GQA), SwiGLU, and `Q4_0` block matrix-vector multiplication.
- Cross-platform SIMD acceleration targeting Apple Silicon (`NEON`) and x86-64 (`AVX2` + `FMA`) laptops, including low-memory (8 GB RAM) machines.

## 3. Target Model & Architecture (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

| Parameter | Value |
|---|---|
| Layers (`num_layers`) | `22` |
| Hidden dimension (`hidden_dim`) | `2048` |
| Intermediate SwiGLU dimension (`intermediate_dim`) | `5632` |
| Query heads / KV heads (`num_query_heads` / `num_kv_heads`) | `32` / `4` (GQA 8:1) |
| Head dimension (`head_dim`) | `64` |
| Vocabulary size (`vocab_size`) | `32000` |
| Max sequence length (`max_seq_len`) | `2048` |
| Normalization & Positional Encoding | RMSNorm ($\epsilon = 10^{-5}$), RoPE ($\theta = 10000.0$) |
| Runtime Precision | `FP32` activations & KV cache; `FP32` (M1) and `Q4_0` (M2+) weights |

## 4. Technology Stack

- **Runtime:** C++17 (`std::thread`, zero external inference runtime dependencies)
- **SIMD Intrinsics:** ARM64 `arm_neon.h`, x86-64 `immintrin.h` (`AVX2` / `FMA`)
- **Build & Test:** CMake ($\ge 3.16$) and CTest
- **Offline Tooling:** Python 3 (`tools/` for checkpoint inspection, `.ie` conversion, tokenizer parity, and golden reference generation)

## 5. Repository Structure

```text
inference-engine/
├── .github/workflows/ci.yml       # Ubuntu x86-64 & macOS ARM64 CI pipeline
├── docs/                          # Architecture, .ie spec, Q4_0 contract, validation, team docs
├── include/inference_engine/      # Frozen public C++17 headers (namespace inference_engine)
├── src/                           # Modular C++17 implementation scaffolding
│   ├── common/                    # Status, dtype metadata, CPU feature detection, tensor descriptors
│   ├── loader/                    # .ie v1 parser, validator, and mmap loader
│   ├── core/                      # KV cache (explicit append + commit_token contract)
│   ├── transformer/               # 22-layer TinyLlama forward pass scaffolding
│   ├── kernels/                   # Q4_0, scalar, NEON, AVX2, ThreadPool, and runtime dispatch
│   ├── tokenizer/                 # Tokenizer & Zephyr chat template formatting
│   ├── sampler/                   # Deterministic greedy & top-p sampler
│   └── app/                       # CLI entry point (inference-engine executable)
├── tools/                         # Offline Python conversion & validation scripts
├── tests/                         # Deterministic CTest unit suites
└── fixtures/                      # Synthetic test fixture rules (weights excluded via .gitignore)
```

## 6. Build & Test Instructions

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

## 7. Model Conversion Instructions (Offline Python)

```bash
# Inspect local Hugging Face TinyLlama checkpoint
python3 tools/inspect_checkpoint.py --checkpoint-dir /path/to/TinyLlama-1.1B-Chat-v1.0

# Convert checkpoint to .ie v1 format (M1: fp32, M2: q4_0)
python3 tools/convert_to_inference_engine.py \
  --input-dir /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output models/tinyllama-1.1b-q4_0.ie \
  --dtype q4_0
```

## 8. Inference Execution Instructions

```bash
# Inspect detected CPU SIMD backend
./build/inference-engine --info

# Run CLI inference (full end-to-end weight execution arrives in M1/M2)
./build/inference-engine \
  --model models/tinyllama-1.1b-q4_0.ie \
  --tokenizer models/tokenizer.model \
  --prompt "Explain CPU cache locality." \
  --threads 4 \
  --max-tokens 64
```

## 9. Benchmarking Methodology

All official benchmarks run CPU-only across the four team laptops (Apple M4, Ryzen 7 7745HX, Intel Core Ultra 7 155H, Intel Core i5-10210U) measuring model load time, TTFT, prefill tokens/sec, decode tokens/sec, peak RSS, and thread scaling against `llama.cpp` under identical `Q4_0` weights and prompts. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## 10. Current Implementation Status

- **Milestone M0 (Repository Foundation):** **Complete.** CMake build, public headers, contract validations (`.ie` v1 binary loader, `Q4_0` 18-byte packing, `KvCache` `append`/`commit_token` separation, autoregressive position indexing), deterministic CTest suites, and CI are active.
- **Milestones M1–M4:** In progress on feature branches. See [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## 11. Team Ownership

| Member | Role | Branch | Primary Scope |
|---|---|---|---|
| **S** | Project Lead & Model Loader | `feature/model-loader` | `src/loader/`, `src/common/`, `tools/convert_to_inference_engine.py` |
| **I** | Quantisation & CPU Optimisation | `feature/quantisation-simd` | `src/kernels/quant_q4_0.cpp`, `neon/`, `avx2/`, `threading/`, `dispatch.cpp` |
| **U** | Scalar Inference & Transformer | `feature/transformer` | `src/kernels/scalar/`, `src/core/kv_cache.cpp`, `src/transformer/` |
| **P** | Tokenizer, Runtime & Benchmarks | `feature/runtime-tokenizer` | `src/tokenizer/`, `src/sampler/`, `src/app/`, `.github/workflows/ci.yml` |
