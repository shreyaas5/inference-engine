# `PROJECT_CONTEXT.md` — Short Project Identity

**Last Verified:** 2026-10-31

## 1. Identity & Competition Context

- **Project / Repository Name:** `inference-engine`
- **Competition:** BIOS V2
- **Problem Statement:** PS08 — Own Inference Engine
- **Local Path:** `/Users/shreyaas/Developer/inference-engine`
- **Remote Repository URL:** https://github.com/shreyaas5/inference-engine (Verified, Private)
- **Primary Objective:** Design, implement, validate, and benchmark a custom, zero-framework C++17 CPU inference engine capable of running a 1.1-billion-parameter language model (`TinyLlama-1.1B-Chat-v1.0`) efficiently on commodity laptops across ARM64 and x86-64 architectures.

## 2. Locked Target Model

- **Hugging Face Model ID:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Architecture:** Decoder-only Transformer (`LlamaForCausalLM`)
- **Hyperparameters:**
  - `num_layers = 22`
  - `hidden_dim = 2048`
  - `intermediate_dim = 5632` (SwiGLU MLP)
  - `num_query_heads = 32`, `num_kv_heads = 4` (Grouped Query Attention, 8 Q heads per KV head), `head_dim = 64`
  - `vocab_size = 32000` (SentencePiece BPE, Zephyr chat template)
  - `max_seq_len = 2048`
  - `rms_norm_eps = 1e-5`, `rope_theta = 10000.0`

## 3. Core Technology Stack

- **Runtime Language:** C++17 (`namespace inference_engine`, executable `inference-engine`)
- **Build & Test System:** CMake ($\ge 3.16$, verified with CMake 4.4.3) and CTest
- **Model File Format:** Custom `.ie` v1 binary format (little-endian, 64-byte header, self-describing tensor directory, 64-byte aligned tensor payloads, POSIX `mmap` zero-copy loading)
- **Numerical Precision:**
  - Activations & KV Cache: `FP32`
  - Weights: `FP32` (Milestone M1 baseline) and `Q4_0` 18-byte / 32-weight block quantisation (Milestone M2+)
- **CPU Acceleration:** Scalar reference kernels (`src/kernels/scalar/`), ARM64 NEON (`src/kernels/neon/`), x86-64 AVX2+FMA (`src/kernels/avx2/`), and a persistent `std::thread` pool (`src/kernels/threading/`)
- **Offline Tooling Only:** Python 3 (`tools/`) for checkpoint inspection, `.ie` conversion, tokenizer parity verification, and Hugging Face golden reference generation.

## 4. Hardware Target Matrix & 4-Member Team Structure

All official benchmarks run **CPU-only inference**.

| Member | Role | Hardware Target | RAM | Target SIMD Backend | Feature Branch |
|---|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4 | 24 GB | ARM64 `NEON` | `feature/model-loader` |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX (RTX 4060 unused) | 16 GB | x86-64 `AVX2` + `FMA` | `feature/quantisation-simd` |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H | 16 GB | x86-64 `AVX2` + `FMA` | `feature/transformer` |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U | 8 GB | x86-64 `AVX2` (runtime checked) | `feature/runtime-tokenizer` |

