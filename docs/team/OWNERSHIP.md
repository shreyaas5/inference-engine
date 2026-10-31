# Team Ownership & Responsibilities

Software module ownership is structured across four engineers (**S**, **I**, **U**, **P**) with clean header boundaries in `include/inference_engine/` so all four members can work in parallel without merge conflicts.

## Summary Matrix

| Member | Role | Development Hardware | Primary Branch | Owned Paths |
|---|---|---|---|---|
| **S** | Project Lead & Model Loader | Apple MacBook Air M4, 24 GB RAM, 1 TB SSD | `feature/model-loader` | `src/loader/`, `src/common/`, `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`, `tests/test_loader_validation.cpp`, `CURRENT_STATUS.md` |
| **I** | Quantisation & CPU Optimisation | AMD Ryzen 7 7745HX, RTX 4060, 16 GB RAM | `feature/quantisation-simd` | `src/kernels/quant_q4_0.cpp`, `src/kernels/avx2/`, `src/kernels/neon/`, `src/kernels/threading/`, `src/kernels/dispatch.cpp`, `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp` |
| **U** | Scalar Inference & Transformer | Intel Core Ultra 7 155H, 16 GB DDR5 | `feature/transformer` | `src/kernels/scalar/`, `src/core/kv_cache.cpp`, `src/transformer/`, `tests/test_kernels_scalar.cpp`, `tests/test_kv_cache.cpp`, `tests/test_golden_forward.cpp`, `tools/generate_golden_reference.py` |
| **P** | Tokenizer, Runtime & Benchmarks | Intel Core i5-10210U, 8 GB RAM | `feature/runtime-tokenizer` | `src/tokenizer/`, `src/sampler/`, `src/app/`, `tools/verify_tokenizer_parity.py`, `tests/test_tokenizer.cpp`, `.github/workflows/ci.yml` |

---

## Detailed Member Responsibilities

### S — Project Lead & Model Loader
- Inspect `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint structure and tensor inventory.
- Define and implement offline model conversion (`tools/convert_to_inference_engine.py`) for FP32 (M1) and Q4_0 (M2).
- Implement `.ie` v1 binary parser, writer, and POSIX `mmap` loader (`src/loader/loader.cpp`).
- Enforce checked integer arithmetic, 64-byte alignment, bounds checking, and overlap detection (`tests/test_loader_validation.cpp`).
- Maintain public integration contracts, review/merge PRs into `integration` and `main`, and keep `CURRENT_STATUS.md` up to date.
- Execute ARM64 NEON validation and benchmarks on Apple M4 hardware.

### I — Quantisation & CPU Optimisation
- Implement upstream-compatible `Q4_0` quantisation, dequantisation, and IEEE 754 FP16 conversions (`src/kernels/quant_q4_0.cpp`).
- Implement x86-64 `AVX2` + `FMA` kernels (`src/kernels/avx2/`) and ARM64 `NEON` kernels (`src/kernels/neon/`, coordinated with S's M4 for hardware testing).
- Implement the persistent C++17 `std::thread` pool (`src/kernels/threading/thread_pool.cpp`) and runtime CPU feature detection / kernel dispatch (`src/kernels/dispatch.cpp`).
- Verify SIMD numerical parity against U's scalar reference (`tests/test_kernels_simd.cpp`) and lead kernel micro-benchmarking.

### U — Scalar Inference & Transformer
- Implement trusted FP32 scalar reference kernels (`src/kernels/scalar/`): `rmsnorm_f32`, `rope_inplace_f32`, `matmul_f32`, `matmul_q4_0_f32`, `swiglu_f32`, `softmax_f32`, and GQA attention.
- Implement `KvCache` (`src/core/kv_cache.cpp`) with strict separation between `append()` and `commit_token()`.
- Implement the 22-layer `TransformerEngine` forward pass (`src/transformer/transformer.cpp`).
- Generate Hugging Face FP32 golden references (`tools/generate_golden_reference.py`) and validate multi-token prefill and autoregressive greedy decoding (`tests/test_golden_forward.cpp`).

### P — Tokenizer, Runtime & Benchmarks
- Integrate SentencePiece C++ into `src/tokenizer/` and implement the TinyLlama Zephyr chat template.
- Verify 100% token ID, BOS/EOS, byte-fallback, and decode parity against Hugging Face (`tools/verify_tokenizer_parity.py`, `tests/test_tokenizer.cpp`).
- Implement greedy and temperature/top-p sampling (`src/sampler/`).
- Implement the CLI application (`src/app/main.cpp`) and benchmark runner.
- Maintain `.github/workflows/ci.yml` and validate low-memory (8 GB RAM) execution on Intel Core i5-10210U.
