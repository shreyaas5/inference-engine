# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

> **Last Updated:** 2026-11-01  
> **Consolidated Baseline:** `integration/m4-full-inference` (`e46c06e`)  
> **Active Milestone:** Milestone M5 (Real TinyLlama 22-Layer End-to-End Streaming Delivery)

---

## 1. Completed Milestones & Tasks (`[x]`)

- [x] **M0 (`cb85411`):** Repository scaffold, CMake build system, frozen `TinyLlamaConfig`, `.ie` v1 format specification (`ADR-001`..`ADR-008`).
- [x] **M1 — Member S (`aa40368`):** `MmapModelLoader`, overflow/overlap/alignment validation, `bind_tinyllama_weights()`, `FP32` converter, `loader_validation` CTest (`6/6`).
- [x] **M1 — Member U (`5115467`):** Scalar reference kernels (`rmsnorm`, `rope_inplace`, `attention_gqa`, `swiglu`, `matmul_f32`), `KvCache` (`append` vs `commit_token`), 16-step `FP32` `TransformerEngine::forward_token()`, `generate_golden_reference.py`, `kernels_scalar` (`7/7`), `kv_cache` (`4/4`), `golden_forward` (`3/3`).
- [x] **M2 — Member S (`cb176b3`, `4515579`):** `Q4_0` and `--keep-embeddings-fp32` converter, `TensorView::data_q4_0()`, unified `TransformerWeightBindings` (`ADR-009`), live 201-tensor inspection and 9-role real `Q4_0` error audit (`inspect_checkpoint.py --eval-real-quant-error`).
- [x] **M2 — Member U (`955bd91`, `ce1026a`):** `Q4_0` and mixed-precision `TransformerEngine::forward_token()`, single-row `Q4_0` embedding dequantisation, `dispatch_matmul_*` + `ThreadPool` wiring, `transformer_q4` CTest (`7/7`), `verify_transformer_q4.py` (`cos_sim >= 0.996359`).
- [x] **M2/M3 — Member I (`f76ab55`, `58c604d`, `b05c4e4`, `1c046cb`):** `Q4_0` block packing, FP16$\leftrightarrow$FP32, NEON & AVX2 SIMD kernels, `ThreadPool` with `std::exception_ptr` (`ADR-010`), native Ryzen 7 7745HX verification, AVX2 2-row register tiling + vectorized scale accumulation + contiguous 2-row `dispatch.cpp` chunk scheduling + `inference_engine_bench` CSVs (`ADR-012`), ARM64 NEON 2-row register tiling port & benchmark calculation audit (`1c046cb`).
- [x] **M2/M3 — Member P (`ce769c8`, `5929a17`, `33d4bdc`, `758abac`, `b96b805`, `da21d62`):** SentencePiece C++ `Tokenizer` (dual processor, special tokens, Zephyr chat template, `15/15` parity on real `tokenizer.model`, subsegment encoding), `Sampler` (greedy, temp, top-p, 64-bit `xorshift*`), CLI generation loop & `--benchmark` harness, GitHub Actions CI (`ubuntu-22.04` + `macos-latest`).
- [x] **M4 Consolidation (`97cf8c3`, `e46c06e`):** Consolidated U, I, P feature branches into `integration/m4-full-inference`, tracking `origin/integration-m4-full-inference`, opened PR #6 with 100% CI pass.
- [x] **M5 Checkpoint & Conversion:** Acquired full `TinyLlama/TinyLlama-1.1B-Chat-v1.0` weights (`model.safetensors`, `2,200,119,864` bytes). Converted and validated `models/tinyllama-1.1b-q4_0.ie` (`619,113,792` bytes) and `models/tinyllama-1.1b-fp32.ie` (`4,400,213,312` bytes).
- [x] **M5 End-to-End Real Inference:** Executed full 22-layer real forward pass with real-time streaming detokenization across 5 test prompts. Verified coherent English text output, zero NaNs, and zero memory corruption.
- [x] **M5 Hardware Benchmarking:** Verified thread scaling (`1, 2, 4, 8` threads) and SIMD microbenchmarks on Apple Silicon NEON (up to 36 tok/s decode, 114.2 GFLOPS).
- [x] **M6 Productisation & Release:** Repository-wide branch synchronisation (zero lost commits); interactive multi-turn chat REPL (`BIOS> `); KV-cache prefix matching and `KvCache::truncate()`; Top-K candidate filtering in Sampler; `--bench-runs <N>` multi-iteration statistical summary; auto-discovery of default model/tokenizer; robust error diagnostics; comprehensive documentation overhaul in `README.md` and `DECISIONS.md` (`ADR-013`). Product Ready.

---

## 2. Active Priority Backlog (`P0` $\rightarrow$ `P2`)

### `P0` — Baseline Merge
- [ ] **`P0-1` (Owner: Lead):** Merge PR #6 (`integration-m4-full-inference`) into `main` after review.

### `P1` — Cross-Platform Validation
- [ ] **`P1-1` (Owner: I):** Run `inference_engine_bench` and real-model CLI benchmarks on x86_64 AVX2 machine to complete hardware matrix.

### `P2` — Polish & Future Enhancements
- [ ] **`P2-1` (Owner: U):** Parallelize `attention_gqa` for long contexts (`pos >= 256`) via ThreadPool (`KI-014`).
- [ ] **`P2-2` (Owner: S):** Optional `posix_madvise(..., POSIX_MADV_WILLNEED)` prefault flag in `MmapModelLoader::open()`.
