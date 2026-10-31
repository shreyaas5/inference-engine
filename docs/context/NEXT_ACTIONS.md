# `NEXT_ACTIONS.md` — Dependency-Ordered Engineering Backlog

> **Last Updated:** 2026-11-01
> **Consolidated Baseline:** `integration-ps08-team-sync-2026-10-31` (`ad246cb`)
> **Active Feature Branch:** `feature/cpu-kernel-optimisation` (Member I)

---

## 1. Completed Milestones & Tasks (`[x]`)

- [x] **M0 (`cb85411`):** Repository scaffold, CMake build system, frozen `TinyLlamaConfig`, `.ie` v1 format specification (`ADR-001`..`ADR-008`).
- [x] **M1 — Member S (`aa40368`):** `MmapModelLoader`, overflow/overlap/alignment validation, `bind_tinyllama_weights()`, `FP32` converter, `loader_validation` CTest (`6/6`).
- [x] **M1 — Member U (`5115467`):** Scalar reference kernels (`rmsnorm`, `rope_inplace`, `attention_gqa`, `swiglu`, `matmul_f32`), `KvCache` (`append` vs `commit_token`), 16-step `FP32` `TransformerEngine::forward_token()`, `generate_golden_reference.py`, `kernels_scalar` (`7/7`), `kv_cache` (`4/4`), `golden_forward` (`3/3`).
- [x] **M2 — Member S (`cb176b3`, `4515579`):** `Q4_0` and `--keep-embeddings-fp32` converter, `TensorView::data_q4_0()`, unified `TransformerWeightBindings` (`ADR-009`), live 201-tensor inspection and 9-role real `Q4_0` error audit (`inspect_checkpoint.py --eval-real-quant-error`).
- [x] **M2 — Member U (`955bd91`, `ce1026a`):** `Q4_0` and mixed-precision `TransformerEngine::forward_token()`, single-row `Q4_0` embedding dequantisation, `dispatch_matmul_*` + `ThreadPool` wiring, `transformer_q4` CTest (`7/7`), `verify_transformer_q4.py` (`cos_sim >= 0.996359`).
- [x] **M2/M3 — Member I (`f76ab55`, `58c604d`, `b05c4e4`, `1c046cb`):** `Q4_0` block packing, FP16$\leftrightarrow$FP32, NEON & AVX2 SIMD kernels, `ThreadPool` with `std::exception_ptr` (`ADR-010`), native Ryzen 7 7745HX verification, AVX2 2-row register tiling + vectorized scale accumulation + contiguous 2-row `dispatch.cpp` chunk scheduling + `inference_engine_bench` CSVs (`ADR-012`), ARM64 NEON 2-row register tiling port & benchmark calculation audit (`1c046cb`).
- [x] **M2/M3 — Member P (`ce769c8`, `5929a17`, `33d4bdc`, `758abac`, `b96b805`, `da21d62`):** SentencePiece C++ `Tokenizer` (dual processor, special tokens, Zephyr chat template, `15/15` parity on real `tokenizer.model`, subsegment encoding), `Sampler` (greedy, temp, top-p, 64-bit `xorshift*`), CLI generation loop & `--benchmark` harness, GitHub Actions CI (`ubuntu-22.04` + `macos-latest`).

---

## 2. Active Priority Backlog (`P0` $\rightarrow$ `P3`)

### `P0` — Project-Blocking Integration & Full-Checkpoint Verification
- [ ] **`P0-1` (Owner: S):** Merge Member U's `origin/feature/transformer-runtime` (`e64078c` — GCC 16 `-include cstdint` fix for `sentencepiece-static`) and Member I's `origin/feature/cpu-kernel-optimisation` (`b05c4e4`, `1c046cb` — AVX2/NEON 2-row register tiling, vectorized scale accumulation, contiguous 2-row `dispatch.cpp` chunking, `inference_engine_bench`) into the consolidated integration baseline while preserving `tests/test_tokenizer.cpp` and reconciled docs (`KI-015`).
- [ ] **`P0-2` (Owner: S):** Resume and complete the download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2,200,120,064` bytes total; currently `325,472,256` bytes on disk) (`KI-004`), verify with `python3 tools/inspect_checkpoint.py --checkpoint-dir models/TinyLlama-1.1B-Chat-v1.0`, and export all three 22-layer `.ie` models:
  - `models/tinyllama-fp32.ie` (`4,400,213,312` bytes)
  - `models/tinyllama-q4_0.ie` (`619,113,792` bytes)
  - `models/tinyllama-q4_0-mixed.ie` (`1,069,673,792` bytes)
- [ ] **`P0-3` (Owner: U):** Once `model.safetensors` is complete, run `python3 tools/generate_golden_reference.py --model-id models/TinyLlama-1.1B-Chat-v1.0 --output-dir tests/fixtures/golden` and execute full 22-layer `TransformerEngine::forward_token` parity validation on `FP32`, pure `Q4_0`, and mixed `Q4_0` checkpoints.
- [ ] **`P0-4` (Owner: P):** Once the 22-layer `.ie` models are exported, run the full 22-layer TinyLlama CLI generation & `--benchmark` matrix (`1, 2, 4, 8` threads; `FP32` vs `Q4_0`) and populate `docs/BENCHMARKING.md` with measured TTFT, prefill tok/s, decode tok/s, CPU utilization %, and Peak RSS MB.

### `P1` — Required for Next Major Milestone (Unblocked Immediately)
- [x] **`P1-1` (Owner: I):** Port vectorized per-block scale accumulation (single end-of-row `vaddvq_f32`) and 2-row register tiling (`m` and `m + 1`) from `src/kernels/avx2/kernels_avx2.cpp` to `src/kernels/neon/kernels_neon.cpp` (`neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32`, `neon::matmul_f32`) (`KI-016` resolved in `1c046cb`).
- [x] **`P1-2` (Owner: I):** Update `benchmarks/bench_kernels.cpp` to select the active SIMD backend dynamically via `detect_best_backend()` instead of hardcoding `SimdBackend::Avx2`, support `--backend <scalar|neon|avx2>`, and support comma-separated `--threads` sweeps (`KI-016` resolved).
- [ ] **`P1-3` (Owner: P):** Implement incremental per-token streaming detokenisation inside the decode loop in `src/app/main.cpp` using prefix-diff decoding (`KI-017`).

### `P2` — Important Engineering Improvements
- [ ] **`P2-1` (Owner: U):** Precompute RoPE $(\cos\theta, \sin\theta)$ lookup tables (`[max_seq_len, head_dim / 2]`) in `TransformerEngine::init()` (`KI-006`) and parallelize `attention_gqa` across query heads for long contexts (`KI-014`).
- [x] **`P2-2` (Owner: I):** Add an adaptive thread/chunk cap in `src/kernels/dispatch.cpp` for small output dimensions (`out_dim <= 256`) where 16T SMT wake-up overhead exceeds 8T physical-core latency, and reconcile the minor prefetch CSV text figures and machine-local links in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` (`KI-018` resolved).
- [ ] **`P2-3` (Owner: P):** Add `--bench-runs <N>` (multi-iteration warmup + median/p95 summary) to `src/app/main.cpp` and add an automated synthetic CLI smoke test to CTest.

### `P3` — Optional Polish
- [ ] **`P3-1` (Owner: S):** Optional `posix_madvise(..., POSIX_MADV_WILLNEED)` prefault flag in `MmapModelLoader::open()`.
- [ ] **`P3-2` (Owner: P):** Optional `--top-k` / `--repeat-penalty` sampling parameters in `Sampler`.
