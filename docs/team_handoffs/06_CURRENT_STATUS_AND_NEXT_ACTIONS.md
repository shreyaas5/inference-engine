# `06_CURRENT_STATUS_AND_NEXT_ACTIONS.md` — Project-Wide Status Report & Prioritised Backlog

> **Document Role:** Concise, evidence-backed project status report, branch/commit ledger, and dependency-ordered action plan across **S**, **U**, **I**, and **P**.
> **Last Reconciled:** 2026-11-01 (against consolidated branch `integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff` and remote feature branches `b05c4e4` and `e64078c`).

---

## 1. Git Branch & Commit Ledger (`VERIFIED` via `git fetch --all`)

| Branch Name | Exact Commit SHA | Owner | Status & Contents |
| :--- | :--- | :--- | :--- |
| `main` / `origin/main` | `177fe492eb349ed17a2ffaeda908d92a0d7e0935` | Protected | Milestone 1 baseline (`M0` `cb85411` + **S** M1 `aa40368` + **U** M1 `5115467`). **Do not commit directly to `main`.** |
| `integration` / `origin/integration` | `783b8f44e5d2fc2ecf830559139cd19ed8a328fd` | Shared | Milestone 2 partial staging (`main` + **I** M2 `f76ab55`). Not yet fast-forwarded to `33d4bdc`. |
| `integration-ps08-team-sync-2026-10-31` | `33d4bdc4486e83f458b6a242d9a2fd051503e6ff` | **S (All)** | **Consolidated 4-member integration baseline.** Contains **S** (`cb176b3`, `4515579`), **U** (`955bd91`, `ce1026a`), **I** (`f76ab55`, `58c604d`), and **P** (`ce769c8`), plus CMake 4.x `sentencepiece-static` & CI macOS/protobuf fixes (`5929a17`, `33d4bdc`). **`9/9` CTest suites passing.** |
| `origin/feature/cpu-kernel-optimisation` | `b05c4e49bf135342e4f94a43a5e4c45763c802e1` | **I** | **Unmerged remote branch (parent `5929a17`).** Adds AVX2 2-row register tiling (`avx2::matmul_q4_0_f32`, `avx2::matmul_f32`), vectorized scale accumulation (`compute_block_unscaled_dot_vec`), contiguous 2-row `dispatch.cpp` chunk scheduling, `inference_engine_bench` target, Ryzen 7 7745HX CSVs, and `ADR-012`. |
| `origin/feature/transformer-runtime` | `e64078c84e7832deb0444f51ffad40c45154438d` | **U** | **Unmerged remote branch (parent `5929a17`).** Adds `target_compile_options(sentencepiece-static PRIVATE -include cstdint)` in `CMakeLists.txt` to fix SentencePiece compilation on GCC 16. |
| `origin/feature/real-checkpoint-validation` | `4515579672c325812bf42670f63a17b2967f3ef4` | **S** | Merged into `33d4bdc` via `0afcafe`. |
| `origin/feature/model-format-q4` | `cb176b3ec16f0025ad0af2c4e6074fa95804ac52` | **S** | Merged into `33d4bdc`. |
| `origin/feature/transformer-q4` | `ce1026a7ef90511425f3c258db1ae139dfc77dee` | **U** | Merged into `33d4bdc` via `a687095`. |
| `origin/feature/quantisation-simd-m3` | `58c604d035df2a272cbffc682681493dcfdc418c` | **I** | Merged into `33d4bdc` via `991a527`. |
| `origin/feature/runtime-tokenizer` | `ce769c882f0883613183812738340195a1f7d71e` | **P** | Merged into `33d4bdc` via `3b82184`. |

---

## 2. Module-by-Module Evidence Classification Matrix

| Module / Capability | Owner | Classification | Verification Evidence |
| :--- | :--- | :--- | :--- |
| `.ie` v1 Binary Format, `MmapModelLoader`, `bind_tinyllama_weights()` | **S** | `VERIFIED` / `TESTED` | `loader_validation` CTest (`6/6` passed) |
| Safetensors $\rightarrow$ `.ie` Converter (`FP32`, `Q4_0`, `--keep-embeddings-fp32`) | **S** | `VERIFIED` / `TESTED` | `convert_to_inference_engine.py --self-test` (`PASS`) |
| 201-Tensor Contract & Real `Q4_0` Error Profile across 9 Roles | **S** | `REAL-CHECKPOINT VALIDATED` | `inspect_checkpoint.py --checkpoint-dir models/TinyLlama-1.1B-Chat-v1.0 --eval-real-quant-error` (`MeanCosSim >= 0.9888`) |
| Full 2.20 GB `model.safetensors` Download & 22-Layer `.ie` Export | **S** | `BLOCKED` | Local `model.safetensors` is `325,472,256` / `2,200,120,064` bytes (~14.8%); resume download to unblock |
| Scalar Reference Kernels (`rmsnorm`, `rope_inplace`, `attention_gqa`, `swiglu`, `matmul_f32`) | **U** | `VERIFIED` / `TESTED` | `kernels_scalar` CTest (`7/7` passed) |
| Contiguous `FP32` `KvCache` (`append` vs `commit_token`) | **U** | `VERIFIED` / `TESTED` | `kv_cache` CTest (`4/4` passed) |
| 16-Step `TransformerEngine::forward_token` (`FP32`, `Q4_0`, Mixed, `ThreadPool`) | **U** | `SYNTHETICALLY VALIDATED` | `golden_forward` (`3/3`), `transformer_q4` (`7/7`), `verify_transformer_q4.py` (`cos_sim >= 0.996359`) |
| Full 22-Layer Pretrained Checkpoint Forward Pass & Golden Logit Parity | **U** | `BLOCKED` | Blocked on complete 2.20 GB `model.safetensors` download & `.ie` conversion |
| `Q4_0` Block Packing, FP16$\leftrightarrow$FP32, `ThreadPool` (`std::exception_ptr`) | **I** | `VERIFIED` / `TESTED` | `quant_packing` (`4/4`) & `kernels_simd` (`8/8`) CTests |
| AVX2+FMA 2-Row Register Tiling, Vector Scale Accum, 2-Row `dispatch.cpp` Chunking | **I** | `VERIFIED` / `TESTED` | Verified on AMD Ryzen 7 7745HX in `b05c4e4` (`21.97x` on `2048x2048` 8T, `162.80 GFLOPS` on `2048x5632` 16T) |
| ARM64 NEON Intrinsics (`src/kernels/neon/kernels_neon.cpp`) | **I** | `PARTIALLY IMPLEMENTED` / `TESTED` | Working NEON intrinsics pass `kernels_simd` (`MaxAbsDiff <= 9.54e-6`), but 2-row register tiling and deferred horizontal reduction from `b05c4e4` are not yet ported to NEON |
| SentencePiece C++ `Tokenizer` (Dual Processor, Special Tokens, Zephyr Template) | **P** | `REAL-CHECKPOINT VALIDATED` | `tokenizer` CTest (`4/4`) + `verify_tokenizer_parity.py --self-test` (`15/15` prompts exact match against real `tokenizer.model`) |
| `Sampler` (Greedy, Temperature, Top-p, 64-bit `xorshift*`, `NaN`/`Inf` Guards) | **P** | `VERIFIED` / `TESTED` | `sampler` CTest (`6/6` passed) |
| CLI Generation Loop & `--benchmark` Harness (`src/app/main.cpp`) | **P** | `SYNTHETICALLY VALIDATED` | Verified E2E on synthetic 1-layer `.ie` model + real `tokenizer.model`; currently uses batch decode after generation (streaming decode is `P1`) |

---

## 3. Automated Test Suite Status (`9 / 9` CTest Suites Passing on `33d4bdc`)

| # | CTest Suite | Owner | Sub-Tests | Status |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `loader_validation` | **S** | `6 / 6` | `PASSED` (`FP32`, `Q4_0`, alignment, overlap, overflow, `bind_tinyllama_weights`) |
| 2 | `quant_packing` | **I** | `4 / 4` | `PASSED` (`sizeof == 18`, FP16 round-trip, split-half nibble layout, `Q4_0` dot accuracy) |
| 3 | `kernels_scalar` | **U** | `7 / 7` | `PASSED` (`rmsnorm`, `rope_inplace`, `attention_gqa` causal/8:1, `swiglu`, `matmul_f32`) |
| 4 | `kernels_simd` | **I** | `8 / 8` (+ odd-row tails in `b05c4e4`) | `PASSED` (SIMD vs scalar parity, TinyLlama shapes, 1000-cycle `ThreadPool` stress, exception safety) |
| 5 | `kv_cache` | **U** | `4 / 4` | `PASSED` (`append` never increments `seq_len()`, `commit_token`, bounds checks, 22-layer sim) |
| 6 | `tokenizer` | **P** | `4 / 4` | `PASSED` (Real `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model` load, encode/decode, special tokens, Zephyr chat template) |
| 7 | `sampler` | **P** | `6 / 6` | `PASSED` (Greedy argmax, `NaN`/`Inf` handling, temperature scaling, top-p filtering, seeded determinism) |
| 8 | `golden_forward` | **U** | `3 / 3` | `PASSED` (Multi-layer `FP32` prefill/decode parity `MaxAbsDiff = 2.98e-7`, `Q4_0` & mixed forward) |
| 9 | `transformer_q4` | **U** | `7 / 7` | `PASSED` (`FP32` vs `Q4_0` `cos_sim >= 0.996359`, mixed `Q4_0`, SIMD vs scalar, 1T vs 4T determinism) |

---

## 4. Blocking Dependencies & Immediate Next Actions by Member

### Critical Blocker Across S, U, and P (`KI-004`)
- **Incomplete Local Checkpoint Download:** `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` is currently `325,472,256` bytes (`310.4 MB`) out of `2,200,120,064` bytes (`2.20 GB`).
- **Resolution:** Resume the download (`curl -L -C -` or `huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 model.safetensors --local-dir models/TinyLlama-1.1B-Chat-v1.0`) and run `tools/convert_to_inference_engine.py` to generate `models/tinyllama-fp32.ie` (`4,400,213,312` B), `models/tinyllama-q4_0.ie` (`619,113,792` B), and `models/tinyllama-q4_0-mixed.ie` (`1,069,673,792` B).

### Immediate Action Queue per Team Member

| Member | Priority | Immediate Next Action |
| :--- | :---: | :--- |
| **S** | **`P0`** | 1. Merge U's `origin/feature/transformer-runtime` (`e64078c` — GCC 16 `<cstdint>` fix) and I's `origin/feature/cpu-kernel-optimisation` (`b05c4e4` — AVX2 2-row register tiling, vectorized scale accumulation, `dispatch.cpp` 2-row chunking, `inference_engine_bench`) into the consolidated integration branch while preserving P's `tests/test_tokenizer.cpp`.<br>2. Resume download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` to `2,200,120,064` bytes and export `models/tinyllama-{fp32,q4_0,q4_0-mixed}.ie`. |
| **U** | **`P0` / `P2`** | 1. (`P0`, once `model.safetensors` finishes) Run `tools/generate_golden_reference.py` and execute full 22-layer `TransformerEngine::forward_token` parity tests on `FP32`, pure `Q4_0`, and mixed `Q4_0` `.ie` checkpoints.<br>2. (`P2`, unblocked now) Precompute RoPE $(\cos\theta, \sin\theta)$ lookup tables (`[max_seq_len, head_dim / 2]`) in `TransformerEngine::init()` (`KI-006`) and parallelize `attention_gqa` across query heads (`KI-014`). |
| **I** | **`P1` / `P2`** | 1. (`P1`, unblocked now) Port vectorized per-block scale accumulation (single end-of-row `vaddvq_f32`) and 2-row register tiling (`m` and `m + 1`) to `src/kernels/neon/kernels_neon.cpp` (`neon::dot_q4_0_f32`, `neon::matmul_q4_0_f32`, `neon::matmul_f32`).<br>2. (`P1`) Update `benchmarks/bench_kernels.cpp` to use `detect_best_backend()` instead of hardcoding `SimdBackend::Avx2` so NEON benchmarks run natively on Apple Silicon.<br>3. (`P2`) Reconcile the minor prefetch CSV numbers in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md` and add an adaptive thread cap for `256x2048` K/V projections in `dispatch.cpp`. |
| **P** | **`P0` / `P1`** | 1. (`P1`, unblocked now) Implement incremental per-token streaming detokenisation in `src/app/main.cpp` (prefix-diff decoding flushed on each generated token) and optional `--bench-runs <N>` warmup/measurement loop.<br>2. (`P0`, once 22-layer `.ie` models are exported) Execute the full 22-layer TinyLlama-1.1B CLI generation & `--benchmark` matrix across thread counts (`1, 2, 4, 8`) and record real TTFT, prefill tok/s, decode tok/s, and Peak RSS MB in `docs/BENCHMARKING.md`. |
