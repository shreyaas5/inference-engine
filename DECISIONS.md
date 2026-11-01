# `DECISIONS.md` — Consolidated Architectural Decision Record (ADR)

**Single Source of Truth for Architectural & Contract Decisions**  
*(Note: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) redirects here to prevent duplicate sources of truth.)*

---

## ADR-001: Project Identity, Namespace, and Competition Separation

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** The project is built for the **BIOS V2** competition under **Problem Statement PS08 — Own Inference Engine**. Earlier planning drafts risked conflating the competition name (`BIOS`) with the software repository and C++ identifiers.
- **Decision:**
  - Project & repository name: `inference-engine`
  - C++ namespace: `inference_engine`
  - Static library target: `inference_engine_core`
  - CLI executable target: `inference-engine`
  - Custom model format extension: `.ie`
- **Alternatives Considered:** Naming the binary or namespace `bios` or `tinyllama_engine`. Rejected to comply with strict project naming rules and keep the engine architecture-clean.
- **Consequences:** All headers, CMake targets, CLI banners, and documentation consistently use `inference-engine` / `inference_engine`.

---

## ADR-002: Locked Primary Target Model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** A 1B-class causal language model is required that exercises modern Llama-2/3 architectural primitives (RMSNorm, RoPE, Grouped Query Attention, SwiGLU) while fitting comfortably in commodity laptop RAM (including an 8 GB DDR4 target machine).
- **Decision:** Lock `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`22` layers, `hidden_dim = 2048`, `intermediate_dim = 5632`, `32` query heads, `4` KV heads, `head_dim = 64`, `vocab_size = 32000`, `max_seq_len = 2048`, $\epsilon = 10^{-5}$, $\theta = 10000.0$) as the primary model.
- **Alternatives Considered:** GPT-2 (lacks RoPE, GQA, SwiGLU, RMSNorm) or 7B models (exceed 8 GB RAM in FP32 and leave minimal headroom in Q4_0).
- **Consequences:** All kernel dimensions, GQA `8:1` head-group ratios, and golden validation tests are tailored and verified against `TinyLlama-1.1B-Chat-v1.0`.

---

## ADR-003: Self-Describing Little-Endian `.ie` v1 Binary Format with 64-Byte Alignment

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** The runtime requires fast, zero-copy `mmap` model loading across macOS (ARM64) and Linux/Windows (x86-64) without relying on external serialization libraries or unsafe C++ struct dumps.
- **Decision:**
  - Define `.ie` version `1` with magic `0x464D4549` (`"IEMF"`), explicit little-endian field-by-field serialization, a 64-byte header, and a variable-length tensor directory.
  - Enforce 64-byte alignment (`payload_offset % 64 == 0`) on every tensor payload.
  - Enforce checked 64-bit arithmetic (`checked_add_u64`, `checked_mul_u64`), exact payload byte-length verification, `Q4_0` innermost dimension divisibility by 32, and sorted interval non-overlap checks in `ModelLoader::load_from_buffer()`.
- **Alternatives Considered:** Raw `fwrite` of C++ structs (non-portable across ABI/padding), reading `.safetensors` directly in C++ (requires JSON parser and lacks guaranteed 64-byte alignment for `Q4_0` blocks).
- **Consequences:** Implemented and verified in [`src/loader/loader.cpp`](src/loader/loader.cpp) and [`tests/test_loader_validation.cpp`](tests/test_loader_validation.cpp).

---

## ADR-004: Upstream-Compatible 18-Byte `Q4_0` Block Format (Split-Half Nibble Layout)

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** 4-bit weight quantisation is required to reduce the 1.1B model footprint from ~4.4 GB (`FP32`) to ~630 MB (`Q4_0`) and compare fairly against `llama.cpp`.
- **Decision:**
  - Group weights into contiguous blocks of `32` along the innermost dimension.
  - Store each block in `18` bytes (`BlockQ4_0`): a 2-byte little-endian IEEE 754 FP16 scale `d` followed by `16` packed bytes `qs[0..15]`.
  - Low nibble (`qs[j] & 0x0F`) stores weight index `j` (`0..15`); high nibble (`(qs[j] >> 4) & 0x0F`) stores weight index `j + 16` (`16..31`).
  - Quantisation scale: `d = signed_max / -8.0f`, `q = clamp(floor(x / d + 8.5f), 0, 15)`, with all-zero blocks producing `d = 0.0f` and `q = 8`.
  - Dequantisation: `w_i = (static_cast<int>(q_i) - 8) * fp16_to_fp32(d)`.
- **Alternatives Considered:** Adjacent nibble pairing `(2j, 2j+1)` or per-block min/max asymmetric `Q4_1`. Rejected because split-half `Q4_0` allows single-instruction 16-byte SIMD loads (`vld1q_u8` / `_mm_loadu_si128`) and matches `llama.cpp` `Q4_0`.
- **Consequences:** Implemented and verified in [`src/kernels/quant_q4_0.cpp`](src/kernels/quant_q4_0.cpp) and [`tests/test_quant_packing.cpp`](tests/test_quant_packing.cpp).

---

## ADR-005: Explicit Separation of `KvCache::append()` and `KvCache::commit_token()`

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** During a single token's forward pass, all 22 transformer layers write their projected K and V vectors for the same sequence position `pos`.
- **Decision:**
  - `KvCache::append(layer_idx, pos, k_data, v_data)` writes K/V data for `(layer_idx, pos)` and **never** increments `seq_len()`.
  - `KvCache::commit_token()` (or `commit_positions(count)`) explicitly increments `seq_len()` **once** after all 22 layers have completed processing for the token(s).
- **Alternatives Considered:** Auto-incrementing `seq_len()` inside `append()` on `layer_idx == num_layers - 1`. Rejected as error-prone and opaque during layer-by-layer unit testing.
- **Consequences:** Implemented and verified in [`src/core/kv_cache.cpp`](src/core/kv_cache.cpp) and [`tests/test_kv_cache.cpp`](tests/test_kv_cache.cpp).

---

## ADR-006: Autoregressive Prefill-to-Decode Position Indexing Contract

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** Off-by-one errors between prompt prefill and autoregressive token generation corrupt RoPE angles and KV cache lookups.
- **Decision:**
  - For an $N$-token prompt (e.g., $N = 12$), prompt tokens are processed sequentially at positions `0, 1, ..., N-1` (`0..11`).
  - The logits output by position `N-1` (`11`) predict the **first generated token**.
  - When generation continues, that first generated token is fed into `forward_token()` at position `N` (`12`), and subsequent generated tokens are processed at `N+1` (`13`), `N+2` (`14`), etc.
- **Alternatives Considered:** Re-running the last prompt token during decode init. Rejected as redundant and error-prone.
- **Consequences:** Documented in [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md) and tested in [`tests/test_golden_forward.cpp`](tests/test_golden_forward.cpp).

---

## ADR-007: SentencePiece C++ Integration for Tokenization

- **Date:** 2026-10-31
- **Status:** Accepted (implemented in working tree; build/parity validation pending)
- **Context:** TinyLlama uses a 32,000-token SentencePiece model (`tokenizer.model`) with byte fallback and Zephyr chat tags.
- **Decision:** Integrate the upstream SentencePiece C++ processor in `src/tokenizer/tokenizer.cpp` for M1 rather than writing a custom BPE/Unigram parser from scratch, and verify 100% token ID parity against Hugging Face using `tools/verify_tokenizer_parity.py`.
- **Alternatives Considered:** Custom C++ BPE tokenizer parser. Rejected because SentencePiece whitespace normalization (`▁`) and byte-fallback rules are intricate and non-core to neural matrix-math execution.
- **Consequences:** `Tokenizer` uses the upstream C++ processor, validates the TinyLlama vocabulary (`32000`) and special-token IDs (`0/1/2`), and handles those special tokens separately from ordinary SentencePiece text. `tools/verify_tokenizer_parity.py` performs model-backed Hugging Face comparisons.

---

## ADR-008: Mixed-Precision `Q4_0` Model Export Policy and Unified `TensorView` Weight Bindings

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** Milestone M2 requires exporting and loading quantised `.ie` v1 models (`DataType::Q4_0`) while preserving 1D RMSNorm precision, maintaining backward compatibility with `FP32` `.ie` models, and eliminating duplicate tensor-binding logic between `ModelLoader` and `TransformerEngine` (`KI-010`).
- **Decision:**
  1. **Self-Describing Mixed-Precision `.ie` v1 Files:** Keep `.ie` format version `1` (`kIeFormatVersion = 1`). When exported with `--dtype q4_0`, the 64-byte header stores `weight_dtype = DataType::Q4_0` (`2`), while each tensor directory entry records its own `TensorDesc::dtype`:
     - All 45 1D RMSNorm scale tensors (`*.input_layernorm.weight`, `*.post_attention_layernorm.weight`, `model.norm.weight`) are always stored as `DataType::FP32` (`368,640` bytes total).
     - All 154 2D decoder layer projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are stored as `DataType::Q4_0` (`544,997,376` bytes total).
     - `model.embed_tokens.weight` and `lm_head.weight` (`[32000, 2048]`) default to `DataType::Q4_0` (`36,864,000` bytes each, yielding a `619,113,472`-byte / `619.11 MB` `.ie` file), with optional `--keep-embeddings-fp32` support (`DataType::FP32`, yielding a `1,069,673,472`-byte `.ie` file). `ModelLoader::bind_transformer_weights` validates and binds both variants seamlessly.
  2. **Typed `TensorView` Accessors:** Extend `TensorView` (`include/inference_engine/tensor.hpp`) with `data_q4_0()` (`const BlockQ4_0*`), `data_f32()` (`const float*`), `dtype()`, `ndim()`, `shape()`, `payload_nbytes()`, `num_elements()`, and `num_blocks()`.
  3. **Single Source of Truth for Weight Bindings (`KI-010` Resolved):** Refactor `TransformerEngine` (`include/inference_engine/transformer.hpp`, `src/transformer/transformer.cpp`) to store `TransformerWeightBindings bindings_{}` populated via `ModelLoader::bind_transformer_weights(&bindings_)`, removing private `LayerWeights` and duplicate `bind_tensor()` helpers.
- **Alternatives Considered:** Incrementing `.ie` format version to `v2` (unnecessary because `.ie` v1 already stores per-tensor `TensorDesc::dtype` in the directory) or quantising 1D RMSNorm weights to `Q4_0` (rejected due to severe normalization scale distortion for `<300 KB` savings).
- **Consequences:** Implemented and verified in `tools/convert_to_inference_engine.py`, `include/inference_engine/tensor.hpp`, `src/loader/loader.cpp`, `src/transformer/transformer.cpp`, and `tests/test_loader_validation.cpp`.

---

## ADR-009: Real TinyLlama Checkpoint Validation Protocol and Memory Architecture Constraints

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** Transitioning from synthetic validation to real `TinyLlama/TinyLlama-1.1B-Chat-v1.0` weights introduces network bandwidth constraints (~2.20 GB safetensors), storage considerations (8.29 GB for source and all target `.ie` formats), and strict zero-copy runtime contracts.
- **Decision:**
  1. **Two-Stage Validation Discipline:** Validate checkpoint metadata and tensor headers (23,088 bytes) and representative projection slices via HTTP Range and local file inspection without requiring a synchronous multi-gigabyte blocking download before verifying architecture and quantisation metrics.
  2. **Dedicated CLI Error Inspection Tool:** Provide `--eval-real-quant-error` in `tools/inspect_checkpoint.py` to evaluate and log real-checkpoint `Q4_0` quantisation metrics (`MAE`, `RMSE`, `MaxAbsErr`, `RelL2`) across all 9 projection functional roles.
  3. **Zero-Duplication Memory Architecture:**
     - `ModelLoader::load_from_file` uses POSIX `mmap` (`PROT_READ`, `MAP_PRIVATE`) so weight pages are mapped directly into the OS page cache with zero memory duplication.
     - `TensorView` instances are strictly non-owning views whose internal pointers are valid only during the lifetime of the backing `ModelLoader`. `ModelLoader` must strictly outlive `TransformerEngine`.
     - `TransformerEngine` pre-allocates an activation workspace (~393 KB total) during `init()`, and performs zero dynamic heap allocations in `forward_token()`.
- **Alternatives Considered:** Allocating owned heap buffers for all weights in C++ (rejected: doubles memory footprint and exceeds 8 GB RAM on commodity laptops) or ignoring real-weight quantisation metrics until full offline download completes (rejected: delays early discovery of numerical outliers).
- **Consequences:** Documented and verified in `tools/inspect_checkpoint.py`, `src/loader/loader.cpp`, and `src/transformer/transformer.cpp`.

---

## ADR-010: Dynamic Task Chunking and Exception Propagation in Persistent ThreadPool

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:**
  1. In multi-threaded matrix-vector multiplication (`dispatch_matmul_*`), row counts can reach up to 32,000 (such as in `lm_head.weight` projection). Individual task-by-task mutex locking resulted in massive contention on high-core CPUs (e.g. AMD Ryzen 7 7745HX, 8C/16T).
  2. If an exception occurred within a worker thread callback, standard `std::thread` workers would terminate the process abruptly (`std::terminate`), or leave worker threads in an indeterminate/deadlocked state.
- **Decision:**
  1. **Dynamic Task Chunking:** Compute dynamic chunk size as `std::max<size_t>(1, num_tasks / (num_threads * 4))`. This achieves optimal load balancing while eliminating over 95% of mutex acquisitions.
  2. **Exception Safety & Propagation:** Private member `std::exception_ptr exception_ptr_{}` captures the first thrown exception in any worker or caller thread. Remaining work is halted gracefully, all worker threads are synchronized back to the idle state via `cv_done_`, and the exception is cleanly rethrown on the calling thread via `std::rethrow_exception`.
- **Alternatives Considered:**
  - Work-stealing queues (unnecessary architectural complexity for predictable parallel-for loops).
  - OpenMP `#pragma omp parallel for` (rejected to maintain standard C++17 portability without compiler runtime dependencies).
- **Consequences:** Implemented in `include/inference_engine/thread_pool.hpp` and `src/kernels/threading/thread_pool.cpp`. Validated via unit tests in `tests/test_kernels_simd.cpp` covering 1,000 rapid successive cycles and forced exception recovery.

---

## ADR-011: Reproducible SentencePiece Build Dependency

- **Date:** 2026-10-31
- **Status:** Accepted
- **Context:** The tokenizer must use the upstream SentencePiece C++ implementation across supported build hosts without requiring a separately installed tokenizer library.
- **Decision:**
  - Prefer a system SentencePiece library only when pkg-config reports version `0.2.0` or newer.
  - Otherwise fetch the upstream repository at commit `17d7580d6407802f85855d2cc9190634e2c95624` and build its static C++ library with optional tests/shared libraries/tcmalloc disabled.
  - Keep the library out of the public tokenizer header through a forward-declared processor type.
- **Alternatives Considered:** A custom tokenizer, relying only on Homebrew/apt installation instructions, or an unpinned latest upstream checkout. Rejected because these reduce compatibility, reproducibility, or ease of building.
- **Consequences:** CI and developer builds can produce the same SentencePiece API implementation without committing tokenizer assets or vendor build outputs. Hugging Face tokenizer assets remain fetched only for verification and local inference.

---

## ADR-012: 2-Row Register Tiling, Vectorized Scale Accumulation, Contiguous 2-Row Dispatch, and Hardware Prefetcher Reliance across AVX2 and NEON

- **Date:** 2026-10-31 (Updated 2026-11-01)
- **Status:** Accepted (Implemented for AVX2 and NEON in `src/kernels/avx2/kernels_avx2.cpp`, `src/kernels/neon/kernels_neon.cpp`, and `src/kernels/dispatch.cpp`)
- **Context:**
  1. In single-token autoregressive decode (`batch_size = 1`), memory-bandwidth-bound matrix-vector operations ($y = W x$) dominate token generation latency. Profiling initial AVX2 and NEON Q4_0 kernels revealed three microarchitectural bottlenecks:
     - Naive per-block horizontal reductions in `dot_q4_0_f32` called `hsum256_ps` / `vaddvq_f32` 64 times per 2048-dim row, incurring ~14-16 cycles of shuffle/add latency per block (~1,000 stall cycles per row).
     - Row-by-row execution in `matmul_q4_0_f32` repeatedly streamed the activation vector $x$ from cache for each row ($M$ times), saturating L1/L2 load ports.
     - Single-row dispatch in `dispatch_matmul_q4_0_f32` pushed up to 32,000 atomic task increments into the `ThreadPool` for vocabulary projection ($32000 \times 2048$), causing severe thread queue contention.
     - Software prefetching (`_mm_prefetch`) was evaluated on AMD Ryzen 7 7745HX (`benchmarks/results/benchmarks_prefetch.csv`) and found to degrade throughput by ~10-23% (e.g. Attention Q dropped from 157.09 GFLOPS to 121.05 GFLOPS) because Zen 4's hardware stream prefetchers accurately predict linear row strides, while manual prefetches wasted front-end issue slots and polluted load queues.
- **Decision:**
  1. **Vectorized Per-Block Scale Accumulation (`compute_block_unscaled_dot_vec`):** Accumulate unscaled vector dot products within each `BlockQ4_0` using vector FMA (`_mm256_fmadd_ps` on AVX2, `vfmaq_f32` on NEON), deferring the horizontal reduction to a single execution at the end of the entire row.
  2. **2-Row Register Tiling (`matmul_q4_0_f32`, `matmul_f32`):** Process output rows in pairs (`m` and `m + 1`), loading each 32-element activation slice (`x0..x3`) once into registers and reusing it across both weight rows (12/16 YMM registers on AVX2, 16/32 registers on NEON, zero spilling), with a clean scalar/1-row tail path when `out_dim` is odd.
  3. **Contiguous 2-Row-Aligned Task Slicing & Adaptive Small-Matrix Cap (`src/kernels/dispatch.cpp`):** Slice `out_dim` into even-sized contiguous row blocks (`rows_per_task = std::max<size_t>(min_rows, (((out_dim + target_tasks - 1) / target_tasks) + 1) & ~size_t(1))`) before invoking `pool->parallel_for`, routing each worker slice directly into `matmul_q4_0_f32` / `matmul_f32`. For small projections (`out_dim <= 256`, such as Attention K/V), clamp task count to physical cores ($\le 8$) and enforce a minimum 16 rows per task, eliminating SMT queue contention and wake-up jitter on sub-microsecond workloads.
  4. **Omit Manual Software Prefetch:** Rely on hardware L1/L2 stream prefetchers for linear row-major scans.
- **Consequences:**
  Verified on AMD Ryzen 7 7745HX (`benchmarks/results/benchmarks_optimised.csv`): 8-thread `2048x2048` Attention Q latency dropped by **30.10%** (`0.0764 ms` -> `0.0534 ms`, `157.09 GFLOPS`, `44.18 GB/s`), and 16-thread `2048x5632` SwiGLU Down reached `162.80 GFLOPS` (`45.79 GB/s`, `23.57x` over scalar). NEON implementation in `src/kernels/neon/kernels_neon.cpp` ported and verified under ARM64 cross-compilation. All 8 CTest suites pass with zero regressions; maximum numerical discrepancy vs. scalar oracle is < 4.2e-5 absolute and < 3.5e-7 relative L2. Fully documented in `docs/benchmarks/CPU_KERNEL_BENCHMARKS.md`.

## ADR-013: Interactive Chat REPL, KV-Cache Prefix Matching & Top-K Sampling

- **Date:** 2026-11-01
- **Status:** Accepted (Implemented in `src/app/main.cpp`, `src/core/kv_cache.cpp`, and `src/sampler/sampler.cpp`)
- **Context:**
  1. For production readiness and competition evaluation (BIOS V2 PS08), the CLI must support interactive multi-turn conversations without requiring full prompt re-computation on every conversational turn.
  2. In multi-turn chat, previous user and assistant turns are retained in conversational history. Re-evaluating the entire conversation prefix from scratch ($O(N^2)$ prefill compute) induces unacceptable latency on CPU.
  3. The `KvCache` commit contract (`append` vs `commit_token`) strictly enforces that `committed_seq_len_` only advances after all layers complete. A safe mechanism was needed to allow sequence length rewinding (`truncate`) when conversational history is cleared or modified, while keeping committed tokens synchronized.
  4. Sampling required $top\text{-}k$ filtering to constrain the candidate pool before nucleus ($top\text{-}p$) filtering.
- **Decision:**
  1. **`KvCache::truncate(new_len)`:** Implemented safe rewinding in `KvCache` that validates `new_len <= committed_seq_len_` and sets `committed_seq_len_ = new_len`. Since all layers append at explicit position indices up to `committed_seq_len_`, rewinding preserves the contract without reallocating storage.
  2. **Prefix-Matching Multi-Turn KV Cache Reuse:** In `src/app/main.cpp`, token IDs committed to the KV cache are tracked in `committed_tokens`. On each turn, the full formatted chat history is tokenized, and the common prefix length `match_len` against `committed_tokens` is determined. If `match_len < committed_tokens.size()`, `truncate(match_len)` rewinds the cache. Prefill then executes exclusively for new tokens from `match_len` to `full_tokens.size() - 1`, achieving near-zero latency multi-turn prompts.
  3. **Interactive REPL UX:** Added `BIOS> ` interactive prompt with auto-discovery of `.ie` models and tokenizers, command parsing (`/help`, `/clear`, `/exit`, `/quit`), streaming output, and context-window monitoring against `max_seq_len` (2048).
  4. **Top-K Candidate Filtering:** Added `top_k` field to `SamplerConfig` and candidate truncation in `Sampler::sample()` prior to softmax and CDF nucleus accumulation.
- **Consequences:**
  Verified multi-turn conversation memory on hardware:
  Turn 1: "Hi, my name is Alice." -> Turn 2: "What is my name?" -> Answer: "Your name is Alice."
  Sub-turn prefill executes in < 30 ms. All 9 CTest suites and unit tests pass. Zero memory leaks or regressions.
