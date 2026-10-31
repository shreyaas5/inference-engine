# `01_S_INTEGRATION_AND_MODEL_SYSTEM.md` — Member S Handoff: Architecture, Model System, Integration & Team Coordination

> **Member:** **S** (Team Lead, System Architect, `.ie` Model Format & Loader Owner, Integration Coordinator)
> **Consolidated Baseline Branch:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Historical Feature Branches (All Merged into `33d4bdc`):**
> - `origin/feature/model-loader` (`aa403689e7ef08f6fb8c1877729a346a5f70595b`)
> - `origin/feature/model-format-q4` (`cb176b3ec16f0025ad0af2c4e6074fa95804ac52` — commits `cd98ae8`, `b13baf8`, `6dd4fed`, `cb176b3`)
> - `origin/feature/real-checkpoint-validation` (`4515579672c325812bf42670f63a17b2967f3ef4` — merged via `0afcafe`)

---

## A. Role and Ownership

### 1. Primary Responsibilities
- **`.ie` v1 Model Binary Format Specification (`VERIFIED`):** Maintain the frozen 64-byte header, variable-length tensor directory, and 64-byte-aligned payload contract ([`docs/FORMAT_SPECIFICATION.md`](../FORMAT_SPECIFICATION.md)).
- **C++17 Zero-Copy Memory-Mapped Loader & Weight Binding (`VERIFIED`):** Own `MmapModelLoader`, `TensorView`, `LayerWeightBindings`, `TransformerWeightBindings`, and `bind_tinyllama_weights()`.
- **Offline Python Checkpoint Inspection & Conversion Tooling (`VERIFIED`):** Own `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` (`FP32`, pure `Q4_0`, and mixed-precision `--keep-embeddings-fp32`).
- **Cross-Team Integration & Branch Governance (`VERIFIED`):** Audit, merge, and verify feature branches from **U**, **I**, and **P** onto the integration branch before promoting milestones to `integration` and `main`.

### 2. Files Owned and Modified by S
- **Owned Public Headers:**
  - [`include/inference_engine/common.hpp`](../../include/inference_engine/common.hpp) — `Status`, `StatusCode`, `DataType`, `SimdBackend`, checked 64-bit arithmetic (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`).
  - [`include/inference_engine/config.hpp`](../../include/inference_engine/config.hpp) — `TinyLlamaConfig`, `ModelFileHeader`, magic `kIeMagic = 0x464D4549U` (`"IEMF"`), `kIeFormatVersion = 1U`, `kIeTensorAlignment = 64U`.
  - [`include/inference_engine/tensor.hpp`](../../include/inference_engine/tensor.hpp) — Non-owning `TensorView` (`data_f32()`, `data_q4_0()`, `row_f32()`).
  - [`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp) — `MmapModelLoader`, `LayerWeightBindings`, `TransformerWeightBindings`, `bind_tinyllama_weights()`.
- **Owned C++ Implementation & Tests:**
  - [`src/common/status.cpp`](../../src/common/status.cpp) — `status_code_to_string()`, `dtype_to_string()`, `dtype_element_size()`, `backend_to_string()`.
  - [`src/loader/loader.cpp`](../../src/loader/loader.cpp) — POSIX `mmap`/`munmap` and Windows `CreateFileMappingA`/`MapViewOfFile`, little-endian byte readers, directory parser, non-overlap/alignment validators, and `bind_tinyllama_weights()`.
  - [`tests/test_loader_validation.cpp`](../../tests/test_loader_validation.cpp) — 6 comprehensive CTest validation cases.
- **Owned Python Tools:**
  - [`tools/inspect_checkpoint.py`](../../tools/inspect_checkpoint.py) — 201-tensor contract inspector, `.ie` header/directory auditor, and `--eval-real-quant-error` evaluator.
  - [`tools/convert_to_inference_engine.py`](../../tools/convert_to_inference_engine.py) — Streaming `safetensors` $\rightarrow$ `.ie` v1 exporter (`--dtype {fp32,q4_0}`, `--keep-embeddings-fp32`, `--verify`, `--self-test`).
- **Co-Owned Build & Context Files:**
  - [`CMakeLists.txt`](../../CMakeLists.txt), [`DECISIONS.md`](../../DECISIONS.md), [`CURRENT_STATUS.md`](../../CURRENT_STATUS.md), [`SESSION_HANDOFF.md`](../../SESSION_HANDOFF.md), `docs/context/*`, `docs/team_handoffs/*`.

### 3. Ownership Boundaries (What S Must NOT Independently Modify)
- Do **not** alter `TransformerEngine::forward_token` math in `src/transformer/transformer.cpp` or scalar reference kernels in `src/kernels/scalar/` without coordinating with **U**.
- Do **not** alter SIMD kernel intrinsics (`src/kernels/avx2/`, `src/kernels/neon/`) or `ThreadPool` internals (`src/kernels/threading/`) without coordinating with **I** (except when resolving merge conflicts during branch integration).
- Do **not** alter SentencePiece tokenisation (`src/tokenizer/`) or sampling algorithms (`src/sampler/`) without coordinating with **P**.

---

## B. Technical Background & Exact Implementation Details

### 1. Exact `.ie` v1 Binary Header Layout (`64` Bytes, Little-Endian) (`VERIFIED`)
Both [`src/loader/loader.cpp`](../../src/loader/loader.cpp) (`parse_and_validate()`) and [`tools/convert_to_inference_engine.py`](../../tools/convert_to_inference_engine.py) (`IE_HEADER_STRUCT = "<12I2fQ"`) serialize and parse the exact 64-byte header below:

| Byte Offset | Type | Field Name | Expected TinyLlama-1.1B Value | Validation Rule in `MmapModelLoader` |
| :--- | :--- | :--- | :--- | :--- |
| `0x00 .. 0x03` | `u32` | `magic` | `0x464D4549` (`"IEMF"`) | Must equal `kIeMagic` (`UnsupportedFormat` otherwise) |
| `0x04 .. 0x07` | `u32` | `version` | `1` | Must equal `kIeFormatVersion` (`UnsupportedFormat` otherwise) |
| `0x08 .. 0x0B` | `u32` | `tensor_count` | `201` (22 layers) or `9*L + 3` | Must be `> 0` and `<= 100000` |
| `0x0C .. 0x0F` | `u32` | `weight_dtype` | `0` (`FP32`) or `2` (`Q4_0`) | Must be a valid `DataType` enum (`0..3`) |
| `0x10 .. 0x13` | `u32` | `num_layers` | `22` | Checked against config (`> 0`) |
| `0x14 .. 0x17` | `u32` | `hidden_dim` | `2048` | Must equal `num_query_heads * head_dim` |
| `0x18 .. 0x1B` | `u32` | `intermediate_dim` | `5632` | Must be `> 0` |
| `0x1C .. 0x1F` | `u32` | `num_query_heads` | `32` | Must be divisible by `num_kv_heads` |
| `0x20 .. 0x23` | `u32` | `num_kv_heads` | `4` | Must be `> 0` |
| `0x24 .. 0x27` | `u32` | `head_dim` | `64` | Must be `> 0` and even |
| `0x28 .. 0x2B` | `u32` | `vocab_size` | `32000` | Must be `> 0` |
| `0x2C .. 0x2F` | `u32` | `max_seq_len` | `2048` | Must be `> 0` |
| `0x30 .. 0x33` | `f32` | `rms_norm_eps` | `1e-5f` (`0x3727C5AC`) | IEEE 754 `float` |
| `0x34 .. 0x37` | `f32` | `rope_theta` | `10000.0f` (`0x461C4000`) | IEEE 754 `float` |
| `0x38 .. 0x3F` | `u64` | `total_file_bytes` | Exact file size in bytes | Must match OS file byte size (`st_size` / `GetFileSizeEx`) |

### 2. Tensor Directory & 64-Byte Payload Alignment (`VERIFIED`)
Immediately following the header at byte `0x40` (`64`), `tensor_count` directory entries are stored back-to-back:
- `name_len` (`u32`, `1 .. 512`), `name` (`name_len` UTF-8 bytes, no null terminator), `dtype` (`u32`), `ndim` (`u32`, `1 .. 4`), `shape[0 .. ndim-1]` (`ndim * u64`), `payload_offset` (`u64`), `payload_bytes` (`u64`).
- **Validation invariants enforced in `MmapModelLoader::parse_and_validate()` (`src/loader/loader.cpp`):**
  1. Duplicate tensor names are rejected (`CorruptedFile`).
  2. Every dimension `shape[d] > 0`, and total elements are computed with `checked_mul_u64` (`CorruptedFile` on 64-bit overflow).
  3. For `DataType::Q4_0`, `shape[ndim - 1] % 32 == 0` is mandatory, and `expected_bytes = (num_elements / 32) * 18`. For `FP32`/`FP16`/`I32`, `expected_bytes = num_elements * dtype_element_size(dtype)`. `payload_bytes` must equal `expected_bytes`.
  4. `payload_offset >= cursor` (cannot overlap header/directory), `payload_offset % 64 == 0` (64-byte alignment), and `payload_offset + payload_bytes <= mapped_size_` (`OutOfBounds` otherwise).
  5. All `[payload_offset, payload_offset + payload_bytes)` intervals are sorted by start offset and checked to ensure zero overlap between adjacent tensors (`CorruptedFile` on overlap).

### 3. `TensorView` and `bind_tinyllama_weights()` (`VERIFIED`)
- [`include/inference_engine/tensor.hpp`](../../include/inference_engine/tensor.hpp):
  - `TensorView::data_f32()` returns `reinterpret_cast<const float*>(data_)` when `dtype_ == DataType::FP32`, else `nullptr`.
  - `TensorView::data_q4_0()` returns `reinterpret_cast<const BlockQ4_0*>(data_)` when `dtype_ == DataType::Q4_0`, else `nullptr`.
- [`src/loader/loader.cpp`](../../src/loader/loader.cpp) (`bind_tinyllama_weights()`):
  - Validates all `9 * num_layers + 3` tensors by exact Hugging Face name and shape.
  - Enforces that 1D RMSNorm tensors (`input_layernorm`, `post_attention_layernorm`, `model.norm.weight`) are strictly `DataType::FP32`.
  - Enforces that 2D projection tensors match `loader.header().weight_dtype` (`FP32` or `Q4_0`), while allowing `model.embed_tokens.weight` and `lm_head.weight` to be either `weight_dtype` or `DataType::FP32` (supporting mixed-precision `--keep-embeddings-fp32` models without a separate header version).

---

## C. Completed Work, Commits, and Verification Evidence

| Deliverable | Commit(s) | Classification | Evidence |
| :--- | :--- | :--- | :--- |
| `.ie` v1 format specification, `MmapModelLoader`, overflow/overlap/alignment validation, `FP32` converter | `aa40368` (`origin/feature/model-loader`) | `VERIFIED` / `TESTED` | `loader_validation` CTest (`6/6` passed) |
| `Q4_0` converter (`--dtype q4_0`, `--keep-embeddings-fp32`), `TensorView::data_q4_0()`, unified `TransformerWeightBindings` | `cd98ae8`, `b13baf8`, `6dd4fed`, `cb176b3` (`origin/feature/model-format-q4`) | `VERIFIED` / `TESTED` | `python3 tools/convert_to_inference_engine.py --self-test` (`FP32`, `Q4_0`, mixed `Q4_0`) |
| Live 201-tensor inspection & real `Q4_0` quantisation error evaluation across all 9 functional tensor roles | `4515579` (`origin/feature/real-checkpoint-validation`) | `REAL-CHECKPOINT VALIDATED` | `python3 tools/inspect_checkpoint.py --checkpoint-dir models/TinyLlama-1.1B-Chat-v1.0 --eval-real-quant-error` |
| 4-member branch consolidation (`S` + `U` + `I` + `P`), CMake 4.x `sentencepiece-static` fix, CI macOS runner & `protobuf` fix | `0afcafe`, `a687095`, `991a527`, `3b82184`, `5929a17`, `33d4bdc` (`integration-ps08-team-sync-2026-10-31`) | `VERIFIED` / `TESTED` | `9/9` CTest suites passing; synthetic 1-layer CLI E2E + `--benchmark` passing |

### Real-Checkpoint `Q4_0` Error Profile (`REAL-CHECKPOINT VALIDATED` on `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors`)
Evaluated on `16,384` real `BF16` $\rightarrow$ `FP32` weights (`512` `Q4_0` blocks) per functional role from the local `model.safetensors` prefix (`325.47 MB` covering `model.embed_tokens.weight` and Layers `0..2`):

| Functional Role | Representative Tensor | Shape | Target `.ie` Dtype | MaxAbsErr | RMSE | Mean Cosine Sim | SQNR (dB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `embed_tokens` | `model.embed_tokens.weight` | `[32000, 2048]` | `Q4_0` (or `FP32`) | `1.3916e-02` | `2.6562e-03` | `0.989677` | `16.82 dB` |
| `input_layernorm` | `model.layers.0.input_layernorm.weight` | `[2048]` | `FP32` (unquantised) | `0.0` | `0.0` | `1.000000` | $\infty$ |
| `attn_q_proj` | `model.layers.0.self_attn.q_proj.weight` | `[2048, 2048]` | `Q4_0` | `2.2949e-02` | `2.7073e-03` | `0.988884` | `16.51 dB` |
| `attn_k_proj` | `model.layers.0.self_attn.k_proj.weight` | `[256, 2048]` | `Q4_0` | `3.2715e-02` | `2.7534e-03` | `0.989166` | `16.65 dB` |
| `attn_v_proj` | `model.layers.0.self_attn.v_proj.weight` | `[256, 2048]` | `Q4_0` | `8.5449e-03` | `1.4307e-03` | `0.989643` | `16.79 dB` |
| `attn_o_proj` | `model.layers.0.self_attn.o_proj.weight` | `[2048, 2048]` | `Q4_0` | `1.4526e-02` | `1.5944e-03` | `0.989122` | `16.58 dB` |
| `post_attn_layernorm` | `model.layers.0.post_attention_layernorm.weight` | `[2048]` | `FP32` (unquantised) | `0.0` | `0.0` | `1.000000` | $\infty$ |
| `mlp_gate_proj` | `model.layers.0.mlp.gate_proj.weight` | `[5632, 2048]` | `Q4_0` | `1.3794e-02` | `2.5877e-03` | `0.989659` | `16.80 dB` |
| `mlp_up_proj` | `model.layers.0.mlp.up_proj.weight` | `[5632, 2048]` | `Q4_0` | `1.6968e-02` | `2.3714e-03` | `0.989529` | `16.75 dB` |
| `mlp_down_proj` | `model.layers.0.mlp.down_proj.weight` | `[2048, 5632]` | `Q4_0` | `1.7090e-02` | `2.3712e-03` | `0.989446` | `16.72 dB` |

---

## D. Current State: What Works vs. What Remains Pending

- **What Works Now (`VERIFIED` / `TESTED` / `REAL-CHECKPOINT VALIDATED`):**
  - `MmapModelLoader` loads and validates `.ie` v1 files in `FP32`, pure `Q4_0`, and mixed `Q4_0` (`--keep-embeddings-fp32`) modes with zero warnings and `6/6` passing unit tests in `loader_validation`.
  - `tools/inspect_checkpoint.py` and `tools/convert_to_inference_engine.py` are fully implemented and verified on synthetic checkpoints and the real 201-tensor `model.safetensors` header.
- **What Remains Unmerged or Incomplete:**
  1. **Two New Remote Teammate Commits Await Integration (`VERIFIED`):**
     - Member I's `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1` — AVX2 2-row register tiling, vectorized scale accumulation, contiguous 2-row `dispatch.cpp` chunk scheduling, `inference_engine_bench`, and Ryzen 7 7745HX benchmark CSVs).
     - Member U's `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d` — GCC 16 `-include cstdint` fix for `sentencepiece-static` in `CMakeLists.txt`).
  2. **Local `model.safetensors` Download Is Truncated (`BLOCKED` — `KI-004`):**
     - `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` on disk is currently `325,472,256` bytes (`310.4 MB`) out of the required `2,200,120,064` bytes (`2.20 GB`). Full 22-layer `.ie` conversion cannot finish until the remaining `1.87 GB` is downloaded.
  3. **`integration` (`783b8f4`) and `main` (`177fe49`) Not Yet Fast-Forwarded (`VERIFIED`):**
     - Consolidated work lives on `integration-ps08-team-sync-2026-10-31` (`33d4bdc`).

---

## E. Dependencies on Other Team Members

- **On Member I (`origin/feature/cpu-kernel-optimisation` at `b05c4e4`):** Note that `b05c4e4` branched from `5929a17` (before `33d4bdc`) and modified shared root docs (`CURRENT_STATUS.md`, `SESSION_HANDOFF.md`) and `tests/test_tokenizer.cpp`. When merging `b05c4e4`, preserve our consolidated `tests/test_tokenizer.cpp` and reconciled documentation while taking I's kernel, dispatch, benchmark, and `ADR-012` additions.
- **On Member U (`origin/feature/transformer-runtime` at `e64078c`):** Cherry-pick or merge the `CMakeLists.txt` `target_compile_options(sentencepiece-static PRIVATE -include cstdint)` fix so GCC 16 Linux builds succeed out of the box.
- **Downstream Consumers (U, I, P):** All three teammates depend on S producing the converted 22-layer `.ie` files (`models/tinyllama-fp32.ie`, `models/tinyllama-q4_0.ie`, `models/tinyllama-q4_0-mixed.ie`) once `model.safetensors` finishes downloading.

---

## F. Remaining Tasks (Priority Order)

### `P0` — Blocks the Entire Project
1. **Integrate `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) and `origin/feature/transformer-runtime` (`e64078c`) into the Consolidated Integration Branch:**
   - Merge/cherry-pick U's `CMakeLists.txt` GCC 16 `<cstdint>` fix (`e64078c`) and I's AVX2 2-row tiling + `dispatch.cpp` chunking + `inference_engine_bench` (`b05c4e4`), keeping `tests/test_tokenizer.cpp` and `.github/workflows/ci.yml` intact from `33d4bdc`.
   - Rebuild and verify all `9/9` CTest suites and `inference_engine_bench`.
2. **Complete Download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2,200,120,064` bytes) and Export Real `.ie` Models:**
   - Resume download with `curl -L -C -` or `huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 model.safetensors --local-dir models/TinyLlama-1.1B-Chat-v1.0`.
   - Verify full checkpoint integrity:
     ```bash
     python3 tools/inspect_checkpoint.py --checkpoint-dir models/TinyLlama-1.1B-Chat-v1.0
     ```
   - Export all three `.ie` variants and verify exact byte sizes:
     ```bash
     python3 tools/convert_to_inference_engine.py --input models/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama-fp32.ie --dtype fp32 --verify
     python3 tools/convert_to_inference_engine.py --input models/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama-q4_0.ie --dtype q4_0 --verify
     python3 tools/convert_to_inference_engine.py --input models/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama-q4_0-mixed.ie --dtype q4_0 --keep-embeddings-fp32 --verify
     ```
   - Expected exact sizes: `4,400,213,312` B (`FP32`), `619,113,792` B (pure `Q4_0`), `1,069,673,792` B (mixed `Q4_0`).

### `P1` — Required for the Next Major Milestone
3. **Coordinate Full 22-Layer Golden Forward Parity & CLI Smoke Tests with U and P:**
   - Once `.ie` files exist on disk, run `./build/inference-engine --model models/tinyllama-q4_0.ie --tokenizer models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model --prompt "Hello" --max-new-tokens 32 --benchmark`.
4. **Promote Consolidated Integration Branch to `integration` (and `main` after team sign-off).**

### `P2` — Important Improvements
5. **Make `benchmarks/bench_kernels.cpp` Backend-Agnostic on ARM64 NEON (Coordinate with I):**
   - Currently `bench_kernels.cpp` in `b05c4e4` hardcodes `avx2::matmul_q4_0_f32` and `SimdBackend::Avx2`; updating it to use `detect_best_backend()` will allow apple-to-apple kernel microbenchmarks on Apple Silicon NEON as well.

### `P3` — Optional Optimisation or Polish
6. **Optional OS Page-Cache Prefault Hint (`posix_madvise(..., POSIX_MADV_WILLNEED)`):**
   - Add an optional `advise_sequential` or `prefault` flag to `MmapModelLoader::open()` to minimize first-token page-fault latency during cold-start benchmarks.

---

## G. New-Session Continuation Prompt (Ready to Paste into Antigravity)

```markdown
You are operating as **Member S (Architecture, `.ie` Model System, Conversion Tooling & Integration Lead)** on the **BIOS V2 — PS08 (`inference-engine`)** project in repository `shreyaas5/inference-engine`.

Follow this exact 10-step autonomous workflow without asking the team lead for background already documented in the repository:

1. **Read the Master Project Context:** Read `AGENTS.md` and `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`.
2. **Read Your Complete Handoff & Shared Interfaces:** Read `docs/team_handoffs/01_S_INTEGRATION_AND_MODEL_SYSTEM.md`, `docs/team_handoffs/05_SHARED_INTERFACES_AND_DEPENDENCIES.md`, and `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`.
3. **Inspect the Current Git State:** Run `git fetch --all --prune`, `git status`, `git branch -a -vv`, and `git log --all --oneline --graph -n 25`.
4. **Identify the Latest Relevant Commits:** Verify the consolidated integration baseline (`integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff`), Member I's unmerged branch `origin/feature/cpu-kernel-optimisation` (`b05c4e49bf135342e4f94a43a5e4c45763c802e1`), and Member U's unmerged branch `origin/feature/transformer-runtime` (`e64078c84e7832deb0444f51ffad40c45154438d`).
5. **Inspect the Actual Source Files:** Read `include/inference_engine/{common,config,tensor,loader}.hpp`, `src/loader/loader.cpp`, `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`, `tests/test_loader_validation.cpp`, and `CMakeLists.txt`. Also check the byte size of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`ls -lh models/TinyLlama-1.1B-Chat-v1.0/`).
6. **Compare Implementation Against Documented State:** Verify that `MmapModelLoader`, `bind_tinyllama_weights()`, and the Python conversion/inspection tools match the documented contracts.
7. **Avoid Reimplementing Completed Functionality:** `.ie` v1 loading, 64-byte alignment/non-overlap validation, `FP32`/`Q4_0`/mixed-precision conversion, and `bind_tinyllama_weights()` are already implemented and tested (`loader_validation` 6/6 passing). Do not rewrite them.
8. **Identify the Highest-Priority Unfinished Tasks (`P0`):**
   - Integrate `origin/feature/transformer-runtime` (`e64078c`) and `origin/feature/cpu-kernel-optimisation` (`b05c4e4`) into the integration branch (preserving `tests/test_tokenizer.cpp` and reconciled docs from `33d4bdc`), and verify all 9 CTest suites + `inference_engine_bench` build cleanly.
   - Check/resume the download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2,200,120,064` bytes), run `python3 tools/inspect_checkpoint.py --checkpoint-dir models/TinyLlama-1.1B-Chat-v1.0`, and convert the `FP32`, pure `Q4_0`, and mixed `Q4_0` `.ie` models using `tools/convert_to_inference_engine.py`.
9. **Produce an Initial Repository Audit and Action Plan:** Output a concise audit confirming branch SHAs, test results, and `model.safetensors` byte size, followed by your step-by-step execution plan.
10. **Begin Implementation Immediately:** Execute the highest-priority `P0` tasks autonomously within your ownership boundaries. Only pause to ask the team lead if an unresolved architectural change affects shared interfaces, a required external resource is unreachable, or a destructive/protected-branch action (`main`) requires explicit sign-off. Before ending your session, update the context files required by `AGENTS.md` Section 8.
```
