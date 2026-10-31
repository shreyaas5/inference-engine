# `AGENTS.md` — Permanent AI Coding Agent Instructions

This file is the mandatory entry point for any AI coding assistant (Antigravity, Claude Code, Codex, Cursor, ChatGPT, or others) working inside `~/Developer/inference-engine`.

---

## 1. Mandatory Startup Reading Order

Before modifying any code, configuration, or documentation, **every AI session MUST read these files in order**:

1. [`AGENTS.md`](AGENTS.md) — Permanent rules, constraints, conventions, and context update protocol (this file).
2. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — Single source of truth for current milestone, branch, build/test state, and active priorities.
3. [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) — Immediate session-to-session continuation instructions, working vs stubbed components, and the exact next task.
4. [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — Concise project identity, competition context (BIOS V2, PS08), hardware matrix, and team division.
5. Relevant technical and status documents in [`docs/context/`](docs/context/):
   - [`docs/context/TECHNICAL_CONTEXT.md`](docs/context/TECHNICAL_CONTEXT.md) — Frozen contracts (`.ie` v1 format, `Q4_0` block layout, `KvCache` commit semantics, RoPE/GQA, position indexing).
   - [`docs/context/IMPLEMENTATION_STATUS.md`](docs/context/IMPLEMENTATION_STATUS.md) — Module-by-module truth table distinguishing `IMPLEMENTED`/`TESTED` code from `IN PROGRESS` stubs.
   - [`docs/context/KNOWN_ISSUES.md`](docs/context/KNOWN_ISSUES.md) — Active bugs, stubs, missing dependencies, and unverified assumptions.
   - [`docs/context/NEXT_ACTIONS.md`](docs/context/NEXT_ACTIONS.md) — Dependency-ordered engineering backlog.
6. **Inspect actual repository code and Git state** (`git status`, `git branch`, `git log -n 5`) before making changes. Where documentation and code ever disagree, **the repository code is the source of truth for what is implemented**, and [`DECISIONS.md`](DECISIONS.md) + [`docs/context/TECHNICAL_CONTEXT.md`](docs/context/TECHNICAL_CONTEXT.md) are the source of truth for intended design.

---

## 2. Project Identity & Naming Rules (Strictly Enforced)

- **Project Name:** `inference-engine`
- **Competition Context:** BIOS V2 — Problem Statement **PS08 (Own Inference Engine)**
- **Local Directory:** `/Users/shreyaas/Developer/inference-engine` (`~/Developer/inference-engine`)
- **Remote Repository:** `https://github.com/shreyaas5/inference-engine` (Private)
- **C++ Namespace:** `inference_engine`
- **Primary Library Target:** `inference_engine_core`
- **CLI Executable Target:** `inference-engine`
- **Test Executable Target:** `inference_engine_tests`
- **Custom Model Extension:** `.ie` (Magic `0x464D4549` / `"IEMF"`, version `1`)

> **CRITICAL RULE:** Never use `BIOS` as the project name, product name, C++ namespace, folder name, binary name, or file prefix. `BIOS V2` is the competition name only.

---

## 3. Frozen Architecture & Correctness Constraints

1. **Primary Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (frozen hyperparameters: `22` layers, `hidden_dim = 2048`, `intermediate_dim = 5632`, `32` query heads, `4` KV heads, `head_dim = 64`, `vocab_size = 32000`, `max_seq_len = 2048`, RMSNorm $\epsilon = 10^{-5}$, RoPE $\theta = 10000.0$, SwiGLU MLP).
2. **No External Inference Runtimes:** All neural network inference must execute in custom C++17. Never link or invoke PyTorch, LibTorch, TensorFlow, ONNX Runtime, `llama.cpp`, GGML, or MLX at runtime. Python (`tools/`) is strictly restricted to offline model inspection/conversion, tokenizer parity verification, and generating golden reference fixtures.
3. **CPU-Only Official Execution:** Official inference and benchmarking target commodity laptop CPUs only (ARM64 NEON and x86-64 AVX2/FMA with runtime CPU feature detection). No GPU/NPU execution.
4. **Scalar Reference First:** `src/kernels/scalar/` is the numerical correctness reference. SIMD kernels (`src/kernels/neon/`, `src/kernels/avx2/`) must be verified against scalar kernels and must never bypass runtime availability checks (`is_backend_available()`).
5. **`.ie` Binary Format Contract:** Little-endian field-by-field serialization (never serialize raw C++ structs), 64-byte header, variable-length tensor directory, 64-byte aligned tensor payloads (`payload_offset % 64 == 0`), checked 64-bit integer arithmetic (`checked_add_u64`, `checked_mul_u64`), exact byte-size validation, and non-overlap validation.
6. **`Q4_0` Block Contract:** `32` weights per block, `18` bytes per block (`2`-byte IEEE 754 FP16 scale `d` + `16` packed bytes `qs[0..15]`). Low nibble (`qs[j] & 0x0F`) stores indices `0..15`; high nibble (`(qs[j] >> 4) & 0x0F`) stores indices `16..31`. Dequantisation is `(q - 8) * fp16_to_fp32(d)`. For multi-dimensional `Q4_0` tensors, the innermost dimension `shape[ndim - 1]` must be divisible by `32`.
7. **`KvCache` Commit Contract:** `KvCache::append(layer_idx, pos, k_data, v_data)` writes K/V vectors for one layer and position and **must NEVER increment `seq_len()`**. `KvCache::commit_token()` (or `commit_positions(count)`) advances `seq_len()` **once per completed token** after all 22 layers have processed that position.
8. **Autoregressive Position Indexing Contract:** A prompt of $N$ tokens (e.g., $N = 12$) is processed at positions `0 .. N-1` (`0..11`). The logits produced at the final prompt position `N-1` (`11`) predict the first generated token. That first generated token is subsequently fed into `forward_token` at position `N` (`12`), followed by `13`, `14`, etc.
9. **GQA Mapping:** `32` query heads share `4` KV heads (`8` query heads per KV head: query head `h_q` maps to KV head `h_q / 8`).
10. **Golden Validation Discipline:** Hugging Face `TinyLlama-1.1B-Chat-v1.0` checkpoints are stored in `BF16`; they must be explicitly cast to `FP32` before generating reference activations/logits and exporting the `.ie` `FP32` model so both engines use bit-identical `FP32` weights.

---

## 4. C++17 Coding Conventions

- **Standard:** C++17 (`-std=c++17`), no compiler-specific language extensions (`CMAKE_CXX_EXTENSIONS OFF`).
- **Warnings:** Code must compile with zero warnings under `-Wall -Wextra -Wpedantic` (Clang/GCC) and `/W4` (MSVC).
- **Error Handling:** Return `inference_engine::Status` (`StatusCode` + message) across subsystem boundaries; avoid throwing exceptions in hot inference or loader paths.
- **Memory Safety:** Use checked 64-bit arithmetic helpers (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`) in [`include/inference_engine/common.hpp`](include/inference_engine/common.hpp) whenever computing buffer sizes or file offsets from external inputs.
- **SIMD Isolation:** Keep architecture-specific intrinsics isolated to `src/kernels/neon/kernels_neon.cpp` (`#if defined(__aarch64__) || defined(_M_ARM64)`) and `src/kernels/avx2/kernels_avx2.cpp` (`#if defined(__AVX2__)`), with scalar fallbacks so all translation units compile on every OS/CPU.

---

## 5. Verified Build & Test Commands

Run these from `/Users/shreyaas/Developer/inference-engine`:

```bash
# 1. Configure CMake (Release build with tests enabled)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON

# 2. Build core static library, CLI binary, and unit test runner
cmake --build build --parallel 4

# 3. Run all 7 CTest unit test suites
ctest --test-dir build --output-on-failure

# 4. Run a single CTest suite directly
./build/inference_engine_tests --suite loader_validation

# 5. Smoke-check the CLI executable
./build/inference-engine --help
./build/inference-engine --info

# 6. Run offline Python tool self-tests (no external checkpoint required)
python3 tools/inspect_checkpoint.py --print-frozen-contract
python3 tools/convert_to_inference_engine.py --self-test
python3 tools/verify_tokenizer_parity.py --self-test
```

> **Note:** Clean up temporary `build/` directories (`rm -rf build`) if needed, or leave `build/` untracked (it is ignored in [`.gitignore`](.gitignore)). Never commit build artifacts.

---

## 6. Git Branch & Team Ownership Workflow

- **Branches:**
  - `main` — Protected checkpoint branch. **Never commit directly to `main`.**
  - `integration` — Shared integration branch for cross-team PRs.
  - `feature/model-loader` — **S** (`src/loader/`, `src/common/`, `tools/inspect_checkpoint.py`, `tools/convert_to_inference_engine.py`, `tests/test_loader_validation.cpp`)
  - `feature/quantisation-simd` — **I** (`src/kernels/quant_q4_0.cpp`, `src/kernels/avx2/`, `src/kernels/neon/`, `src/kernels/threading/`, `src/kernels/dispatch.cpp`, `tests/test_quant_packing.cpp`, `tests/test_kernels_simd.cpp`)
  - `feature/transformer` — **U** (`src/kernels/scalar/`, `src/core/kv_cache.cpp`, `src/transformer/`, `tools/generate_golden_reference.py`, `tests/test_kernels_scalar.cpp`, `tests/test_kv_cache.cpp`, `tests/test_golden_forward.cpp`)
  - `feature/runtime-tokenizer` — **P** (`src/tokenizer/`, `src/sampler/`, `src/app/`, `tools/verify_tokenizer_parity.py`, `tests/test_tokenizer.cpp`, `.github/workflows/ci.yml`)
- **Commit Style:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`, `chore:`).
- **Forbidden Files in Git:** Never commit `*.ie`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.gguf`, `*.model`, `*.npy`, or `*.npz`.

---

## 7. What Must Never Be Assumed or Claimed

1. **Never claim a module is complete merely because its `.cpp` file exists or compiles.** Check [`docs/context/IMPLEMENTATION_STATUS.md`](docs/context/IMPLEMENTATION_STATUS.md) and read the source file.
2. **Never claim end-to-end TinyLlama inference works** until a real converted `.ie` checkpoint is loaded and `TransformerEngine::forward_token()` executes all 22 layers against real weights (currently `forward_token()` in [`src/transformer/transformer.cpp`](src/transformer/transformer.cpp) is an M0 stub that writes zero KV vectors and zero logits).
3. **Never claim SentencePiece tokenization is active** until SentencePiece C++ is linked in `CMakeLists.txt` and `Tokenizer::encode()` in [`src/tokenizer/tokenizer.cpp`](src/tokenizer/tokenizer.cpp) replaces the M0 byte-offset fallback.
4. **Never claim NEON/AVX2 `Q4_0` acceleration is active** until `neon::dot_q4_0_f32` and `avx2::dot_q4_0_f32` replace their current delegation to `scalar::dot_q4_0_f32`.
5. **Never fabricate or estimate benchmark numbers, tolerances, or collaborator GitHub usernames.**

---

## 8. Mandatory End-of-Session Context Update Protocol

Before ending any session that modifies code, tests, build configuration, or design decisions, you **MUST** update:

1. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — Reflect current branch, commit, test counts, completed work, and top 3 next priorities.
2. [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) — Record what changed, what works, what remains broken/stubbed, and the exact next command/task.
3. [`docs/context/IMPLEMENTATION_STATUS.md`](docs/context/IMPLEMENTATION_STATUS.md) — Update module statuses (`NOT STARTED`, `IN PROGRESS`, `IMPLEMENTED`, `TESTED`, `REFERENCE VERIFIED`) with evidence.
4. [`docs/context/KNOWN_ISSUES.md`](docs/context/KNOWN_ISSUES.md) — Add new issues or mark resolved ones with verification evidence.
5. [`docs/context/NEXT_ACTIONS.md`](docs/context/NEXT_ACTIONS.md) — Check off completed items and reprioritize.
6. [`docs/context/SESSION_LOG.md`](docs/context/SESSION_LOG.md) — Append a dated session entry summarizing objective, files touched, tests run, and remaining work.
7. [`DECISIONS.md`](DECISIONS.md) — Record any new or updated architectural decision (`ADR-XXX`).

