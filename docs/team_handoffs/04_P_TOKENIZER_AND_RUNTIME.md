# `04_P_TOKENIZER_AND_RUNTIME.md` — Member P Handoff: Tokenizer, Sampling, Generation Runtime, Benchmarking & CI

> **Member:** **P** (SentencePiece Tokenizer, Zephyr Chat Template, Sampler, CLI Application, End-to-End Benchmark Harness, and GitHub Actions CI Owner)
> **Consolidated Baseline Branch:** `integration-ps08-team-sync-2026-10-31` (`33d4bdc4486e83f458b6a242d9a2fd051503e6ff`)
> **Historical Feature Branch (Merged into `33d4bdc` via `3b82184`):**
> - `origin/feature/runtime-tokenizer` (`ce769c882f0883613183812738340195a1f7d71e`)

---

## A. Role and Ownership

### 1. Primary Responsibilities
- **C++17 SentencePiece Tokenizer & Zephyr Chat Template (`VERIFIED` / `REAL-CHECKPOINT VALIDATED`):** Own `Tokenizer` (`include/inference_engine/tokenizer.hpp`, `src/tokenizer/tokenizer.cpp`) backed by Google `sentencepiece-static` (`v0.2.0`), special-token splitting (`<unk>`, `<s>`, `</s>`), dummy-prefix whitespace handling, and TinyLlama's Zephyr chat template (`apply_chat_template`).
- **Offline Tokenizer Parity Harness (`REAL-CHECKPOINT VALIDATED`):** Own `tools/verify_tokenizer_parity.py` verifying 100% bit-exact token ID and decoded string parity between C++ `inference-engine` and Python `sentencepiece` across 15 test cases using the real `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`.
- **Token Sampler (`VERIFIED` / `TESTED`):** Own `Sampler` (`include/inference_engine/sampler.hpp`, `src/sampler/sampler.cpp`) implementing greedy argmax, temperature scaling, top-p (nucleus) filtering, `NaN`/`Inf` sanitization, and deterministic 64-bit `xorshift*` pseudo-random sampling.
- **CLI Runtime & End-to-End Benchmark Harness (`VERIFIED` / `SYNTHETICALLY VALIDATED`):** Own `src/app/main.cpp` implementing `--info`, `--tokenize`, `--decode`, autoregressive prefill/decode generation, and `--benchmark` (TTFT, prefill tok/s, decode tok/s, wall time, CPU time, CPU utilization %, and Peak RSS MB).
- **GitHub Actions Multi-OS CI Pipeline (`VERIFIED`):** Own `.github/workflows/ci.yml` (`ubuntu-22.04` x86-64 AVX2 + `macos-latest` ARM64 NEON).

### 2. Files Owned and Modified by P
- **Owned Public Headers:**
  - [`include/inference_engine/tokenizer.hpp`](../../include/inference_engine/tokenizer.hpp)
  - [`include/inference_engine/sampler.hpp`](../../include/inference_engine/sampler.hpp)
- **Owned C++ Implementation:**
  - [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp)
  - [`src/sampler/sampler.cpp`](../../src/sampler/sampler.cpp)
  - [`src/app/main.cpp`](../../src/app/main.cpp)
- **Owned Tests, Tooling & CI:**
  - [`tests/test_tokenizer.cpp`](../../tests/test_tokenizer.cpp)
  - [`tests/test_sampler.cpp`](../../tests/test_sampler.cpp)
  - [`tools/verify_tokenizer_parity.py`](../../tools/verify_tokenizer_parity.py)
  - [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

### 3. Ownership Boundaries (What P Must NOT Independently Modify)
- Do **not** modify `.ie` binary format parsing or `MmapModelLoader` (`src/loader/`) without coordinating with **S**.
- Do **not** modify `TransformerEngine::forward_token` (`src/transformer/`), `KvCache` (`src/core/`), or scalar kernels (`src/kernels/scalar/`) without coordinating with **U**.
- Do **not** modify SIMD kernels (`src/kernels/avx2/`, `src/kernels/neon/`), `quant_q4_0.cpp`, or `ThreadPool` (`src/kernels/threading/`) without coordinating with **I**.

---

## B. Technical Background & Exact Implementation Details

### 1. SentencePiece C++ Integration & Dual-Processor Dummy-Prefix Control (`src/tokenizer/tokenizer.cpp`) (`VERIFIED`)
- **Vocabulary & Special Token Contract:**
  - `vocab_size = 32000`, `<unk> = 0`, `<s> = 1` (BOS), `</s> = 2` (EOS), `pad_id = -1` (disabled).
  - During `Tokenizer::init(model_path)`: loads the `.model` protobuf into two `sentencepiece::SentencePieceProcessor` instances:
    1. `processor_`: standard processor with default `add_dummy_prefix = true` (prepends SentencePiece `▁` before the first piece of a normal text segment).
    2. `processor_no_prefix_`: configured via ` SetEncodeExtraOptions("no-bos:no-eos")` and mutated `ModelProto` (`normalizer_spec()->set_add_dummy_prefix(false)`) so text immediately following an explicit special token (`<s>`, `</s>`, `<unk>`) without a space does **not** receive a spurious leading `▁` space token (`29871`).
- **Special-Token Aware `Tokenizer::encode(text, add_bos, add_eos, out_ids)`:**
  - Scans `text` for literal occurrences of `<unk>`, `<s>`, and `</s>`.
  - Emits exact special token IDs (`0`, `1`, `2`) for matched literals and encodes intervening plain-text spans with `processor_` (at start of string or after whitespace) or `processor_no_prefix_` (immediately adjacent to a preceding special token), achieving 100% parity with Hugging Face `LlamaTokenizer`.
- **Special-Token Aware `Tokenizer::decode(ids, out_text)`:**
  - Filters out BOS (`1`) and EOS (`2`), maps `<unk>` (`0`) to `" <unk> "`, and decodes contiguous runs of normal piece IDs via SentencePiece.
- **Zephyr Chat Template (`Tokenizer::apply_chat_template`):**
  - Formats `std::vector<ChatMessage>` as:
    ```text
    <|role|>
    {content}</s>
    ```
  - Appends `<|assistant|>\n` when `add_generation_prompt == true`.

### 2. Token Sampler (`src/sampler/sampler.cpp`) (`VERIFIED`)
- **Deterministic PRNG:** Uses a 64-bit `xorshift*` generator (`next_u64()` with multiplier `0x2545F4914F6CDD1DULL` and 53-bit uniform double conversion `(next_u64() >> 11) * (1.0 / 9007199254740992.0)`). If `seed == 0`, substitutes default seed `0x9E3779B97F4A7C15ULL`.
- **Sampling Pipeline (`Sampler::sample(logits, vocab_size, out_token_id)`):**
  1. Validates `logits != nullptr` and `vocab_size > 0`.
  2. **Greedy Fast Path:** If `temperature <= 0.0f` or `top_p <= 0.0f`, calls `sample_greedy()` which returns the lowest index of the maximum finite logit (falling back to token `0` if all logits are `NaN`/`-Inf`).
  3. **Temperature & Softmax:** Divides finite logits by `temperature` (skipping `NaN` and `-Inf` by assigning probability `0.0`), subtracts `max_logit`, exponentiates in `double` precision, and normalizes probabilities. If all probabilities underflow to `0`, falls back to greedy argmax.
  4. **Top-p (Nucleus) Filtering:** When `0.0f < top_p < 1.0f`, sorts `(prob, token_id)` pairs in descending order of probability (tie-breaking by lower `token_id`), retains the smallest prefix whose cumulative probability $\ge \text{top\_p}$ (always keeping at least 1 token), renormalizes the retained prefix, and draws a sample using `next_f64()`.

### 3. CLI Generation Loop & Benchmark Harness (`src/app/main.cpp`) (`VERIFIED`)
- **CLI Modes:**
  - `--help` / `--info`: Prints engine metadata, frozen TinyLlama config, and detected CPU SIMD backend.
  - `--tokenize --tokenizer <path> [--prompt <text> | --chat-system <sys> --chat-user <usr>] [--add-bos] [--add-eos]`: Prints comma-separated token IDs (`IDs: [...]`).
  - `--decode <id0,id1,...> --tokenizer <path>`: Decodes comma-separated IDs back to text.
  - **Full Generation Mode (`--model <path.ie> --tokenizer <tokenizer.model> --prompt <text>`):**
    1. Opens `.ie` model via `MmapModelLoader::open`, binds weights via `bind_tinyllama_weights`, and initializes `TransformerEngine` with `config`, `weights`, `backend`, and `threads`.
    2. Formats prompt (raw or Zephyr chat template if `--system` is passed) and encodes with `add_bos = true`.
    3. **Prefill (`i = 0 .. prompt_ids.size() - 1`):** Calls `engine.forward_token(prompt_ids[i], i, logits.data())`.
    4. **First Sample (`pos = prompt_ids.size() - 1`):** Samples first generated token from the final prefill logits; records **Time To First Token (TTFT)**.
    5. **Autoregressive Decode:** For each generated token `next_token` (while `next_token != eos_id`, `step < max_new_tokens`, and `next_pos < max_seq_len`), calls `engine.forward_token(next_token, next_pos, logits.data())` and samples the subsequent token.
    6. **Detokenisation Note (`PARTIALLY IMPLEMENTED` — Batch Decode):** Currently `src/app/main.cpp` collects `generated_ids` during the decode loop and calls `tokenizer.decode(generated_ids, &generated_text)` once after generation completes rather than streaming piece-by-piece to `stdout` during decode.
    7. **`--benchmark` Output:** Prints structured latency/throughput metrics (`Prefill latency`, `Prefill throughput`, `TTFT`, `Decode latency`, `Decode throughput`, `Wall time`, `CPU time`, `CPU utilization %` via `getrusage`/`GetProcessTimes`, and `Peak RSS MB`).

---

## C. Completed Work, Commits, and Verification Evidence

| Deliverable | Commit(s) | Classification | Evidence |
| :--- | :--- | :--- | :--- |
| SentencePiece C++ `Tokenizer` (dual processor, special tokens, Zephyr chat template) | `ce769c8` (`origin/feature/runtime-tokenizer`) | `REAL-CHECKPOINT VALIDATED` | `tokenizer` CTest (`4/4` passed) + `tools/verify_tokenizer_parity.py --self-test` (`15/15` prompts exact match against real `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`) |
| `Sampler` (greedy, temperature, top-p, seeded 64-bit `xorshift*`, `NaN`/`Inf` safety) | `ce769c8` (`origin/feature/runtime-tokenizer`) | `VERIFIED` / `TESTED` | `sampler` CTest (`6/6` passed) |
| CLI generation runtime & `--benchmark` harness (`src/app/main.cpp`) | `ce769c8`, `5929a17` | `SYNTHETICALLY VALIDATED` | Verified E2E with synthetic 1-layer `.ie` model + real `tokenizer.model` (`338.9 tok/s` prefill, `329.1 tok/s` decode on 4 threads) |
| GitHub Actions CI workflow (`.github/workflows/ci.yml`) | `ce769c8`, `33d4bdc` | `VERIFIED` | Configured for `ubuntu-22.04` and `macos-latest` with SentencePiece parity step |

---

## D. Current State: Synthetic vs. Real-Checkpoint Verification Boundary

- **Real-Checkpoint Validated (`REAL-CHECKPOINT VALIDATED`):**
  - `Tokenizer` has been validated directly against the real `models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model` (`499,723` bytes) and `tokenizer_config.json` across 15 diverse prompts (BOS/EOS flags, multi-turn Zephyr chat templates, punctuation, numbers, CJK/emoji UTF-8 byte fallbacks, and adjacent special tokens).
- **Synthetically Validated Only (`SYNTHETICALLY VALIDATED`):**
  - The full CLI generation loop (`--model ... --tokenizer ... --prompt ... --benchmark`) has been tested end-to-end using a synthetic 1-layer `Q4_0` `.ie` model and real `tokenizer.model`.
  - It has **not yet** been executed against the full 22-layer pretrained `TinyLlama-1.1B-Chat-v1.0` `.ie` checkpoint because `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` on disk is currently partially downloaded (`325.47 MB` / `2.20 GB`).

---

## E. Dependencies on Other Team Members

- **On Member S:** Completion of the 2.20 GB `model.safetensors` download and conversion into `models/tinyllama-fp32.ie`, `models/tinyllama-q4_0.ie`, and `models/tinyllama-q4_0-mixed.ie`.
- **On Member U:** 22-layer `TransformerEngine::forward_token` execution (already wired into `src/app/main.cpp`).
- **On Member I:** Multi-threaded SIMD kernel dispatch (`dispatch_matmul_*` + `ThreadPool`, already wired into `TransformerEngine`).

---

## F. Remaining Tasks (Priority Order)

### `P0` — Blocks the Entire Project
1. **Execute End-to-End 22-Layer Real-Checkpoint CLI Generation & Benchmark Matrix Once `.ie` Models Are Converted:**
   - Run greedy (`--temp 0`) and stochastic (`--temp 0.7 --top-p 0.9 --seed 42`) generation on `models/tinyllama-q4_0.ie`, `models/tinyllama-q4_0-mixed.ie`, and `models/tinyllama-fp32.ie`:
     ```bash
     ./build/inference-engine \
       --model models/tinyllama-q4_0.ie \
       --tokenizer models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
       --system "You are a helpful assistant." \
       --prompt "Explain why custom C++ inference engines are fast in two sentences." \
       --max-new-tokens 64 \
       --temp 0.0 \
       --threads 8 \
       --benchmark
     ```
   - Record real 22-layer TTFT, prefill tok/s, decode tok/s, and Peak RSS MB across thread counts (`1, 2, 4, 8`) in `docs/BENCHMARKING.md`.

### `P1` — Required for the Next Major Milestone
2. **Implement Incremental Streaming Detokenisation During Decode in `src/app/main.cpp`:**
   - Currently `src/app/main.cpp` decodes all `generated_ids` in a single batch after the decode loop finishes.
   - Add incremental prefix-diff decoding (decoding `generated_ids[0..k]` vs `generated_ids[0..k-1]` and flushing newly completed UTF-8 characters to `std::cout << std::flush` on every token step) so users see real-time token streaming during interactive CLI runs, while keeping `Tokenizer::decode(generated_ids, ...)` parity intact.

### `P2` — Important Improvements
3. **Add Automated CLI Integration Smoke Test to CTest (`tests/`):**
   - Add a lightweight CTest or Python smoke script that builds a tiny synthetic `.ie` model, invokes `./build/inference-engine --model ... --tokenizer models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model --prompt "Hello" --max-new-tokens 4 --temp 0 --benchmark`, and asserts exit code `0` and non-empty output.
4. **Multi-Iteration Warmup/Measurement Flag (`--bench-runs <N>`) in `src/app/main.cpp`:**
   - Allow `--benchmark` to optionally run 1 warmup pass + $N$ measured passes (calling `engine.kv_cache().clear()` between passes) and report median/mean/p95 prefill and decode tok/s per [`docs/BENCHMARKING.md`](../BENCHMARKING.md).

### `P3` — Optional Optimisation or Polish
5. **Top-k Sampling (`--top-k`) or Repetition Penalty (`--repeat-penalty`):**
   - Optional sampling extensions in `SamplingParams` and `Sampler::sample` if desired for longer open-ended generation demos.

---

## G. New-Session Continuation Prompt (Ready to Paste into Antigravity)

```markdown
You are operating as **Member P (Tokenizer, Sampling, Generation Runtime, Benchmarking & CI Owner)** on the **BIOS V2 — PS08 (`inference-engine`)** project in repository `shreyaas5/inference-engine`.

Follow this exact 10-step autonomous workflow without asking the team lead for background already documented in the repository:

1. **Read the Master Project Context:** Read `AGENTS.md` and `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`.
2. **Read Your Complete Handoff & Shared Interfaces:** Read `docs/team_handoffs/04_P_TOKENIZER_AND_RUNTIME.md`, `docs/team_handoffs/05_SHARED_INTERFACES_AND_DEPENDENCIES.md`, and `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`.
3. **Inspect the Current Git State:** Run `git fetch --all --prune`, `git status`, `git branch -a -vv`, and `git log --all --oneline --graph -n 20`.
4. **Identify the Latest Relevant Commits:** Verify the consolidated baseline (`integration-ps08-team-sync-2026-10-31` at `33d4bdc4486e83f458b6a242d9a2fd051503e6ff`) and your merged feature branch `origin/feature/runtime-tokenizer` (`ce769c882f0883613183812738340195a1f7d71e`).
5. **Inspect the Actual Source Files:** Read `include/inference_engine/{tokenizer,sampler}.hpp`, `src/tokenizer/tokenizer.cpp`, `src/sampler/sampler.cpp`, `src/app/main.cpp`, `tools/verify_tokenizer_parity.py`, `tests/test_tokenizer.cpp`, `tests/test_sampler.cpp`, and `.github/workflows/ci.yml`. Also check `ls -lh models/` and `ls -lh models/TinyLlama-1.1B-Chat-v1.0/`.
6. **Compare Implementation Against Documented State:** Confirm that `Tokenizer` (dual SentencePiece processor), `Sampler` (64-bit `xorshift*`, greedy/temperature/top-p), and `src/app/main.cpp` match the documented implementation, and note that `src/app/main.cpp` currently performs batch detokenisation after the decode loop rather than per-token streaming output.
7. **Avoid Reimplementing Completed Functionality:** SentencePiece integration, special-token splitting, Zephyr chat template formatting, `Sampler`, and `tools/verify_tokenizer_parity.py` (`15/15` prompts passing) are already implemented and verified. Do not rewrite them.
8. **Identify the Highest-Priority Unfinished Tasks (`P0` / `P1` / `P2`):**
   - If the full 22-layer `.ie` checkpoint (`models/tinyllama-q4_0.ie` / `models/tinyllama-fp32.ie`) is available on disk, execute `P0`: run end-to-end 22-layer CLI generation and `--benchmark` across thread counts (`1, 2, 4, 8`) and record the real metrics in `docs/BENCHMARKING.md`.
   - Regardless of whether the 2.20 GB checkpoint download has finished, proceed immediately to `P1` & `P2`:
     1. Implement incremental per-token streaming detokenisation in `src/app/main.cpp` (using prefix-diff decoding so SentencePiece word-boundary spaces `▁` are preserved accurately while streaming tokens to `stdout` in real time).
     2. Optionally add `--bench-runs <N>` (calling `engine.kv_cache().clear()` between iterations) and an automated CLI E2E smoke test.
9. **Produce an Initial Repository Audit and Action Plan:** Output a concise summary of your audit, test results (`ctest --test-dir build --output-on-failure` and `python3 tools/verify_tokenizer_parity.py --self-test`), and your implementation plan.
10. **Begin Implementation Immediately:** Work autonomously within your owned files (`src/tokenizer/`, `src/sampler/`, `src/app/main.cpp`, `tests/test_{tokenizer,sampler}.cpp`, `tools/verify_tokenizer_parity.py`, `.github/workflows/ci.yml`). Ask the team lead only if a change affects shared headers outside your ownership. Update context files per `AGENTS.md` Section 8 before finishing.
```
