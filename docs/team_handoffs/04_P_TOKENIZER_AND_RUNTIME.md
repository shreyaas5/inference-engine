# `04_P_TOKENIZER_AND_RUNTIME.md` — Member P Module Handoff

**Owner:** Member P (SentencePiece C++ Tokenizer, Zephyr Chat Template, Token Sampler, CLI Application, Benchmarking Harness, & CI Workflow)
**Consolidated Branch:** `integration-ps08-team-sync-2026-10-31`
**Merged Feature Branch:**
- `origin/feature/runtime-tokenizer` (`ce769c8`) — SentencePiece C++ integration (`src/tokenizer/tokenizer.cpp`, `CMakeLists.txt`), stochastic temperature + top-p nucleus `Sampler` (`src/sampler/sampler.cpp`, `tests/test_sampler.cpp`), CLI inference & `--benchmark` harness (`src/app/main.cpp`), `tools/verify_tokenizer_parity.py`, and `.github/workflows/ci.yml`

---

## 1. Owned Files & Responsibilities

| File Path | Role | Status |
|---|---|---|
| [`include/inference_engine/tokenizer.hpp`](../../include/inference_engine/tokenizer.hpp), [`src/tokenizer/tokenizer.cpp`](../../src/tokenizer/tokenizer.cpp) | SentencePiece C++ wrapper (`Tokenizer::load`, `encode`, `decode`), special-token splitting (`<unk>=0`, `<s>=1`, `</s>=2`), and Zephyr chat template (`Tokenizer::format_chat_prompt`) | `REFERENCE VERIFIED (Real tokenizer.model)` |
| [`include/inference_engine/sampler.hpp`](../../include/inference_engine/sampler.hpp), [`src/sampler/sampler.cpp`](../../src/sampler/sampler.cpp) | Greedy argmax (`temperature <= 0`), temperature scaling, top-p nucleus truncation, seeded `std::mt19937_64` RNG, and non-finite logit filtering | `TESTED (8/8 CTest)` |
| [`src/app/main.cpp`](../../src/app/main.cpp) | CLI executable (`inference-engine`): `--model`, `--tokenizer`, `--prompt`, `--chat`, `--threads`, `--max-tokens`, `--temp`, `--top-p`, `--seed`, `--benchmark`, `--encode-only`, `--decode-only`, `--format-chat`, `--info` | `TESTED (Synthetic .ie + Real Tokenizer)` |
| [`tools/verify_tokenizer_parity.py`](../../tools/verify_tokenizer_parity.py) | Offline self-test + live Hugging Face `AutoTokenizer` vs C++ `--encode-only`/`--decode-only`/`--format-chat` parity harness | `TESTED (--self-test)` |
| [`tests/test_tokenizer.cpp`](../../tests/test_tokenizer.cpp), [`tests/test_sampler.cpp`](../../tests/test_sampler.cpp) | Model-backed `tokenizer` CTest suite and deterministic `sampler` CTest suite | `TESTED (8/8 CTest)` |
| [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml), [`docs/BENCHMARKING.md`](../../docs/BENCHMARKING.md) | GitHub Actions CI pipeline and 4-laptop benchmark methodology | `IMPLEMENTED` |

---

## 2. Frozen Technical Contracts Owned by P

### 2.1 SentencePiece C++ & Special Token Contract
- `Tokenizer::load(model_path)` loads `tokenizer.model` via `sentencepiece::SentencePieceProcessor` and verifies:
  - `GetPieceSize() == 32000` (`kTinyLlamaVocabSize`)
  - `unk_id() == 0` (`<unk>`), `bos_id() == 1` (`<s>`), `eos_id() == 2` (`</s>`)
- `Tokenizer::encode(text, add_bos, add_eos, out_ids)` splits input text on exact literal occurrences of `<unk>`, `<s>`, and `</s>` (matching Hugging Face `LlamaTokenizer` added-token splitting) and encodes ordinary text segments via SentencePiece `Encode(segment, &piece_ids)`.
- `Tokenizer::decode(ids, out_text)` groups contiguous ordinary token runs for SentencePiece `Decode(ordinary_run, &piece)` while rendering `0 -> "<unk>"`, `1 -> "<s>"`, and `2 -> "</s>"`, preserving leading spaces across special-token boundaries.

### 2.2 Zephyr Chat Template Contract (`Tokenizer::format_chat_prompt`)
Matches `TinyLlama/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json` byte-for-byte:
```text
<|system|>
{system_content}</s>
<|user|>
{user_content}</s>
<|assistant|>
```
Role tags (`<|system|>`, `<|user|>`, `<|assistant|>`) are tokenized as ordinary text pieces, whereas `</s>` is split as special token ID `2`. When `--chat` is passed to `inference-engine`, the formatted prompt is encoded with `add_bos = true, add_eos = false`.

### 2.3 CLI Position Indexing & Benchmark Telemetry Contract
- Given `prompt_ids` of length $N$:
  1. **Prefill:** Loops `pos = 0 .. N - 1`, calling `engine.forward_token(prompt_ids[pos], pos, logits.data())`.
  2. **First Generated Token & TTFT:** Samples `next_token = sampler.sample(logits.data(), logits.size())` from the logits produced at `pos = N - 1` and records `ttft_ms` immediately.
  3. **Autoregressive Decode:** For `generated_count = 1 .. max_new_tokens - 1` while `next_token != kEosTokenId`:
     - Computes `position = prompt_ids.size() + generated_count - 1` (`N, N+1, ...`)
     - Calls `engine.forward_token(next_token, position, logits.data())`
     - Samples `next_token = sampler.sample(logits.data(), logits.size())`
- `--benchmark` prints measured:
  - `Model load` (`ms`), `TTFT` (`ms`), `Prompt tokens`, `Prompt throughput` (`tokens/s`), `Generated tokens`, `Decode throughput` (`tokens/s`), `Peak resident memory` (`MiB` via `getrusage` / `GetProcessMemoryInfo`), `CPU utilization` (`%`), and `Threads requested`.

---

## 3. Ready-to-Paste Antigravity Continuation Prompt for Member P

```markdown
You are acting as Member P (SentencePiece Tokenizer, Sampler, CLI Runtime & Benchmarking) for `inference-engine` (BIOS V2 — PS08).

Before making any changes, read:
1. `AGENTS.md`
2. `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`
3. `docs/team_handoffs/04_P_TOKENIZER_AND_RUNTIME.md`
4. `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`

Current branch: `integration-ps08-team-sync-2026-10-31` (consolidates S `4515579`, U `ce1026a`, I `58c604d`, and P `ce769c8`; all 8/8 CTest suites including `tokenizer` and `sampler` pass).

Your next engineering tasks:
1. In a Python environment with `transformers` and `sentencepiece` installed, run the full live Hugging Face vs C++ binary parity check:
   `python3 tools/verify_tokenizer_parity.py --model-dir models/TinyLlama-1.1B-Chat-v1.0 --engine-bin ./build/inference-engine`
2. Once the converted `models/tinyllama-1.1b-q4_0.ie` and `models/tinyllama-1.1b-fp32.ie` files are available, run end-to-end CLI generation (`--chat` and raw `--prompt`) and collect `--benchmark` runs across `--threads 1, 2, 4, 8` on your Intel Core i5-10210U laptop (and coordinate with S, I, and U to populate the 4-laptop comparison table in `docs/BENCHMARKING.md`).
```
