# `NEW_SESSION_PROMPT.md` — Copy-Paste Startup Prompt for Fresh AI Sessions

Copy and paste the block below into any brand-new AI coding session (Antigravity, Claude Code, Codex, Cursor, ChatGPT, etc.) when resuming work on `inference-engine`.

---

```markdown
# CONTEXT HANDOFF — RESUME WORK ON `inference-engine`

You are joining an active C++17 systems-engineering repository at:
`/Users/shreyaas/Developer/inference-engine` (`~/Developer/inference-engine`)

Remote repository: `https://github.com/shreyaas5/inference-engine` (Private)

## 1. Project Identity & Locked Constraints
- **Project Name / Executable:** `inference-engine` (C++ namespace: `inference_engine`).
- **Competition Context:** BIOS V2 — Problem Statement PS08 (Own Inference Engine). NEVER use "BIOS" as the project name, namespace, binary name, or folder name.
- **Target Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, hidden_dim=2048, intermediate_dim=5632, 32 query heads, 4 KV heads [GQA 8:1], head_dim=64, vocab_size=32000, max_seq_len=2048, RMSNorm eps=1e-5, RoPE theta=10000.0, SwiGLU).
- **Runtime Rules:** 100% custom C++17 CPU-only inference (Scalar reference + ARM64 NEON + x86-64 AVX2 + persistent `std::thread` pool). No PyTorch, TensorFlow, ONNX Runtime, or `llama.cpp` in the inference runtime. Python (`tools/`) is restricted strictly to offline checkpoint conversion, tokenizer parity verification, and generating Hugging Face FP32 golden references.
- **Critical Technical Contracts:**
  1. `.ie` v1 model format uses explicit little-endian serialization (no raw struct dumps), a 64-byte header, tensor directory, 64-byte aligned tensor payloads (`payload_offset % 64 == 0`), checked 64-bit integer arithmetic, exact byte-size validation, and non-overlap validation.
  2. `Q4_0` uses 32 weights per 18-byte block (`2B` FP16 scale `d` + `16B` packed nibbles `qs[0..15]`), where low nibbles (`qs[j] & 0x0F`) store indices `0..15`, high nibbles (`(qs[j] >> 4) & 0x0F`) store indices `16..31`, and dequantisation is `(q - 8) * fp16_to_fp32(d)`.
  3. `KvCache::append(layer_idx, pos, k, v)` writes K/V for a single layer and position and MUST NOT increment `seq_len()`. `KvCache::commit_token()` advances `seq_len()` ONCE after all 22 layers finish processing the token.
  4. Autoregressive position indexing: an N-token prompt (e.g. 12 tokens) is processed at positions `0..N-1` (`0..11`). Final prompt logits at position `N-1` (`11`) predict the first generated token, which is subsequently processed at position `N` (`12`), then `13`, `14`, etc.
- **4-Person Team Structure:**
  - **S** (`feature/model-loader`): Project lead, `.ie` loader, offline converter (`tools/convert_to_inference_engine.py`).
  - **I** (`feature/quantisation-simd`): `Q4_0`, NEON, AVX2, `ThreadPool`, kernel dispatch.
  - **U** (`feature/transformer`): Scalar reference kernels, `KvCache`, 22-layer `TransformerEngine`, golden validation.
  - **P** (`feature/runtime-tokenizer`): SentencePiece C++ tokenizer, sampler, CLI, CI, benchmarks.

## 2. Mandatory Startup Procedure (Execute Before Writing Any Code)
Do not guess or give generic advice. Perform these steps right now:

1. Read the context files in this exact order:
   - `AGENTS.md`
   - `CURRENT_STATUS.md`
   - `SESSION_HANDOFF.md`
   - `PROJECT_CONTEXT.md`
   - `docs/context/IMPLEMENTATION_STATUS.md`
   - `docs/context/KNOWN_ISSUES.md`
   - `docs/context/NEXT_ACTIONS.md`
   - `docs/context/TECHNICAL_CONTEXT.md`
2. Inspect the live Git and build state:
   - Run `git status`, `git branch -a`, and `git log -n 5 --oneline --decorate`.
   - Run `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 && ctest --test-dir build --output-on-failure`.
3. Treat the repository code as the source of truth for what is currently implemented, and `DECISIONS.md` + `docs/context/TECHNICAL_CONTEXT.md` as the source of truth for intended design.
4. Check for any contradictions between the documentation and actual code/Git state.

## 3. Your First Response
In your first response, provide a concise report covering:
1. Current Git branch, latest commit, working tree status, and CTest pass/fail results.
2. Any contradictions or uncommitted changes discovered between documentation and code.
3. A clear summary of what is actually implemented vs what is currently stubbed.
4. The highest-priority next action from `SESSION_HANDOFF.md` / `docs/context/NEXT_ACTIONS.md` (or ask me which team member role `S`, `I`, `U`, or `P` we are executing if not specified).

Do not redo completed M0 work, do not alter frozen contracts without approval, never work directly on `main`, and remember to update `CURRENT_STATUS.md`, `SESSION_HANDOFF.md`, `docs/context/IMPLEMENTATION_STATUS.md`, `docs/context/KNOWN_ISSUES.md`, `docs/context/NEXT_ACTIONS.md`, and `docs/context/SESSION_LOG.md` before ending a substantial coding session.
```

