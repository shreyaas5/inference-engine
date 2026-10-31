# `MASTER_CONTEXT.md` — Comprehensive Project Context

**Project:** `inference-engine`  
**Competition:** BIOS V2 — Problem Statement **PS08 (Own Inference Engine)**  
**Local Repository:** `/Users/shreyaas/Developer/inference-engine`  
**Remote Repository:** https://github.com/shreyaas5/inference-engine (Private)  
**Last Updated:** 2026-10-31

---

## 1. Problem Statement Interpretation (BIOS V2 — PS08)

Problem Statement **PS08 — Own Inference Engine** challenges engineering teams to build a custom language model inference runtime from the ground up rather than wrapping an existing framework (such as PyTorch, ONNX Runtime, TensorFlow, or `llama.cpp`).

To succeed under PS08, `inference-engine` is engineered around four pillars:

1. **Zero-Framework C++17 Execution:** Every tensor lookup, normalization, positional rotation, matrix-vector product, attention head calculation, KV cache update, and token sampling step is implemented directly in C++17 (`namespace inference_engine`).
2. **Real 1.1B Transformer Scale:** Rather than a toy model, the runtime targets `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, 2048 hidden size, 5632 SwiGLU intermediate size, 32 query heads, 4 KV heads, 32,000 vocabulary).
3. **Systems-Level Memory & Quantisation Efficiency:** Using a custom 64-byte aligned `.ie` v1 binary file format loaded via `mmap` and 18-byte `Q4_0` block quantisation, the model footprint drops from ~4.4 GB (`FP32`) to ~630 MB (`Q4_0`), enabling smooth CPU inference even on an 8 GB RAM laptop.
4. **Rigorous Verification & Cross-Architecture Benchmarking:** Numerical parity is verified stage-by-stage against a trusted `FP32` Hugging Face golden reference before enabling `Q4_0`, ARM64 `NEON`, x86-64 `AVX2`, and multi-threaded execution, culminating in a 4-laptop benchmark comparison against `llama.cpp`.

---

## 2. End-to-End Inference Pipeline

```text
[Offline Phase — Python 3 (tools/)]
  Hugging Face Checkpoint (TinyLlama-1.1B-Chat-v1.0, BF16 safetensors)
       |
       +--> inspect_checkpoint.py      (Verify hyperparameters & 201 tensors)
       +--> convert_to_inference_engine.py (Cast BF16->FP32, optional Q4_0 quant, write 64B-aligned .ie v1)
       +--> generate_golden_reference.py   (Export FP32 intermediate activations, logits & greedy tokens)
       +--> verify_tokenizer_parity.py     (Verify SentencePiece token IDs & Zephyr chat template)

[Runtime Phase — Native C++17 (inference-engine)]
  .ie v1 Model File + SentencePiece tokenizer.model + User Prompt
       |
       v
  1. ModelLoader (src/loader/loader.cpp)
       - Validates 64B header, directory, 64B alignment, bounds & non-overlap
       - Maps tensor payloads zero-copy via POSIX mmap
       |
       v
  2. Tokenizer (src/tokenizer/tokenizer.cpp)
       - Formats Zephyr chat prompt (<|system|>, <|user|>, <|assistant|>)
       - Encodes prompt text into token IDs t_0 .. t_{N-1}
       |
       v
  3. TransformerEngine (src/transformer/transformer.cpp)
       - Prefill loop: pos = 0 .. N-1
           For layer = 0 .. 21:
             RMSNorm -> QKV MatMul (FP32 or Q4_0) -> RoPE(pos)
             -> KvCache::append(layer, pos, K, V) [does NOT advance seq_len]
             -> Causal GQA (32 Q heads : 4 KV heads) -> O_proj -> Residual
             -> RMSNorm -> SwiGLU (Gate & Up) -> Down_proj -> Residual
           KvCache::commit_token() [advances seq_len once after all 22 layers]
       - Final RMSNorm & LM Head at pos = N-1 -> Logits [32000]
       |
       v
  4. Sampler (src/sampler/sampler.cpp) & Decode Loop
       - Samples first generated token \hat{t}_N from pos = N-1 logits
       - Feeds \hat{t}_N into forward_token at pos = N, then N+1, ... until EOS or max_new_tokens
```

---

## 3. Hardware Environment & Four-Laptop Benchmark Strategy

The team operates across four distinct commodity laptop architectures, providing natural coverage of Apple Silicon (`ARM64 NEON`) and both AMD and Intel `x86-64 AVX2`:

| Member | Laptop Hardware | Memory | OS / Architecture | Target Backend |
|---|---|---|---|---|
| **S** | Apple MacBook Air M4 | 24 GB Unified | macOS ARM64 | `NEON` |
| **I** | AMD Ryzen 7 7745HX (RTX 4060 disabled for benchmarks) | 16 GB DDR5 | Windows/Linux x86-64 | `AVX2` + `FMA` |
| **U** | Intel Core Ultra 7 155H | 16 GB DDR5 | Windows/Linux x86-64 | `AVX2` + `FMA` |
| **P** | Intel Core i5-10210U | 8 GB DDR4 | Windows/Linux x86-64 | `AVX2` (runtime verified) |

---

## 4. Document Map

- **Entry & Rules:** [`AGENTS.md`](../../AGENTS.md)
- **Quick Identity:** [`PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md)
- **Live Progress:** [`CURRENT_STATUS.md`](../../CURRENT_STATUS.md)
- **Immediate Next Steps:** [`SESSION_HANDOFF.md`](../../SESSION_HANDOFF.md)
- **Architectural Decisions:** [`DECISIONS.md`](../../DECISIONS.md)
- **Milestones M0–M4:** [`ROADMAP.md`](../../ROADMAP.md)
- **Team & Issues:** [`TEAM_STATUS.md`](../../TEAM_STATUS.md) and [`docs/context/TEAM_HANDOFF.md`](TEAM_HANDOFF.md)
- **Technical Contracts:** [`docs/context/TECHNICAL_CONTEXT.md`](TECHNICAL_CONTEXT.md), [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), [`docs/DATA_FLOW.md`](../DATA_FLOW.md), [`docs/FORMAT_SPECIFICATION.md`](../FORMAT_SPECIFICATION.md), [`docs/QUANTISATION.md`](../QUANTISATION.md), [`docs/VALIDATION.md`](../VALIDATION.md), [`docs/BENCHMARKING.md`](../BENCHMARKING.md)
- **Module Matrix:** [`docs/context/IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)
- **Known Issues & Stubs:** [`docs/context/KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
- **Ordered Backlog:** [`docs/context/NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
- **Session History:** [`docs/context/SESSION_LOG.md`](SESSION_LOG.md)
- **Copy-Paste Prompt for Fresh AI Sessions:** [`docs/context/NEW_SESSION_PROMPT.md`](NEW_SESSION_PROMPT.md)

