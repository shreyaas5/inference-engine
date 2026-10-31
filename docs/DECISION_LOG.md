# Architectural Decision Log (`DECISION_LOG.md`)

## ADR-001: Project Identity and Namespace Standardization (`inference-engine` / `inference_engine`)

- **Date:** 2026-10-31
- **Status:** Accepted (Frozen)
- **Decision:** Standardize the project name, repository name, root directory, and CLI executable as `inference-engine`, and the C++ namespace as `inference_engine`.
- **Rationale:** Ensures consistent, unambiguous naming across CMake targets, C++ headers, CLI tooling, documentation, and GitHub workflows.

## ADR-002: Explicit Little-Endian `.ie` Binary Format with 64-Byte Alignment

- **Date:** 2026-10-31
- **Status:** Accepted (Frozen)
- **Decision:** Design `.ie` v1 with explicit little-endian field-by-field serialization, a 64-byte header, a self-describing tensor directory, 64-byte aligned tensor payloads, checked 64-bit arithmetic, and non-overlap verification.
- **Rationale:** Raw C++ struct dumps break across compilers/ABIs and are vulnerable to integer overflow or misaligned SIMD loads on corrupted inputs. 64-byte alignment matches CPU cache lines and allows zero-copy `mmap` backing for both NEON and AVX2 kernels.

## ADR-003: Upstream-Compatible 18-Byte `Q4_0` Block Layout

- **Date:** 2026-10-31
- **Status:** Accepted (Frozen)
- **Decision:** Use 32 weights per `Q4_0` block stored in 18 bytes (`uint16_t d` FP16 scale + `uint8_t qs[16]`), where low nibbles store indices `0..15`, high nibbles store indices `16..31`, and dequantisation computes `(q - 8) * fp16_to_fp32(d)`.
- **Rationale:** Enables direct SIMD nibble unpacking (`& 0x0F` and `>> 4`) on 16-byte vector registers and guarantees apples-to-apples numerical and performance comparability against `llama.cpp` `Q4_0`.

## ADR-004: Decoupled `KvCache::append()` and `KvCache::commit_token()`

- **Date:** 2026-10-31
- **Status:** Accepted (Frozen)
- **Decision:** `KvCache::append(layer_idx, pos, k, v)` writes K/V vectors for a single layer and position without mutating `seq_len()`. A separate `KvCache::commit_token()` (or `commit_positions()`) advances `seq_len()` only after all 22 layers finish.
- **Rationale:** Prevents a classic bug where calling `append()` inside the 22-layer loop increments sequence length 22 times per token.

## ADR-005: SentencePiece C++ Integration for Tokenization

- **Date:** 2026-10-31
- **Status:** Accepted
- **Decision:** Use SentencePiece C++ for loading TinyLlama's `tokenizer.model` in M1 rather than writing a custom BPE engine from scratch, verified via `tools/verify_tokenizer_parity.py`.
- **Rationale:** TinyLlama uses a standard SentencePiece unigram/BPE model with byte fallback (`<0x00>`..`<0xFF>`). Using SentencePiece eliminates subtle tokenization divergence while keeping all neural inference 100% custom C++17.
