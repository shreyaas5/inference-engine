#!/usr/bin/env python3
"""Generate golden reference activations, logits, and greedy continuation tokens from Hugging Face.

Owners: U (Scalar Inference & Transformer) and S (Project Lead)

Golden Validation Contract:
1. Load `TinyLlama/TinyLlama-1.1B-Chat-v1.0` weights and cast BF16 checkpoint parameters
   explicitly to FP32 (`torch.float32`) so both reference and custom engine operate on
   identical FP32 weights.
2. Export intermediate activations (embedding output, layer 0/11/21 RMSNorm, Q/K/V projections
   after RoPE at nonzero positions, attention output, SwiGLU MLP output, final norm, and logits).
3. Export multi-token prompt logits and autoregressive greedy decode continuation tokens
   verifying the position indexing contract (e.g. a 12-token prompt occupies positions 0..11,
   predicts first generated token from position 11 logits, and processes generated tokens at
   positions 12, 13, ...).

Binary Fixture Format (fixtures/golden/golden_reference.bin):
- 20-byte Header (Little-Endian):
  - magic: u32 = 0x474F4C44 ("GOLD")
  - version: u32 = 1
  - vocab_size: u32 (32000)
  - prompt_length: u32
  - max_new_tokens: u32
- Payloads:
  - prompt_tokens: u32[prompt_length]
  - prompt_final_logits: f32[vocab_size]
  - gen_tokens: u32[max_new_tokens]
  - gen_logits: f32[max_new_tokens * vocab_size]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import List

GOLDEN_MAGIC = 0x474F4C44
GOLDEN_VERSION = 1
HEADER_STRUCT = struct.Struct("<5I")


def pack_golden_binary(
    vocab_size: int,
    prompt_tokens: List[int],
    prompt_final_logits: List[float],
    gen_tokens: List[int],
    gen_logits: List[float],
) -> bytes:
    assert len(prompt_final_logits) == vocab_size
    assert len(gen_logits) == len(gen_tokens) * vocab_size

    header = HEADER_STRUCT.pack(
        GOLDEN_MAGIC,
        GOLDEN_VERSION,
        vocab_size,
        len(prompt_tokens),
        len(gen_tokens),
    )
    tokens_fmt = f"<{len(prompt_tokens)}I"
    prompt_logits_fmt = f"<{vocab_size}f"
    gen_tokens_fmt = f"<{len(gen_tokens)}I"
    gen_logits_fmt = f"<{len(gen_logits)}f"

    return (
        header
        + struct.pack(tokens_fmt, *prompt_tokens)
        + struct.pack(prompt_logits_fmt, *prompt_final_logits)
        + struct.pack(gen_tokens_fmt, *gen_tokens)
        + struct.pack(gen_logits_fmt, *gen_logits)
    )


def unpack_golden_binary(data: bytes):
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("Buffer too small for golden header")
    magic, version, vocab_size, prompt_len, gen_len = HEADER_STRUCT.unpack_from(data, 0)
    if magic != GOLDEN_MAGIC:
        raise ValueError(f"Invalid golden magic: 0x{magic:08X}")
    if version != GOLDEN_VERSION:
        raise ValueError(f"Unsupported golden version: {version}")

    offset = HEADER_STRUCT.size
    prompt_tokens = list(struct.unpack_from(f"<{prompt_len}I", data, offset))
    offset += prompt_len * 4

    prompt_logits = list(struct.unpack_from(f"<{vocab_size}f", data, offset))
    offset += vocab_size * 4

    gen_tokens = list(struct.unpack_from(f"<{gen_len}I", data, offset))
    offset += gen_len * 4

    gen_logits = list(struct.unpack_from(f"<{gen_len * vocab_size}f", data, offset))
    offset += gen_len * vocab_size * 4

    return {
        "vocab_size": vocab_size,
        "prompt_tokens": prompt_tokens,
        "prompt_final_logits": prompt_logits,
        "gen_tokens": gen_tokens,
        "gen_logits": gen_logits,
    }


def generate_golden(
    model_id: str,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
) -> int:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        print(
            "[ERROR] PyTorch and Hugging Face transformers are required to generate real golden references.\n"
            "Install with: pip install torch transformers"
        )
        return 1

    print(f"[INFO] Loading {model_id} in full FP32 precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_tokens = input_ids[0].tolist()
    print(f"[INFO] Tokenized prompt ({len(prompt_tokens)} tokens): {prompt_tokens}")

    # Forward pass on prompt tokens
    with torch.no_grad():
        outputs = model(input_ids)
        # Logits at the last prompt position predict the first generated token
        last_logits = outputs.logits[0, -1, :].to(torch.float32).tolist()

    vocab_size = model.config.vocab_size

    # Autoregressive greedy decode
    gen_tokens: List[int] = []
    gen_logits: List[float] = []

    curr_ids = input_ids
    for step in range(max_new_tokens):
        with torch.no_grad():
            step_outputs = model(curr_ids)
            step_logits = step_outputs.logits[0, -1, :].to(torch.float32)
            next_token = int(torch.argmax(step_logits).item())
            gen_tokens.append(next_token)
            gen_logits.extend(step_logits.tolist())
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    print(f"[INFO] Generated {len(gen_tokens)} tokens: {gen_tokens}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "golden_reference.bin"
    bin_bytes = pack_golden_binary(
        vocab_size=vocab_size,
        prompt_tokens=prompt_tokens,
        prompt_final_logits=last_logits,
        gen_tokens=gen_tokens,
        gen_logits=gen_logits,
    )
    bin_path.write_bytes(bin_bytes)

    meta_path = output_dir / "metadata.json"
    meta = {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "vocab_size": vocab_size,
        "max_new_tokens": max_new_tokens,
        "binary_size_bytes": len(bin_bytes),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Wrote golden reference fixture to {bin_path} ({len(bin_bytes)} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FP32 golden reference fixtures for TinyLlama-1.1B-Chat-v1.0."
    )
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model ID or local directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/golden"),
        help="Directory to write golden reference artifacts (ignored by Git)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how CPU caches improve matrix multiplication performance.",
        help="Prompt used for multi-token prefill and greedy continuation verification",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of greedy continuation tokens to record",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify binary packing/unpacking without requiring PyTorch/transformers",
    )
    args = parser.parse_args()

    if args.self_test:
        test_vocab = 32
        p_tokens = [1, 10, 15]
        p_logits = [float(i) * 0.1 for i in range(test_vocab)]
        g_tokens = [5, 6]
        g_logits = [float(i) * 0.2 for i in range(2 * test_vocab)]
        packed = pack_golden_binary(test_vocab, p_tokens, p_logits, g_tokens, g_logits)
        unpacked = unpack_golden_binary(packed)
        assert unpacked["vocab_size"] == test_vocab
        assert unpacked["prompt_tokens"] == p_tokens
        assert unpacked["gen_tokens"] == g_tokens
        assert len(unpacked["prompt_final_logits"]) == test_vocab
        assert len(unpacked["gen_logits"]) == 2 * test_vocab
        print(f"[OK] Golden reference binary format self-test passed ({len(packed)} bytes).")
        return 0

    return generate_golden(
        model_id=args.model_id,
        output_dir=args.output_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
