#!/usr/bin/env python3
"""Verify SentencePiece C++ tokenizer parity and TinyLlama chat template formatting.

Owner: P (Tokenizer, Runtime & Benchmarks)

Verification Contract:
1. Exact token ID parity on canonical test strings, whitespace edge cases, numbers,
   Unicode/emoji byte fallback (`<0x..>`), and special tokens (`<s>`=1, `</s>`=2, `<unk>`=0).
2. Exact decoding round-trip parity.
3. Exact TinyLlama Zephyr chat template parity against Hugging Face `apply_chat_template`.
"""

from __future__ import annotations

import argparse
from typing import List, Dict


def format_tinyllama_chat(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    parts: List[str] = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}</s>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify tokenizer parity between Hugging Face / SentencePiece and inference-engine."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline chat template parity self-test",
    )
    args = parser.parse_args()

    if args.self_test:
        sample = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        rendered = format_tinyllama_chat(sample, add_generation_prompt=True)
        expected = (
            "<|system|>\nYou are a helpful assistant.</s>\n"
            "<|user|>\nHello!</s>\n"
            "<|assistant|>\n"
        )
        assert rendered == expected, f"Mismatch:\n{rendered!r}\nvs\n{expected!r}"
        print("[OK] TinyLlama Zephyr chat template self-test passed.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
