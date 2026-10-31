#!/usr/bin/env python3
"""Verify TinyLlama SentencePiece token IDs and chat formatting against Hugging Face."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def format_tinyllama_chat(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    parts: List[str] = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role in {"system", "user", "assistant"}:
            parts.append(f"<|{role}|>\n{message['content']}</s>\n")
        if index + 1 == len(messages) and add_generation_prompt:
            parts.append("<|assistant|>\n")
    return "".join(parts)


def run_engine(engine: Path, tokenizer_model: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(engine), "--tokenizer", str(tokenizer_model), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def parse_token_ids(output: str) -> List[int]:
    prefix = "Encoded token IDs:"
    if not output.startswith(prefix):
        raise AssertionError(f"Unexpected --encode-only output: {output!r}")
    values = output[len(prefix) :].strip()
    return [int(value) for value in values.split()] if values else []


def self_test() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    expected = (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        "<|user|>\nHello!</s>\n"
        "<|assistant|>\n"
    )
    assert format_tinyllama_chat(messages) == expected
    assert format_tinyllama_chat([], add_generation_prompt=True) == ""
    print("[OK] TinyLlama tokenizer_config.json chat-template contract self-test passed.")


def verify_huggingface(args: argparse.Namespace) -> None:
    try:
        from transformers import LlamaTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The parity check requires transformers and sentencepiece; "
            "install the project's tokenizer verification dependencies."
        ) from error

    tokenizer_model = Path(args.tokenizer_model).resolve()
    tokenizer_config_path = Path(args.tokenizer_config).resolve()
    engine = Path(args.engine).resolve()
    config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    if not config.get("chat_template"):
        raise AssertionError("TinyLlama tokenizer_config.json has no chat_template")

    tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_model),
        unk_token=config["unk_token"],
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        pad_token=config["pad_token"],
        legacy=config.get("legacy", False),
        add_bos_token=False,
        add_eos_token=False,
        chat_template=config["chat_template"],
    )
    if (tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 1, 2):
        raise AssertionError(
            "TinyLlama special-token IDs changed: "
            f"unk={tokenizer.unk_token_id}, bos={tokenizer.bos_token_id}, "
            f"eos={tokenizer.eos_token_id}"
        )
    if tokenizer.vocab_size != 32000:
        raise AssertionError(f"Expected vocabulary size 32000, got {tokenizer.vocab_size}")

    added_tokens = {
        entry["content"]
        for entry in config.get("added_tokens_decoder", {}).values()
    }
    if any(token in added_tokens for token in ("<|system|>", "<|user|>", "<|assistant|>")):
        raise AssertionError("Zephyr role markers must not be assumed to be special tokens")

    test_strings = [
        "Hello, world!",
        "  leading and trailing  ",
        "multiple   spaces\nand\ttabs",
        "Numbers: 0 42 3.14159",
        "Punctuation: !?.,;:()[]{}",
        "Unicode: café — 東京 🌍",
        "<s>text</s><unk>",
    ]
    for text in test_strings:
        expected_ids = tokenizer.encode(text, add_special_tokens=False)
        actual_ids = parse_token_ids(
            run_engine(engine, tokenizer_model, "--encode-only", "--prompt", text)
        )
        if actual_ids != expected_ids:
            raise AssertionError(
                f"Token ID mismatch for {text!r}:\n"
                f"  Hugging Face: {expected_ids}\n"
                f"  C++ engine:   {actual_ids}"
            )

        decoded = tokenizer.decode(
            expected_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        engine_decoded = run_engine(
            engine,
            tokenizer_model,
            "--decode-only",
            "--token-ids",
            " ".join(str(token_id) for token_id in expected_ids),
        )
        if engine_decoded != decoded:
            raise AssertionError(
                f"Decode mismatch for {text!r}:\n"
                f"  Hugging Face: {decoded!r}\n"
                f"  C++ engine:   {engine_decoded!r}"
            )

        wrapped_ids = parse_token_ids(
            run_engine(
                engine,
                tokenizer_model,
                "--encode-only",
                "--add-bos",
                "--add-eos",
                "--prompt",
                text,
            )
        )
        if wrapped_ids != [tokenizer.bos_token_id, *expected_ids, tokenizer.eos_token_id]:
            raise AssertionError(f"BOS/EOS placement mismatch for {text!r}")

    chat_messages = [{"role": "user", "content": "Hello!"}]
    expected_chat = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    actual_chat = run_engine(
        engine,
        tokenizer_model,
        "--format-chat",
        "--prompt",
        chat_messages[0]["content"],
    )
    if actual_chat != expected_chat:
        raise AssertionError(
            f"Chat-template mismatch:\n  Hugging Face: {expected_chat!r}\n"
            f"  C++ engine:   {actual_chat!r}"
        )

    print(
        f"[OK] Hugging Face parity passed for {len(test_strings)} text cases, "
        "BOS/EOS handling, decode, and the chat template."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify TinyLlama tokenizer parity between Hugging Face and C++."
    )
    parser.add_argument("--self-test", action="store_true", help="Run offline template checks")
    parser.add_argument("--engine", help="Path to the built inference-engine executable")
    parser.add_argument("--tokenizer-model", help="Path to TinyLlama tokenizer.model")
    parser.add_argument("--tokenizer-config", help="Path to TinyLlama tokenizer_config.json")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            return 0
        if not args.engine or not args.tokenizer_model or not args.tokenizer_config:
            parser.error("--engine, --tokenizer-model, and --tokenizer-config are required")
        verify_huggingface(args)
    except (AssertionError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
