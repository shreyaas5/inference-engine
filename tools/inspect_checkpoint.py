#!/usr/bin/env python3
"""Inspect a Hugging Face TinyLlama-1.1B-Chat-v1.0 checkpoint or local .ie file.

Owner: S (Project Lead & Model Loader)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

FROZEN_TINYLLAMA_CONFIG: Dict[str, Any] = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_hidden_layers": 22,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


def inspect_local_config(checkpoint_dir: Path) -> int:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        print(f"[ERROR] config.json not found in {checkpoint_dir}")
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=== Checkpoint Hyperparameter Verification ===")
    mismatches = 0
    for key, expected in FROZEN_TINYLLAMA_CONFIG.items():
        if key in ("model_id", "head_dim"):
            continue
        actual = cfg.get(key)
        status = "OK" if actual == expected else "MISMATCH"
        if status != "OK":
            mismatches += 1
        print(f"  {key:28s}: actual={actual!r} expected={expected!r} [{status}]")

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensors file(s).")
    return 0 if mismatches == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect TinyLlama-1.1B-Chat-v1.0 checkpoint metadata and tensor inventory."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Path to local Hugging Face checkpoint directory containing config.json and *.safetensors",
    )
    parser.add_argument(
        "--print-frozen-contract",
        action="store_true",
        help="Print the frozen TinyLlama-1.1B hyperparameter contract as JSON",
    )
    args = parser.parse_args()

    if args.print_frozen_contract:
        print(json.dumps(FROZEN_TINYLLAMA_CONFIG, indent=2))
        return 0

    if args.checkpoint_dir is None:
        parser.print_help()
        return 0

    return inspect_local_config(args.checkpoint_dir)


if __name__ == "__main__":
    raise SystemExit(main())
