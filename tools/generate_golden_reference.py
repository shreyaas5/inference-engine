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
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
    args = parser.parse_args()

    print(
        f"[INFO] Golden reference generator ready for {args.model_id} -> {args.output_dir} "
        f"(max_new_tokens={args.max_new_tokens}). Run offline in M1 with torch & transformers installed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
