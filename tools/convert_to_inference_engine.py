#!/usr/bin/env python3
"""Offline converter from Hugging Face TinyLlama-1.1B-Chat-v1.0 to `.ie` v1 binary format.

Owner: S (Project Lead & Model Loader)

Binary Format Contract (.ie v1):
- Little-endian serialization throughout (never dump native C/C++ structs).
- 64-byte header:
  - magic: u32 = 0x464D4549 ("IEMF")
  - version: u32 = 1
  - tensor_count: u32
  - weight_dtype: u32 (0 = FP32, 1 = FP16, 2 = Q4_0, 3 = I32)
  - num_layers: u32 (22)
  - hidden_dim: u32 (2048)
  - intermediate_dim: u32 (5632)
  - num_query_heads: u32 (32)
  - num_kv_heads: u32 (4)
  - head_dim: u32 (64)
  - vocab_size: u32 (32000)
  - max_seq_len: u32 (2048)
  - rms_norm_eps: f32 (1e-5)
  - rope_theta: f32 (10000.0)
  - total_file_bytes: u64
- Tensor directory entries (one per tensor):
  - name_len: u32
  - name: UTF-8 bytes (name_len)
  - dtype: u32
  - ndim: u32 (1..4)
  - shape: 4 x u64 (unused trailing dimensions set to 0)
  - payload_offset: u64 (must be 64-byte aligned)
  - payload_nbytes: u64 (must match exact expected byte size for shape & dtype)
- Tensor payloads:
  - Each payload starts at a 64-byte aligned offset (`payload_offset % 64 == 0`).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

IE_MAGIC = 0x464D4549
IE_VERSION = 1
IE_HEADER_STRUCT = struct.Struct("<12I2fQ")
IE_ALIGNMENT = 64


def align_up(value: int, alignment: int = IE_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def pack_header(
    tensor_count: int,
    weight_dtype: int,
    num_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int,
    rms_norm_eps: float,
    rope_theta: float,
    total_file_bytes: int,
) -> bytes:
    header = IE_HEADER_STRUCT.pack(
        IE_MAGIC,
        IE_VERSION,
        tensor_count,
        weight_dtype,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rms_norm_eps,
        rope_theta,
        total_file_bytes,
    )
    assert len(header) == 64
    return header


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TinyLlama-1.1B-Chat-v1.0 checkpoint into `.ie` v1 format."
    )
    parser.add_argument("--input-dir", type=Path, help="Input Hugging Face checkpoint directory")
    parser.add_argument("--output", type=Path, help="Output .ie model path")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "q4_0"],
        default="fp32",
        help="Target linear weight precision (default: fp32)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify 64-byte header packing contract without external model weights",
    )
    args = parser.parse_args()

    if args.self_test:
        hdr = pack_header(
            tensor_count=0,
            weight_dtype=0,
            num_layers=22,
            hidden_dim=2048,
            intermediate_dim=5632,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=64,
            vocab_size=32000,
            max_seq_len=2048,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            total_file_bytes=64,
        )
        print(f"[OK] Packed .ie v1 header ({len(hdr)} bytes), magic=0x{IE_MAGIC:08X}")
        return 0

    if args.input_dir is None or args.output is None:
        parser.print_help()
        return 0

    print(
        f"[INFO] M0 scaffolding: full safetensors -> .ie ({args.dtype}) conversion is scheduled for M1/M2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
