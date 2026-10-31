#!/usr/bin/env python3
"""Inspect a Hugging Face TinyLlama-1.1B-Chat-v1.0 checkpoint (.safetensors) or converted .ie v1 file.

Owner: S (Project Lead & Model Loader)
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "tie_word_embeddings": False,
}

SAFETENSORS_DTYPE_BYTES: Dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
}

IE_MAGIC = 0x464D4549
IE_VERSION = 1
IE_HEADER_STRUCT = struct.Struct("<12I2fQ")
IE_DTYPE_NAMES = {0: "FP32", 1: "FP16", 2: "Q4_0", 3: "I32"}


def build_expected_llama_inventory(
    num_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
) -> Dict[str, List[int]]:
    q_dim = num_query_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    expected: Dict[str, List[int]] = {
        "model.embed_tokens.weight": [vocab_size, hidden_dim],
        "model.norm.weight": [hidden_dim],
        "lm_head.weight": [vocab_size, hidden_dim],
    }
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        expected[f"{prefix}.input_layernorm.weight"] = [hidden_dim]
        expected[f"{prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_dim]
        expected[f"{prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_dim]
        expected[f"{prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_dim]
        expected[f"{prefix}.self_attn.o_proj.weight"] = [hidden_dim, q_dim]
        expected[f"{prefix}.post_attention_layernorm.weight"] = [hidden_dim]
        expected[f"{prefix}.mlp.gate_proj.weight"] = [intermediate_dim, hidden_dim]
        expected[f"{prefix}.mlp.up_proj.weight"] = [intermediate_dim, hidden_dim]
        expected[f"{prefix}.mlp.down_proj.weight"] = [hidden_dim, intermediate_dim]
    return expected


def read_safetensors_header(sf_path: Path) -> Tuple[int, Dict[str, Any], int]:
    file_size = sf_path.stat().st_size
    if file_size < 8:
        raise ValueError(f"Safetensors file {sf_path} is smaller than 8 bytes ({file_size})")

    with sf_path.open("rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        if hdr_len == 0 or 8 + hdr_len > file_size:
            raise ValueError(
                f"Invalid safetensors header length {hdr_len} for file size {file_size}"
            )
        raw_json = f.read(hdr_len)

    header = json.loads(raw_json.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header JSON root must be an object")
    return hdr_len, header, file_size


def inspect_local_checkpoint(
    checkpoint_dir: Path, verbose: bool = False, allow_scaled_config: bool = False
) -> int:
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

    num_layers = int(cfg["num_hidden_layers"])
    hidden_dim = int(cfg["hidden_size"])
    intermediate_dim = int(cfg["intermediate_size"])
    num_q_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = hidden_dim // num_q_heads
    vocab_size = int(cfg["vocab_size"])

    expected_inventory = build_expected_llama_inventory(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
    )

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensors file(s).")
    if not safetensor_files:
        return 0 if mismatches == 0 else 2

    discovered_tensors: Dict[str, Dict[str, Any]] = {}
    total_payload_bytes = 0

    for sf_path in safetensor_files:
        hdr_len, header, file_size = read_safetensors_header(sf_path)
        data_base = 8 + hdr_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in discovered_tensors:
                print(f"[ERROR] Duplicate tensor {name} across safetensors files")
                return 3
            dtype = meta["dtype"]
            shape = list(meta["shape"])
            offsets = meta["data_offsets"]
            nbytes = offsets[1] - offsets[0]
            elem_count = 1
            for d in shape:
                elem_count *= d
            expected_nbytes = elem_count * SAFETENSORS_DTYPE_BYTES[dtype]
            if nbytes != expected_nbytes or data_base + offsets[1] > file_size:
                print(f"[ERROR] Invalid byte range for tensor {name} in {sf_path.name}")
                return 3
            discovered_tensors[name] = {
                "dtype": dtype,
                "shape": shape,
                "offsets": offsets,
                "nbytes": nbytes,
                "file": sf_path.name,
            }
            total_payload_bytes += nbytes
            if verbose:
                print(f"  [{sf_path.name}] {name:48s} dtype={dtype:4s} shape={shape} nbytes={nbytes}")

    missing = set(expected_inventory.keys()) - set(discovered_tensors.keys())
    extra = set(discovered_tensors.keys()) - set(expected_inventory.keys())
    shape_errors = 0
    for name, exp_shape in expected_inventory.items():
        if name in discovered_tensors and discovered_tensors[name]["shape"] != exp_shape:
            print(
                f"[ERROR] Shape mismatch for {name}: actual={discovered_tensors[name]['shape']} expected={exp_shape}"
            )
            shape_errors += 1

    print(
        f"=== Safetensors Inventory Summary: {len(discovered_tensors)}/{len(expected_inventory)} tensors, "
        f"{total_payload_bytes:,} payload bytes ==="
    )
    if missing or extra or shape_errors > 0:
        print(f"[ERROR] Inventory mismatch: missing={len(missing)}, extra={len(extra)}, shape_errors={shape_errors}")
        return 3

    print("[OK] All tensor names, shapes, and byte offsets match ModelConfig.")
    return 0 if (mismatches == 0 or allow_scaled_config) else 2


def inspect_ie_file(ie_path: Path, verbose: bool = False) -> int:
    file_size = ie_path.stat().st_size
    if file_size < 64:
        print(f"[ERROR] .ie file {ie_path} is smaller than 64-byte header")
        return 1

    with ie_path.open("rb") as f:
        hdr_bytes = f.read(64)
        (
            magic,
            version,
            tensor_count,
            weight_dtype,
            num_layers,
            hidden_dim,
            intermediate_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            total_file_bytes,
        ) = IE_HEADER_STRUCT.unpack(hdr_bytes)

        if magic != IE_MAGIC or version != IE_VERSION or total_file_bytes != file_size:
            print(
                f"[ERROR] Invalid .ie header: magic=0x{magic:08X}, version={version}, "
                f"header_bytes={total_file_bytes}, actual_size={file_size}"
            )
            return 2

        print(
            f"=== .ie v1 File Header ({ie_path.name}) ===\n"
            f"  tensors={tensor_count}, weight_dtype={IE_DTYPE_NAMES.get(weight_dtype, weight_dtype)}, "
            f"layers={num_layers}, hidden={hidden_dim}, inter={intermediate_dim}, "
            f"heads={num_q_heads}/{num_kv_heads} (dim={head_dim}), vocab={vocab_size}, "
            f"max_seq={max_seq_len}, total_bytes={total_file_bytes:,}"
        )

        dir_entry_struct = struct.Struct("<2I4Q2Q")
        for idx in range(tensor_count):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            unpacked = dir_entry_struct.unpack(f.read(56))
            dtype_raw, ndim = unpacked[0], unpacked[1]
            shape = list(unpacked[2:6])[:ndim]
            offset, nbytes = unpacked[6], unpacked[7]
            if offset % 64 != 0:
                print(f"[ERROR] Tensor {name} payload_offset {offset} is not 64-byte aligned")
                return 3
            if verbose:
                print(
                    f"  [{idx:3d}] {name:48s} dtype={IE_DTYPE_NAMES.get(dtype_raw, dtype_raw):4s} "
                    f"shape={shape} offset={offset} nbytes={nbytes}"
                )

    print(f"[OK] Verified .ie v1 structure and 64-byte alignment across {tensor_count} tensors.")
    return 0


def inspect_remote_hf_header(
    repo_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", verbose: bool = False
) -> int:
    base = f"https://huggingface.co/{repo_id}/resolve/main"
    with urllib.request.urlopen(f"{base}/config.json", timeout=15) as r:
        cfg = json.loads(r.read().decode("utf-8"))

    print(f"=== Remote Checkpoint Hyperparameter Verification ({repo_id}) ===")
    mismatches = 0
    for key, expected in FROZEN_TINYLLAMA_CONFIG.items():
        if key in ("model_id", "head_dim"):
            continue
        actual = cfg.get(key)
        status = "OK" if actual == expected else "MISMATCH"
        if status != "OK":
            mismatches += 1
        print(f"  {key:28s}: actual={actual!r} expected={expected!r} [{status}]")

    req = urllib.request.Request(f"{base}/model.safetensors", headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=15) as r:
        hdr_len = struct.unpack("<Q", r.read(8))[0]

    req2 = urllib.request.Request(
        f"{base}/model.safetensors", headers={"Range": f"bytes=8-{8 + hdr_len - 1}"}
    )
    with urllib.request.urlopen(req2, timeout=15) as r:
        header = json.loads(r.read(hdr_len).decode("utf-8"))

    expected_inventory = build_expected_llama_inventory(
        num_layers=int(cfg["num_hidden_layers"]),
        hidden_dim=int(cfg["hidden_size"]),
        intermediate_dim=int(cfg["intermediate_size"]),
        num_query_heads=int(cfg["num_attention_heads"]),
        num_kv_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]),
        vocab_size=int(cfg["vocab_size"]),
    )

    tensors = [(k, v) for k, v in header.items() if k != "__metadata__"]
    total_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for _, v in tensors)
    print(
        f"=== Remote model.safetensors Header ({hdr_len:,} bytes): "
        f"{len(tensors)}/{len(expected_inventory)} tensors, {total_bytes:,} payload bytes ==="
    )

    for idx, (name, meta) in enumerate(tensors):
        if name not in expected_inventory or meta["shape"] != expected_inventory[name]:
            print(f"[ERROR] Unexpected tensor or shape: {name} {meta}")
            return 3
        if verbose:
            print(
                f"  [{idx:3d}] {name:48s} dtype={meta['dtype']:4s} "
                f"shape={meta['shape']} offsets={meta['data_offsets']}"
            )

    if len(tensors) != len(expected_inventory):
        print("[ERROR] Tensor count mismatch against expected inventory")
        return 3

    print("[OK] All 201 remote TinyLlama-1.1B-Chat-v1.0 tensors verified against ModelConfig.")
    return 0 if mismatches == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect TinyLlama-1.1B-Chat-v1.0 checkpoint metadata and .ie v1 model files."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Path to local Hugging Face checkpoint directory containing config.json and *.safetensors",
    )
    parser.add_argument(
        "--ie-file",
        type=Path,
        help="Path to a converted .ie v1 binary model file to inspect and validate",
    )
    parser.add_argument(
        "--inspect-remote-hf",
        action="store_true",
        help="Fetch and verify the live TinyLlama-1.1B-Chat-v1.0 config.json and safetensors header via HTTP Range",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every tensor descriptor during inspection",
    )
    parser.add_argument(
        "--allow-scaled-config",
        action="store_true",
        help="Allow scaled-down synthetic ModelConfig dimensions when inspecting test checkpoints",
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

    if args.inspect_remote_hf:
        return inspect_remote_hf_header(verbose=args.verbose)

    if args.ie_file is not None:
        return inspect_ie_file(args.ie_file, verbose=args.verbose)

    if args.checkpoint_dir is not None:
        return inspect_local_checkpoint(
            args.checkpoint_dir,
            verbose=args.verbose,
            allow_scaled_config=args.allow_scaled_config,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
