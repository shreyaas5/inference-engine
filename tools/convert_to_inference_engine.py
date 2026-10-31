#!/usr/bin/env python3
"""Offline converter from Hugging Face TinyLlama-1.1B-Chat-v1.0 (.safetensors) to `.ie` v1 binary format.

Owner: S (Project Lead & Model Loader)

Key Engineering Guarantees:
1. Exact adherence to `.ie` v1 FORMAT_SPECIFICATION.md (64-byte little-endian header, variable-length
   tensor directory, 64-byte aligned tensor payloads, exact byte-size & non-overlap invariants).
2. Preserves exact tensor ordering and dimensions from the source `.safetensors` checkpoint.
3. Validates the complete `2 + 9 * num_layers` Llama tensor inventory against `config.json` before writing.
4. Two-pass streaming conversion:
   - Pass 1 parses `.safetensors` headers (~24 KB), computes all 64-byte aligned offsets, and writes the
     `.ie` header and directory.
   - Pass 2 streams tensor payloads in bounded chunks (default 4 MiB), converting `BF16` or `F16` to
     little-endian IEEE 754 `FP32` without loading multiple full model copies into RAM.
"""

from __future__ import annotations

import array
import argparse
import json
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

IE_MAGIC = 0x464D4549
IE_VERSION = 1
IE_HEADER_STRUCT = struct.Struct("<12I2fQ")
IE_DIR_META_STRUCT = struct.Struct("<2I4Q2Q")
IE_ALIGNMENT = 64
MAX_TENSOR_DIMS = 4
MAX_TENSOR_NAME_LEN = 256

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_Q4_0 = 2
DTYPE_I32 = 3

SAFETENSORS_DTYPE_BYTES: Dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


@dataclass
class SourceTensorRecord:
    name: str
    sf_path: Path
    src_dtype: str
    shape: List[int]
    src_data_begin: int
    src_data_end: int
    dst_dtype: int
    dst_offset: int = 0
    dst_nbytes: int = 0

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


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


def pack_directory_entry(rec: SourceTensorRecord) -> bytes:
    name_bytes = rec.name.encode("utf-8")
    if len(name_bytes) == 0 or len(name_bytes) > MAX_TENSOR_NAME_LEN:
        raise ValueError(f"Invalid tensor name length for {rec.name!r}")
    ndim = len(rec.shape)
    if ndim == 0 or ndim > MAX_TENSOR_DIMS:
        raise ValueError(f"Invalid ndim {ndim} for {rec.name!r}")
    padded_shape = list(rec.shape) + [0] * (MAX_TENSOR_DIMS - ndim)
    meta = IE_DIR_META_STRUCT.pack(
        rec.dst_dtype,
        ndim,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        rec.dst_offset,
        rec.dst_nbytes,
    )
    return struct.pack("<I", len(name_bytes)) + name_bytes + meta


def bf16_bytes_to_fp32_bytes(raw_bf16: bytes) -> bytes:
    """Convert little-endian bfloat16 bytes to little-endian IEEE 754 float32 bytes.

    Each 16-bit bfloat16 value `u16` maps to exact IEEE 754 binary32 bits `u32 = u16 << 16`.
    """
    if len(raw_bf16) % 2 != 0:
        raise ValueError("BF16 byte buffer length must be even")
    if HAS_NUMPY:
        u16 = np.frombuffer(raw_bf16, dtype="<u2")
        u32 = u16.astype("<u4") << 16
        return u32.tobytes()

    u16_arr = array.array("H")
    u16_arr.frombytes(raw_bf16)
    if sys.byteorder != "little":
        u16_arr.byteswap()
    u32_arr = array.array("I", (int(v) << 16 for v in u16_arr))
    if sys.byteorder != "little":
        u32_arr.byteswap()
    return u32_arr.tobytes()


def f16_bytes_to_fp32_bytes(raw_f16: bytes) -> bytes:
    """Convert little-endian IEEE 754 binary16 bytes to little-endian binary32 bytes."""
    if len(raw_f16) % 2 != 0:
        raise ValueError("F16 byte buffer length must be even")
    if HAS_NUMPY:
        f16 = np.frombuffer(raw_f16, dtype="<f2")
        return f16.astype("<f4").tobytes()

    count = len(raw_f16) // 2
    vals = struct.unpack(f"<{count}e", raw_f16)
    return struct.pack(f"<{count}f", *vals)


def build_expected_inventory(cfg: Dict[str, Any]) -> Dict[str, List[int]]:
    num_layers = int(cfg["num_hidden_layers"])
    hidden_dim = int(cfg["hidden_size"])
    intermediate_dim = int(cfg["intermediate_size"])
    num_q_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = hidden_dim // num_q_heads
    vocab_size = int(cfg["vocab_size"])

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    expected: Dict[str, List[int]] = {
        "model.embed_tokens.weight": [vocab_size, hidden_dim],
        "model.norm.weight": [hidden_dim],
        "lm_head.weight": [vocab_size, hidden_dim],
    }
    for i in range(num_layers):
        p = f"model.layers.{i}"
        expected[f"{p}.input_layernorm.weight"] = [hidden_dim]
        expected[f"{p}.self_attn.q_proj.weight"] = [q_dim, hidden_dim]
        expected[f"{p}.self_attn.k_proj.weight"] = [kv_dim, hidden_dim]
        expected[f"{p}.self_attn.v_proj.weight"] = [kv_dim, hidden_dim]
        expected[f"{p}.self_attn.o_proj.weight"] = [hidden_dim, q_dim]
        expected[f"{p}.post_attention_layernorm.weight"] = [hidden_dim]
        expected[f"{p}.mlp.gate_proj.weight"] = [intermediate_dim, hidden_dim]
        expected[f"{p}.mlp.up_proj.weight"] = [intermediate_dim, hidden_dim]
        expected[f"{p}.mlp.down_proj.weight"] = [hidden_dim, intermediate_dim]
    return expected


def parse_safetensors_SourceRecords(
    safetensor_files: List[Path],
    expected_inventory: Dict[str, List[int]],
    strict_inventory: bool = True,
) -> List[SourceTensorRecord]:
    records: List[SourceTensorRecord] = []
    seen_names: set[str] = set()

    for sf_path in safetensor_files:
        file_size = sf_path.stat().st_size
        if file_size < 8:
            raise ValueError(f"Safetensors file {sf_path} is truncated (< 8 bytes)")

        with sf_path.open("rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            if hdr_len == 0 or 8 + hdr_len > file_size:
                raise ValueError(
                    f"Corrupted safetensors header length {hdr_len} in {sf_path.name}"
                )
            header = json.loads(f.read(hdr_len).decode("utf-8"))

        if not isinstance(header, dict):
            raise ValueError(f"Safetensors header in {sf_path.name} is not a JSON object")

        data_section_base = 8 + hdr_len
        file_ranges: List[Tuple[int, int]] = []

        # Preserve exact key ordering from the safetensors header.
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in seen_names:
                raise ValueError(f"Duplicate tensor name {name!r} in checkpoint")
            seen_names.add(name)

            src_dtype = meta.get("dtype")
            if src_dtype not in SAFETENSORS_DTYPE_BYTES:
                raise ValueError(f"Unsupported safetensors dtype {src_dtype!r} for {name}")

            shape = [int(d) for d in meta.get("shape", [])]
            if len(shape) == 0 or len(shape) > MAX_TENSOR_DIMS or any(d <= 0 for d in shape):
                raise ValueError(f"Invalid shape {shape} for tensor {name}")

            offsets = meta.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or int(offsets[0]) < 0
                or int(offsets[1]) < int(offsets[0])
            ):
                raise ValueError(f"Invalid data_offsets {offsets} for tensor {name}")

            rel_begin, rel_end = int(offsets[0]), int(offsets[1])
            abs_begin = data_section_base + rel_begin
            abs_end = data_section_base + rel_end
            if abs_end > file_size:
                raise ValueError(
                    f"Tensor {name} data_offsets [{rel_begin}, {rel_end}] exceed file size {file_size}"
                )

            num_elems = 1
            for d in shape:
                num_elems *= d
            expected_src_bytes = num_elems * SAFETENSORS_DTYPE_BYTES[src_dtype]
            if (rel_end - rel_begin) != expected_src_bytes:
                raise ValueError(
                    f"Tensor {name} byte length {rel_end - rel_begin} != expected {expected_src_bytes}"
                )

            file_ranges.append((rel_begin, rel_end))
            records.append(
                SourceTensorRecord(
                    name=name,
                    sf_path=sf_path,
                    src_dtype=src_dtype,
                    shape=shape,
                    src_data_begin=abs_begin,
                    src_data_end=abs_end,
                    dst_dtype=DTYPE_FP32,
                )
            )

        # Verify no overlapping tensor byte ranges inside the safetensors file.
        file_ranges.sort(key=lambda r: r[0])
        for idx in range(1, len(file_ranges)):
            if file_ranges[idx][0] < file_ranges[idx - 1][1]:
                raise ValueError(f"Overlapping tensor data_offsets detected in {sf_path.name}")

    if strict_inventory:
        actual_names = {r.name for r in records}
        expected_names = set(expected_inventory.keys())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"Checkpoint tensor inventory mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        for rec in records:
            exp_shape = expected_inventory[rec.name]
            if rec.shape != exp_shape:
                raise ValueError(
                    f"Tensor {rec.name} shape {rec.shape} does not match config expected shape {exp_shape}"
                )

    return records


def convert_safetensors_to_ie_fp32(
    input_dir: Path,
    output_path: Path,
    strict_inventory: bool = True,
    chunk_elements: int = 2 * 1024 * 1024,
) -> int:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    num_layers = int(cfg["num_hidden_layers"])
    hidden_dim = int(cfg["hidden_size"])
    intermediate_dim = int(cfg["intermediate_size"])
    num_q_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = hidden_dim // num_q_heads
    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg.get("max_position_embeddings", 2048))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(cfg.get("rope_theta", 10000.0))

    if num_q_heads * head_dim != hidden_dim or num_q_heads % num_kv_heads != 0:
        raise ValueError("Invalid attention head configuration in config.json")

    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors files found in {input_dir}")

    expected_inventory = build_expected_inventory(cfg)
    records = parse_safetensors_SourceRecords(
        safetensor_files=safetensor_files,
        expected_inventory=expected_inventory,
        strict_inventory=strict_inventory,
    )

    # Pass 1: Compute directory size, 64-byte aligned offsets, and total file size.
    dir_bytes = 0
    for rec in records:
        name_len = len(rec.name.encode("utf-8"))
        dir_bytes += 4 + name_len + 56

    payload_cursor = align_up(64 + dir_bytes, IE_ALIGNMENT) if records else 64
    for rec in records:
        payload_cursor = align_up(payload_cursor, IE_ALIGNMENT)
        rec.dst_offset = payload_cursor
        rec.dst_nbytes = rec.num_elements * 4
        payload_cursor += rec.dst_nbytes

    total_file_bytes = payload_cursor
    header_bytes = pack_header(
        tensor_count=len(records),
        weight_dtype=DTYPE_FP32,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        total_file_bytes=total_file_bytes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 2: Stream header, directory, and chunked FP32 tensor payloads to disk.
    with output_path.open("wb") as out_f:
        out_f.write(header_bytes)
        for rec in records:
            out_f.write(pack_directory_entry(rec))

        for rec in records:
            cur_pos = out_f.tell()
            if cur_pos > rec.dst_offset:
                raise RuntimeError(f"Output cursor {cur_pos} exceeded dst_offset {rec.dst_offset}")
            if cur_pos < rec.dst_offset:
                out_f.write(b"\x00" * (rec.dst_offset - cur_pos))

            elem_bytes = SAFETENSORS_DTYPE_BYTES[rec.src_dtype]
            chunk_src_bytes = chunk_elements * elem_bytes
            remaining = rec.src_data_end - rec.src_data_begin
            written_payload = 0

            with rec.sf_path.open("rb") as src_f:
                src_f.seek(rec.src_data_begin)
                while remaining > 0:
                    to_read = min(remaining, chunk_src_bytes)
                    raw_chunk = src_f.read(to_read)
                    if len(raw_chunk) != to_read:
                        raise IOError(f"Unexpected EOF while reading {rec.name} from {rec.sf_path}")
                    remaining -= to_read

                    if rec.src_dtype == "BF16":
                        fp32_chunk = bf16_bytes_to_fp32_bytes(raw_chunk)
                    elif rec.src_dtype == "F16":
                        fp32_chunk = f16_bytes_to_fp32_bytes(raw_chunk)
                    elif rec.src_dtype == "F32":
                        fp32_chunk = raw_chunk
                    else:
                        raise ValueError(f"Unsupported source dtype {rec.src_dtype}")

                    out_f.write(fp32_chunk)
                    written_payload += len(fp32_chunk)

            if written_payload != rec.dst_nbytes:
                raise RuntimeError(
                    f"Written byte count {written_payload} != dst_nbytes {rec.dst_nbytes} for {rec.name}"
                )

        final_pos = out_f.tell()
        if final_pos != total_file_bytes:
            raise RuntimeError(
                f"Final .ie file position {final_pos} != expected total_file_bytes {total_file_bytes}"
            )

    return total_file_bytes


def float_to_bf16_u16(val: float) -> int:
    """Pack a Python float into IEEE 754 bfloat16 upper-16 bits for self-test fixture generation."""
    bits = struct.unpack("<I", struct.pack("<f", val))[0]
    return (bits >> 16) & 0xFFFF


def run_self_test() -> int:
    hdr = pack_header(
        tensor_count=0,
        weight_dtype=DTYPE_FP32,
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
    assert len(hdr) == 64

    # End-to-end synthetic 2-layer Llama safetensors -> .ie v1 conversion & reload verification.
    cfg = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    expected_inv = build_expected_inventory(cfg)
    assert len(expected_inv) == 21  # 3 + 9 * 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

        st_header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
        payload_chunks: List[bytes] = []
        expected_fp32_by_name: Dict[str, List[float]] = {}
        cursor = 0

        for idx, (name, shape) in enumerate(expected_inv.items()):
            num_elems = 1
            for d in shape:
                num_elems *= d
            u16_vals: List[int] = []
            fp32_vals: List[float] = []
            for e in range(num_elems):
                sample_f = ((idx + 1) * 0.125) * (1.0 if (e % 2 == 0) else -0.5) + (e % 7) * 0.0625
                u16 = float_to_bf16_u16(sample_f)
                exact_f32 = struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
                u16_vals.append(u16)
                fp32_vals.append(exact_f32)

            raw_bytes = struct.pack(f"<{num_elems}H", *u16_vals)
            st_header[name] = {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [cursor, cursor + len(raw_bytes)],
            }
            payload_chunks.append(raw_bytes)
            expected_fp32_by_name[name] = fp32_vals
            cursor += len(raw_bytes)

        hdr_json = json.dumps(st_header).encode("utf-8")
        sf_path = tmp_dir / "model.safetensors"
        with sf_path.open("wb") as f:
            f.write(struct.pack("<Q", len(hdr_json)))
            f.write(hdr_json)
            for chunk in payload_chunks:
                f.write(chunk)

        ie_out = tmp_dir / "synthetic_model.ie"
        total_bytes = convert_safetensors_to_ie_fp32(tmp_dir, ie_out, strict_inventory=True, chunk_elements=256)
        assert ie_out.exists() and ie_out.stat().st_size == total_bytes

        # Reload and verify every tensor's alignment, shape, and exact FP32 values.
        with ie_out.open("rb") as f:
            unpacked_hdr = IE_HEADER_STRUCT.unpack(f.read(64))
            assert unpacked_hdr[0] == IE_MAGIC
            assert unpacked_hdr[1] == IE_VERSION
            assert unpacked_hdr[2] == 21
            assert unpacked_hdr[14] == total_bytes

            loaded_records = []
            for _ in range(21):
                nlen = struct.unpack("<I", f.read(4))[0]
                tname = f.read(nlen).decode("utf-8")
                meta = IE_DIR_META_STRUCT.unpack(f.read(56))
                ndim = meta[1]
                tshape = list(meta[2:6])[:ndim]
                toffset, tnbytes = meta[6], meta[7]
                assert toffset % IE_ALIGNMENT == 0
                loaded_records.append((tname, tshape, toffset, tnbytes))

            assert [r[0] for r in loaded_records] == list(expected_inv.keys())

            for tname, tshape, toffset, tnbytes in loaded_records:
                assert tshape == expected_inv[tname]
                f.seek(toffset)
                raw_fp32 = f.read(tnbytes)
                count = tnbytes // 4
                actual_floats = list(struct.unpack(f"<{count}f", raw_fp32))
                assert actual_floats == expected_fp32_by_name[tname]

        # Negative test: missing required tensor in safetensors must raise ValueError.
        incomplete_hdr = dict(st_header)
        incomplete_hdr.pop("model.norm.weight")
        inc_json = json.dumps(incomplete_hdr).encode("utf-8")
        with (tmp_dir / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(inc_json)))
            f.write(inc_json)
            for chunk in payload_chunks:
                f.write(chunk)
        try:
            convert_safetensors_to_ie_fp32(tmp_dir, tmp_dir / "bad.ie", strict_inventory=True)
            raise AssertionError("Expected missing tensor in safetensors to raise ValueError")
        except ValueError:
            pass

    print(
        f"[OK] Packed .ie v1 header (64 bytes, magic=0x{IE_MAGIC:08X}) and verified "
        f"streaming BF16->FP32 safetensors conversion across 21 synthetic Llama tensors."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TinyLlama-1.1B-Chat-v1.0 (.safetensors) checkpoint into `.ie` v1 format."
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
        help="Run end-to-end header packing and synthetic BF16 safetensors -> .ie FP32 conversion self-test",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.input_dir is None or args.output is None:
        parser.print_help()
        return 0

    if args.dtype != "fp32":
        print("[ERROR] Q4_0 conversion is scheduled for Milestone M2. Use --dtype fp32 for M1.", file=sys.stderr)
        return 2

    total_bytes = convert_safetensors_to_ie_fp32(args.input_dir, args.output, strict_inventory=True)
    print(f"[OK] Converted {args.input_dir} -> {args.output} ({total_bytes:,} bytes, dtype=FP32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
