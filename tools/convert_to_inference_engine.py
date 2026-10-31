#!/usr/bin/env python3
"""Offline converter from Hugging Face TinyLlama-1.1B-Chat-v1.0 (.safetensors) to `.ie` v1 binary format.

Owner: S (Project Lead & Model Loader)

Key Engineering Guarantees:
1. Exact adherence to `.ie` v1 FORMAT_SPECIFICATION.md (64-byte little-endian header, variable-length
   tensor directory, 64-byte aligned tensor payloads, exact byte-size & non-overlap invariants).
2. Preserves exact tensor ordering and dimensions from the source `.safetensors` checkpoint.
3. Validates the complete `3 + 9 * num_layers` Llama tensor inventory against `config.json` before writing.
4. Supports both `--dtype fp32` and `--dtype q4_0` (with explicit mixed-precision policy):
   - In `fp32` mode (`header.weight_dtype = FP32`), all 201 tensors are stored as IEEE 754 `FP32`.
   - In `q4_0` mode (`header.weight_dtype = Q4_0`), 1D RMSNorm scale tensors (`*.input_layernorm.weight`,
     `*.post_attention_layernorm.weight`, `model.norm.weight`) are preserved in `FP32`, while 2D weight
     matrices are quantised to 32-element `BlockQ4_0` blocks (18 bytes/block: 2-byte IEEE 754 FP16 scale `d`
     + 16 packed bytes `qs[0..15]` using split-half low/high nibbles matching `src/kernels/quant_q4_0.cpp`
     and upstream GGML `Q4_0` bit-for-bit).
   - Optional `--keep-embeddings-fp32` keeps `model.embed_tokens.weight` and `lm_head.weight` in `FP32`
     while quantising all 154 decoder layer 2D projection matrices to `Q4_0`.
5. Two-pass streaming conversion:
   - Pass 1 parses `.safetensors` headers (~24 KB), validates shapes and `Q4_0` 32-element block divisibility,
     computes all 64-byte aligned offsets, and writes the `.ie` header and directory.
   - Pass 2 streams tensor payloads in bounded chunks that are exact multiples of 32 elements, converting
     `BF16`/`F16`/`F32` to `FP32` or `Q4_0` without loading multiple full model copies into RAM.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_PACKED_BYTES = 16
Q4_0_BLOCK_BYTES = 18

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


@dataclass
class QuantErrorStats:
    num_elements: int = 0
    max_abs_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0
    sum_sq_signal: float = 0.0

    @property
    def mean_abs_error(self) -> float:
        return self.sum_abs_error / self.num_elements if self.num_elements > 0 else 0.0

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sum_sq_error / self.num_elements) if self.num_elements > 0 else 0.0

    @property
    def relative_l2_error(self) -> float:
        if self.sum_sq_signal <= 0.0:
            return 0.0
        return math.sqrt(self.sum_sq_error / self.sum_sq_signal)

    def update(self, other: "QuantErrorStats") -> None:
        self.num_elements += other.num_elements
        if other.max_abs_error > self.max_abs_error:
            self.max_abs_error = other.max_abs_error
        self.sum_abs_error += other.sum_abs_error
        self.sum_sq_error += other.sum_sq_error
        self.sum_sq_signal += other.sum_sq_signal


def align_up(value: int, alignment: int = IE_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def f32_round(val: float) -> float:
    """Round a Python float to exact single-precision IEEE 754 binary32."""
    return struct.unpack("<f", struct.pack("<f", val))[0]


def fp32_bits_to_fp16_u16_scalar(fbits: int) -> int:
    """Bit-exact replica of `inference_engine::fp32_to_fp16` in `src/kernels/quant_q4_0.cpp`."""
    sign = (fbits >> 16) & 0x8000
    exp = ((fbits >> 23) & 0xFF) - 127 + 15
    mantissa = fbits & 0x007FFFFF

    if exp <= 0:
        if exp < -10:
            return sign
        mantissa = (mantissa | 0x00800000) >> (1 - exp)
        if mantissa & 0x00001000:
            mantissa += 0x00002000
        return (sign | (mantissa >> 13)) & 0xFFFF

    if exp == (0xFF - 127 + 15):
        if mantissa == 0:
            return (sign | 0x7C00) & 0xFFFF
        mantissa >>= 13
        return (sign | 0x7C00 | mantissa | (1 if mantissa == 0 else 0)) & 0xFFFF

    if mantissa & 0x00001000:
        mantissa += 0x00002000
        if mantissa & 0x00800000:
            mantissa = 0
            if exp + 1 >= 31:
                return (sign | 0x7C00) & 0xFFFF
            return (sign | ((exp + 1) << 10)) & 0xFFFF

    if exp >= 31:
        return (sign | 0x7C00) & 0xFFFF

    return (sign | (exp << 10) | (mantissa >> 13)) & 0xFFFF


def fp32_to_fp16_u16_scalar(val: float) -> int:
    fbits = struct.unpack("<I", struct.pack("<f", val))[0]
    return fp32_bits_to_fp16_u16_scalar(fbits)


def fp16_u16_to_fp32_scalar(bits: int) -> float:
    """Bit-exact replica of `inference_engine::fp16_to_fp32` in `src/kernels/quant_q4_0.cpp`."""
    sign = (bits & 0x8000) << 16
    exp = (bits >> 10) & 0x1F
    mantissa = bits & 0x03FF

    if exp == 0:
        if mantissa == 0:
            out_bits = sign
        else:
            e = 127 - 14
            while (mantissa & 0x0400) == 0:
                mantissa <<= 1
                e -= 1
            mantissa &= 0x03FF
            out_bits = sign | (e << 23) | (mantissa << 13)
    elif exp == 31:
        out_bits = sign | 0x7F800000 | (mantissa << 13)
    else:
        out_bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13)

    return struct.unpack("<f", struct.pack("<I", out_bits & 0xFFFFFFFF))[0]


def fp32_to_fp16_u16_numpy(f32_arr: "np.ndarray") -> "np.ndarray":
    """Vectorized bit-exact replica of `inference_engine::fp32_to_fp16` over a `<f4` NumPy array."""
    fbits = f32_arr.astype("<f4", copy=False).view("<u4")
    sign = (fbits >> np.uint32(16)) & np.uint32(0x8000)
    exp = ((fbits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127 + 15
    mantissa = fbits & np.uint32(0x007FFFFF)

    out = np.zeros_like(fbits, dtype="<u2")

    # Normal range: 0 < exp < 31
    norm_mask = (exp > 0) & (exp < 31)
    if np.any(norm_mask):
        m_norm = mantissa[norm_mask].copy()
        e_norm = exp[norm_mask].copy()
        s_norm = sign[norm_mask]
        round_up = (m_norm & np.uint32(0x00001000)) != 0
        m_norm = np.where(round_up, m_norm + np.uint32(0x00002000), m_norm)
        carry = (m_norm & np.uint32(0x00800000)) != 0
        m_norm = np.where(carry, np.uint32(0), m_norm)
        e_norm = np.where(carry, e_norm + 1, e_norm)
        res_norm = np.where(
            e_norm >= 31,
            s_norm | np.uint32(0x7C00),
            s_norm | (e_norm.astype(np.uint32) << np.uint32(10)) | (m_norm >> np.uint32(13)),
        )
        out[norm_mask] = res_norm.astype("<u2")

    # Subnormal / underflow range: exp <= 0
    sub_mask = exp <= 0
    if np.any(sub_mask):
        e_sub = exp[sub_mask]
        s_sub = sign[sub_mask]
        m_sub = mantissa[sub_mask]
        res_sub = s_sub.copy()
        valid_sub = e_sub >= -10
        if np.any(valid_sub):
            shift = (1 - e_sub[valid_sub]).astype(np.uint32)
            m_v = (m_sub[valid_sub] | np.uint32(0x00800000)) >> shift
            m_v = np.where((m_v & np.uint32(0x00001000)) != 0, m_v + np.uint32(0x00002000), m_v)
            res_sub[valid_sub] = s_sub[valid_sub] | (m_v >> np.uint32(13))
        out[sub_mask] = res_sub.astype("<u2")

    # Overflow / Inf / NaN range: exp >= 31
    ovf_mask = exp >= 31
    if np.any(ovf_mask):
        e_ovf = exp[ovf_mask]
        s_ovf = sign[ovf_mask]
        m_ovf = mantissa[ovf_mask]
        is_nan = (e_ovf == (0xFF - 127 + 15)) & (m_ovf != 0)
        m_shifted = m_ovf >> np.uint32(13)
        nan_payload = m_shifted | np.where(m_shifted == 0, np.uint32(1), np.uint32(0))
        res_ovf = np.where(
            is_nan,
            s_ovf | np.uint32(0x7C00) | nan_payload,
            s_ovf | np.uint32(0x7C00),
        )
        out[ovf_mask] = res_ovf.astype("<u2")

    return out


def quantize_fp32_bytes_to_q4_0_bytes(
    fp32_bytes: bytes,
    error_stats: Optional[QuantErrorStats] = None,
) -> bytes:
    """Quantise little-endian IEEE 754 FP32 bytes into packed 18-byte `BlockQ4_0` records.

    Matches `inference_engine::quantize_row_q4_0` (`src/kernels/quant_q4_0.cpp`) and upstream GGML
    `Q4_0` bit-for-bit:
      - 32 elements per block, 18 bytes per block (`uint16_t d` + `uint8_t qs[16]`)
      - Scale `d = signed_max / -8.0f`, stored as IEEE 754 FP16 (`fp32_to_fp16(d)`)
      - Inverse scale `id = (d != 0.0f) ? (1.0f / d) : 0.0f` computed from the FP32 `d`
      - Low nibble `qs[j] & 0x0F` stores element `j` (`0..15`)
      - High nibble `(qs[j] >> 4) & 0x0F` stores element `j + 16` (`16..31`)
      - Quantised code `q = clamp(floor(x * id + 8.5f), 0, 15)`
    """
    if len(fp32_bytes) == 0 or (len(fp32_bytes) % (Q4_0_BLOCK_ELEMENTS * 4)) != 0:
        raise ValueError(
            f"FP32 byte length ({len(fp32_bytes)}) must be a positive multiple of 128 bytes (32 floats)"
        )

    if HAS_NUMPY:
        blocks = np.frombuffer(fp32_bytes, dtype="<f4").reshape(-1, Q4_0_BLOCK_ELEMENTS)
        num_blocks = blocks.shape[0]
        abs_blocks = np.abs(blocks)
        # np.argmax selects the first index on ties, matching `if (amax < av)` in quant_q4_0.cpp.
        max_idx = np.argmax(abs_blocks, axis=1)
        row_arange = np.arange(num_blocks)
        amax = abs_blocks[row_arange, max_idx]
        signed_max = np.where(amax > np.float32(0.0), blocks[row_arange, max_idx], np.float32(0.0))

        d_f32 = (signed_max / np.float32(-8.0)).astype("<f4")
        with np.errstate(divide="ignore", invalid="ignore"):
            id_f32 = np.where(
                d_f32 != np.float32(0.0),
                np.float32(1.0) / d_f32,
                np.float32(0.0),
            ).astype("<f4")

        d_u16 = fp32_to_fp16_u16_numpy(d_f32)

        x0 = (blocks[:, :16] * id_f32[:, None]) + np.float32(8.5)
        x1 = (blocks[:, 16:32] * id_f32[:, None]) + np.float32(8.5)

        q0 = np.clip(np.floor(x0), 0.0, 15.0).astype(np.uint8)
        q1 = np.clip(np.floor(x1), 0.0, 15.0).astype(np.uint8)
        packed_qs = (q0 & np.uint8(0x0F)) | ((q1 & np.uint8(0x0F)) << np.uint8(4))

        out_blocks = np.empty((num_blocks, Q4_0_BLOCK_BYTES), dtype=np.uint8)
        out_blocks[:, 0:2] = d_u16.view(np.uint8).reshape(num_blocks, 2)
        out_blocks[:, 2:18] = packed_qs

        if error_stats is not None:
            d_deq = d_u16.view("<f2").astype(np.float64)
            deq0 = (q0.astype(np.float64) - 8.0) * d_deq[:, None]
            deq1 = (q1.astype(np.float64) - 8.0) * d_deq[:, None]
            orig64 = blocks.astype(np.float64)
            diff0 = np.abs(orig64[:, :16] - deq0)
            diff1 = np.abs(orig64[:, 16:32] - deq1)
            chunk_stats = QuantErrorStats(
                num_elements=int(blocks.size),
                max_abs_error=float(max(np.max(diff0), np.max(diff1))),
                sum_abs_error=float(np.sum(diff0) + np.sum(diff1)),
                sum_sq_error=float(np.sum(diff0 * diff0) + np.sum(diff1 * diff1)),
                sum_sq_signal=float(np.sum(orig64 * orig64)),
            )
            error_stats.update(chunk_stats)

        return out_blocks.tobytes()

    # Pure-Python exact IEEE 754 float32 fallback when NumPy is not installed.
    num_elements = len(fp32_bytes) // 4
    num_blocks = num_elements // Q4_0_BLOCK_ELEMENTS
    floats = struct.unpack(f"<{num_elements}f", fp32_bytes)
    out = bytearray(num_blocks * Q4_0_BLOCK_BYTES)

    for b in range(num_blocks):
        base = b * Q4_0_BLOCK_ELEMENTS
        amax_val = 0.0
        signed_max_val = 0.0
        for i in range(Q4_0_BLOCK_ELEMENTS):
            v = floats[base + i]
            av = abs(v)
            if amax_val < av:
                amax_val = av
                signed_max_val = v

        d_val = f32_round(signed_max_val / -8.0)
        id_val = f32_round(1.0 / d_val) if d_val != 0.0 else 0.0
        d_bits = fp32_to_fp16_u16_scalar(d_val)

        out_off = b * Q4_0_BLOCK_BYTES
        out[out_off] = d_bits & 0xFF
        out[out_off + 1] = (d_bits >> 8) & 0xFF

        d_deq_f32 = fp16_u16_to_fp32_scalar(d_bits) if error_stats is not None else 0.0

        for j in range(Q4_0_PACKED_BYTES):
            w0 = floats[base + j]
            w1 = floats[base + j + Q4_0_PACKED_BYTES]
            x0_val = f32_round(f32_round(w0 * id_val) + 8.5)
            x1_val = f32_round(f32_round(w1 * id_val) + 8.5)
            q0_int = max(0, min(15, int(math.floor(x0_val))))
            q1_int = max(0, min(15, int(math.floor(x1_val))))
            out[out_off + 2 + j] = (q0_int & 0x0F) | ((q1_int & 0x0F) << 4)

            if error_stats is not None:
                deq_w0 = (q0_int - 8) * d_deq_f32
                deq_w1 = (q1_int - 8) * d_deq_f32
                err0 = abs(w0 - deq_w0)
                err1 = abs(w1 - deq_w1)
                error_stats.num_elements += 2
                if err0 > error_stats.max_abs_error:
                    error_stats.max_abs_error = err0
                if err1 > error_stats.max_abs_error:
                    error_stats.max_abs_error = err1
                error_stats.sum_abs_error += err0 + err1
                error_stats.sum_sq_error += err0 * err0 + err1 * err1
                error_stats.sum_sq_signal += w0 * w0 + w1 * w1

    return bytes(out)


def dequantize_q4_0_bytes_to_fp32_list(q4_bytes: bytes) -> List[float]:
    """Dequantise packed 18-byte `BlockQ4_0` records back into FP32 floats for parity verification."""
    if len(q4_bytes) == 0 or (len(q4_bytes) % Q4_0_BLOCK_BYTES) != 0:
        raise ValueError(f"Q4_0 byte length ({len(q4_bytes)}) must be a positive multiple of 18")
    num_blocks = len(q4_bytes) // Q4_0_BLOCK_BYTES
    out: List[float] = [0.0] * (num_blocks * Q4_0_BLOCK_ELEMENTS)
    for b in range(num_blocks):
        off = b * Q4_0_BLOCK_BYTES
        d_bits = q4_bytes[off] | (q4_bytes[off + 1] << 8)
        d = fp16_u16_to_fp32_scalar(d_bits)
        base = b * Q4_0_BLOCK_ELEMENTS
        for j in range(Q4_0_PACKED_BYTES):
            packed = q4_bytes[off + 2 + j]
            q_low = (packed & 0x0F) - 8
            q_high = ((packed >> 4) & 0x0F) - 8
            out[base + j] = f32_round(float(q_low) * d)
            out[base + j + Q4_0_PACKED_BYTES] = f32_round(float(q_high) * d)
    return out


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


def select_target_tensor_dtype(
    tensor_name: str,
    shape: List[int],
    model_weight_dtype: int,
    keep_embeddings_fp32: bool = False,
) -> int:
    """Determine the target `.ie` DataType for a given tensor under the mixed-precision policy.

    Policy rules:
    1. If `model_weight_dtype == DTYPE_FP32`, all tensors are stored as `DTYPE_FP32`.
    2. If `model_weight_dtype == DTYPE_Q4_0`:
       - 1D RMSNorm tensors (`ndim == 1`, e.g. `*.input_layernorm.weight`,
         `*.post_attention_layernorm.weight`, `model.norm.weight`) are ALWAYS stored as `DTYPE_FP32`.
       - If `keep_embeddings_fp32` is True, `model.embed_tokens.weight` and `lm_head.weight` are kept
         in `DTYPE_FP32`.
       - All remaining 2D+ weight matrices are quantised to `DTYPE_Q4_0`, provided their innermost
         dimension `shape[-1]` is a positive multiple of 32.
    """
    if model_weight_dtype == DTYPE_FP32:
        return DTYPE_FP32
    if model_weight_dtype != DTYPE_Q4_0:
        raise ValueError(f"Unsupported model_weight_dtype: {model_weight_dtype}")

    if len(shape) == 1 or tensor_name.endswith("layernorm.weight") or tensor_name == "model.norm.weight":
        return DTYPE_FP32

    if keep_embeddings_fp32 and tensor_name in ("model.embed_tokens.weight", "lm_head.weight"):
        return DTYPE_FP32

    if shape[-1] <= 0 or (shape[-1] % Q4_0_BLOCK_ELEMENTS) != 0:
        raise ValueError(
            f"Tensor {tensor_name!r} with shape {shape} cannot be quantised to Q4_0: "
            f"innermost dimension ({shape[-1]}) is not divisible by {Q4_0_BLOCK_ELEMENTS}"
        )
    return DTYPE_Q4_0


def compute_tensor_dst_nbytes(num_elements: int, shape: List[int], dst_dtype: int, name: str = "") -> int:
    if dst_dtype == DTYPE_FP32:
        return num_elements * 4
    if dst_dtype == DTYPE_Q4_0:
        if len(shape) == 0 or (shape[-1] % Q4_0_BLOCK_ELEMENTS) != 0 or (num_elements % Q4_0_BLOCK_ELEMENTS) != 0:
            raise ValueError(
                f"Tensor {name!r} shape {shape} (num_elements={num_elements}) is not divisible by "
                f"{Q4_0_BLOCK_ELEMENTS} for Q4_0"
            )
        return (num_elements // Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES
    raise ValueError(f"Unsupported destination dtype {dst_dtype} for tensor {name!r}")


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
    model_weight_dtype: int = DTYPE_FP32,
    keep_embeddings_fp32: bool = False,
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

            dst_dtype = select_target_tensor_dtype(
                tensor_name=name,
                shape=shape,
                model_weight_dtype=model_weight_dtype,
                keep_embeddings_fp32=keep_embeddings_fp32,
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
                    dst_dtype=dst_dtype,
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


def resolve_input_files(
    input_path: Path, config_override: Optional[Path] = None
) -> Tuple[Path, List[Path]]:
    """Resolve either a checkpoint directory or a specific `.safetensors` file to `(config_path, [safetensor_files])`."""
    if input_path.is_file():
        if not input_path.name.endswith(".safetensors"):
            raise ValueError(f"Input file {input_path} is not a .safetensors file")
        safetensor_files = [input_path]
        config_path = config_override if config_override is not None else (input_path.parent / "config.json")
    elif input_path.is_dir():
        safetensor_files = sorted(input_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No *.safetensors files found in {input_path}")
        config_path = config_override if config_override is not None else (input_path / "config.json")
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    return config_path, safetensor_files


def convert_safetensors_to_ie(
    input_path: Path,
    output_path: Path,
    target_dtype: str = "fp32",
    strict_inventory: bool = True,
    chunk_elements: int = 2 * 1024 * 1024,
    keep_embeddings_fp32: bool = False,
    config_override: Optional[Path] = None,
    out_error_stats: Optional[QuantErrorStats] = None,
) -> int:
    if target_dtype == "fp32":
        model_weight_dtype = DTYPE_FP32
    elif target_dtype == "q4_0":
        model_weight_dtype = DTYPE_Q4_0
    else:
        raise ValueError(f"Unsupported target_dtype {target_dtype!r}; expected 'fp32' or 'q4_0'")

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    # Ensure chunk_elements is always a positive multiple of 32 so Q4_0 blocks are never split across chunks.
    if chunk_elements % Q4_0_BLOCK_ELEMENTS != 0:
        chunk_elements = max(
            Q4_0_BLOCK_ELEMENTS,
            ((chunk_elements + Q4_0_BLOCK_ELEMENTS - 1) // Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_ELEMENTS,
        )

    config_path, safetensor_files = resolve_input_files(input_path, config_override=config_override)

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

    expected_inventory = build_expected_inventory(cfg)
    records = parse_safetensors_SourceRecords(
        safetensor_files=safetensor_files,
        expected_inventory=expected_inventory,
        strict_inventory=strict_inventory,
        model_weight_dtype=model_weight_dtype,
        keep_embeddings_fp32=keep_embeddings_fp32,
    )

    # Pass 1: Compute directory size, 64-byte aligned offsets, and exact payload byte sizes.
    dir_bytes = 0
    for rec in records:
        name_len = len(rec.name.encode("utf-8"))
        dir_bytes += 4 + name_len + 56

    payload_cursor = align_up(64 + dir_bytes, IE_ALIGNMENT) if records else 64
    for rec in records:
        payload_cursor = align_up(payload_cursor, IE_ALIGNMENT)
        rec.dst_offset = payload_cursor
        rec.dst_nbytes = compute_tensor_dst_nbytes(
            num_elements=rec.num_elements,
            shape=rec.shape,
            dst_dtype=rec.dst_dtype,
            name=rec.name,
        )
        payload_cursor += rec.dst_nbytes

    total_file_bytes = payload_cursor
    header_bytes = pack_header(
        tensor_count=len(records),
        weight_dtype=model_weight_dtype,
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

    # Pass 2: Stream header, directory, and chunked FP32 / Q4_0 tensor payloads to disk.
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

                    if rec.dst_dtype == DTYPE_FP32:
                        dst_chunk = fp32_chunk
                    elif rec.dst_dtype == DTYPE_Q4_0:
                        dst_chunk = quantize_fp32_bytes_to_q4_0_bytes(
                            fp32_chunk, error_stats=out_error_stats
                        )
                    else:
                        raise ValueError(f"Unsupported destination dtype {rec.dst_dtype}")

                    out_f.write(dst_chunk)
                    written_payload += len(dst_chunk)

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


def convert_safetensors_to_ie_fp32(
    input_dir: Path,
    output_path: Path,
    strict_inventory: bool = True,
    chunk_elements: int = 2 * 1024 * 1024,
) -> int:
    """Backward-compatible wrapper for M1 FP32 conversion."""
    return convert_safetensors_to_ie(
        input_path=input_dir,
        output_path=output_path,
        target_dtype="fp32",
        strict_inventory=strict_inventory,
        chunk_elements=chunk_elements,
    )


def convert_safetensors_to_ie_q4_0(
    input_path: Path,
    output_path: Path,
    strict_inventory: bool = True,
    chunk_elements: int = 2 * 1024 * 1024,
    keep_embeddings_fp32: bool = False,
    out_error_stats: Optional[QuantErrorStats] = None,
) -> int:
    """Convert a `.safetensors` checkpoint directory or file to `.ie` v1 `Q4_0` format."""
    return convert_safetensors_to_ie(
        input_path=input_path,
        output_path=output_path,
        target_dtype="q4_0",
        strict_inventory=strict_inventory,
        chunk_elements=chunk_elements,
        keep_embeddings_fp32=keep_embeddings_fp32,
        out_error_stats=out_error_stats,
    )


def float_to_bf16_u16(val: float) -> int:
    """Pack a Python float into IEEE 754 bfloat16 upper-16 bits for self-test fixture generation."""
    bits = struct.unpack("<I", struct.pack("<f", val))[0]
    return (bits >> 16) & 0xFFFF


def reference_ggml_quantize_block_q4_0(weights_32: List[float]) -> bytes:
    """Independent scalar GGML reference for a single 32-element Q4_0 block."""
    assert len(weights_32) == 32
    amax = 0.0
    max_val = 0.0
    for v in weights_32:
        av = abs(v)
        if amax < av:
            amax = av
            max_val = v
    d = f32_round(max_val / -8.0)
    id_val = f32_round(1.0 / d) if d != 0.0 else 0.0
    d_fp16 = fp32_to_fp16_u16_scalar(d)
    qs = bytearray(16)
    for j in range(16):
        x0 = f32_round(f32_round(weights_32[j] * id_val) + 8.5)
        x1 = f32_round(f32_round(weights_32[j + 16] * id_val) + 8.5)
        xi0 = min(15, max(0, int(math.floor(x0))))
        xi1 = min(15, max(0, int(math.floor(x1))))
        qs[j] = xi0 | (xi1 << 4)
    return struct.pack("<H", d_fp16) + bytes(qs)


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

    # 1. Verify Q4_0 block packing against independent GGML scalar reference (including zero blocks).
    zero_f32 = struct.pack("<32f", *([0.0] * 32))
    zero_q4 = quantize_fp32_bytes_to_q4_0_bytes(zero_f32)
    assert len(zero_q4) == 18
    assert zero_q4 == reference_ggml_quantize_block_q4_0([0.0] * 32)
    assert fp16_u16_to_fp32_scalar(struct.unpack("<H", zero_q4[:2])[0]) == 0.0
    assert zero_q4[2:] == bytes([0x88] * 16)
    assert dequantize_q4_0_bytes_to_fp32_list(zero_q4) == [0.0] * 32

    ramp_vals = [f32_round((i - 15.5) * 0.1375 * (-1.0 if i % 3 == 0 else 1.0)) for i in range(64)]
    ramp_f32 = struct.pack("<64f", *ramp_vals)
    ramp_q4 = quantize_fp32_bytes_to_q4_0_bytes(ramp_f32)
    ref_ramp_q4 = reference_ggml_quantize_block_q4_0(ramp_vals[:32]) + reference_ggml_quantize_block_q4_0(
        ramp_vals[32:]
    )
    assert ramp_q4 == ref_ramp_q4

    # Verify NumPy vectorized vs pure-Python fallback bit-exact parity across 8 blocks (256 floats).
    global HAS_NUMPY
    saved_has_numpy = HAS_NUMPY
    try:
        edge_vals = [
            f32_round(math.sin(i * 0.37) * 7.25 + (1e-6 if i % 11 == 0 else 0.0))
            for i in range(256)
        ]
        edge_f32 = struct.pack("<256f", *edge_vals)
        HAS_NUMPY = False
        py_stats = QuantErrorStats()
        py_q4 = quantize_fp32_bytes_to_q4_0_bytes(edge_f32, error_stats=py_stats)
        if saved_has_numpy:
            HAS_NUMPY = True
            np_stats = QuantErrorStats()
            np_q4 = quantize_fp32_bytes_to_q4_0_bytes(edge_f32, error_stats=np_stats)
            assert np_q4 == py_q4
            assert abs(np_stats.mean_abs_error - py_stats.mean_abs_error) < 1e-12
            assert abs(np_stats.rmse - py_stats.rmse) < 1e-12
    finally:
        HAS_NUMPY = saved_has_numpy

    # 2. End-to-end synthetic 2-layer Llama safetensors -> .ie v1 conversion (FP32 & Q4_0).
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
        expected_q4_bytes_by_name: Dict[str, bytes] = {}
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
            if len(shape) == 2:
                ref_blocks = b"".join(
                    reference_ggml_quantize_block_q4_0(fp32_vals[b : b + 32])
                    for b in range(0, num_elems, 32)
                )
                expected_q4_bytes_by_name[name] = ref_blocks
            cursor += len(raw_bytes)

        hdr_json = json.dumps(st_header).encode("utf-8")
        sf_path = tmp_dir / "model.safetensors"
        with sf_path.open("wb") as f:
            f.write(struct.pack("<Q", len(hdr_json)))
            f.write(hdr_json)
            for chunk in payload_chunks:
                f.write(chunk)

        # 2a. Verify FP32 conversion & reload.
        ie_fp32_out = tmp_dir / "synthetic_model_fp32.ie"
        total_fp32_bytes = convert_safetensors_to_ie_fp32(
            tmp_dir, ie_fp32_out, strict_inventory=True, chunk_elements=256
        )
        assert ie_fp32_out.exists() and ie_fp32_out.stat().st_size == total_fp32_bytes

        with ie_fp32_out.open("rb") as f:
            unpacked_hdr = IE_HEADER_STRUCT.unpack(f.read(64))
            assert unpacked_hdr[0] == IE_MAGIC
            assert unpacked_hdr[1] == IE_VERSION
            assert unpacked_hdr[2] == 21
            assert unpacked_hdr[3] == DTYPE_FP32
            assert unpacked_hdr[14] == total_fp32_bytes

            loaded_records = []
            for _ in range(21):
                nlen = struct.unpack("<I", f.read(4))[0]
                tname = f.read(nlen).decode("utf-8")
                meta = IE_DIR_META_STRUCT.unpack(f.read(56))
                tdtype, ndim = meta[0], meta[1]
                tshape = list(meta[2:6])[:ndim]
                toffset, tnbytes = meta[6], meta[7]
                assert tdtype == DTYPE_FP32
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

        # 2b. Verify Q4_0 conversion (via direct .safetensors file path and small non-multiple chunk_elements=100
        # to exercise streaming chunk-boundary rounding and multi-chunk Q4_0 block packing).
        ie_q4_out = tmp_dir / "synthetic_model_q4_0.ie"
        q4_stats = QuantErrorStats()
        total_q4_bytes = convert_safetensors_to_ie(
            input_path=sf_path,
            output_path=ie_q4_out,
            target_dtype="q4_0",
            strict_inventory=True,
            chunk_elements=100,  # exercises rounding to 128 elements (4 Q4_0 blocks per chunk)
            out_error_stats=q4_stats,
        )
        assert ie_q4_out.exists() and ie_q4_out.stat().st_size == total_q4_bytes
        assert total_q4_bytes < total_fp32_bytes
        assert q4_stats.num_elements > 0
        assert q4_stats.max_abs_error < 0.25
        assert q4_stats.rmse < 0.08

        with ie_q4_out.open("rb") as f:
            unpacked_hdr = IE_HEADER_STRUCT.unpack(f.read(64))
            assert unpacked_hdr[0] == IE_MAGIC
            assert unpacked_hdr[1] == IE_VERSION
            assert unpacked_hdr[2] == 21
            assert unpacked_hdr[3] == DTYPE_Q4_0
            assert unpacked_hdr[14] == total_q4_bytes

            q4_records = []
            for _ in range(21):
                nlen = struct.unpack("<I", f.read(4))[0]
                tname = f.read(nlen).decode("utf-8")
                meta = IE_DIR_META_STRUCT.unpack(f.read(56))
                tdtype, ndim = meta[0], meta[1]
                tshape = list(meta[2:6])[:ndim]
                toffset, tnbytes = meta[6], meta[7]
                assert toffset % IE_ALIGNMENT == 0
                q4_records.append((tname, tdtype, tshape, toffset, tnbytes))

            for tname, tdtype, tshape, toffset, tnbytes in q4_records:
                assert tshape == expected_inv[tname]
                f.seek(toffset)
                raw_payload = f.read(tnbytes)
                if len(tshape) == 1:
                    assert tdtype == DTYPE_FP32
                    count = tnbytes // 4
                    actual_floats = list(struct.unpack(f"<{count}f", raw_fp32 := raw_payload))
                    assert actual_floats == expected_fp32_by_name[tname]
                else:
                    assert tdtype == DTYPE_Q4_0
                    assert tnbytes == (tshape[0] * tshape[1] // 32) * 18
                    assert raw_payload == expected_q4_bytes_by_name[tname]

        # 2c. Verify Q4_0 conversion with --keep-embeddings-fp32 mixed-precision policy.
        ie_q4_keep_emb = tmp_dir / "synthetic_model_q4_0_fp32_emb.ie"
        convert_safetensors_to_ie(
            input_path=tmp_dir,
            output_path=ie_q4_keep_emb,
            target_dtype="q4_0",
            strict_inventory=True,
            keep_embeddings_fp32=True,
        )
        with ie_q4_keep_emb.open("rb") as f:
            f.seek(64)
            for _ in range(21):
                nlen = struct.unpack("<I", f.read(4))[0]
                tname = f.read(nlen).decode("utf-8")
                meta = IE_DIR_META_STRUCT.unpack(f.read(56))
                tdtype, ndim = meta[0], meta[1]
                if ndim == 1 or tname in ("model.embed_tokens.weight", "lm_head.weight"):
                    assert tdtype == DTYPE_FP32
                else:
                    assert tdtype == DTYPE_Q4_0

        # 3. Negative test: non-multiple-of-32 2D shape in Q4_0 mode must raise ValueError.
        bad_shape_hdr = dict(st_header)
        bad_shape_hdr["lm_head.weight"] = {
            "dtype": "BF16",
            "shape": [128, 30],  # 30 is not divisible by 32
            "data_offsets": [0, 128 * 30 * 2],
        }
        bad_shape_sf = tmp_dir / "bad_shape.safetensors"
        bad_shape_json = json.dumps(bad_shape_hdr).encode("utf-8")
        with bad_shape_sf.open("wb") as f:
            f.write(struct.pack("<Q", len(bad_shape_json)))
            f.write(bad_shape_json)
            for chunk in payload_chunks:
                f.write(chunk)
        try:
            convert_safetensors_to_ie(
                input_path=bad_shape_sf,
                output_path=tmp_dir / "bad_q4.ie",
                target_dtype="q4_0",
                strict_inventory=False,
            )
            raise AssertionError("Expected non-multiple-of-32 Q4_0 shape to raise ValueError")
        except ValueError:
            pass

        # 4. Negative test: missing required tensor in safetensors must raise ValueError.
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
        f"streaming BF16->FP32 and BF16->Q4_0 safetensors conversion across 21 synthetic Llama tensors "
        f"(Q4_0 MAE={q4_stats.mean_abs_error:.6f}, RMSE={q4_stats.rmse:.6f}, MaxErr={q4_stats.max_abs_error:.6f})."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TinyLlama-1.1B-Chat-v1.0 (.safetensors) checkpoint into `.ie` v1 format."
    )
    parser.add_argument(
        "--input",
        "--input-dir",
        dest="input_path",
        type=Path,
        help="Input Hugging Face checkpoint directory or model.safetensors file path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit path to config.json (defaults to <input_dir>/config.json)",
    )
    parser.add_argument("--output", type=Path, help="Output .ie model path")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "q4_0"],
        default="fp32",
        help="Target linear weight precision (default: fp32)",
    )
    parser.add_argument(
        "--keep-embeddings-fp32",
        action="store_true",
        help="When --dtype q4_0 is used, keep model.embed_tokens.weight and lm_head.weight in FP32",
    )
    parser.add_argument(
        "--report-error",
        action="store_true",
        help="Compute and print Q4_0 quantisation error metrics (MAE, RMSE, Max Abs Error, Rel L2)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run end-to-end header packing and synthetic BF16 safetensors -> .ie FP32 & Q4_0 conversion self-test",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.input_path is None or args.output is None:
        parser.print_help()
        return 0

    err_stats = QuantErrorStats() if (args.dtype == "q4_0" and args.report_error) else None
    total_bytes = convert_safetensors_to_ie(
        input_path=args.input_path,
        output_path=args.output,
        target_dtype=args.dtype,
        strict_inventory=True,
        keep_embeddings_fp32=args.keep_embeddings_fp32,
        config_override=args.config,
        out_error_stats=err_stats,
    )
    print(
        f"[OK] Converted {args.input_path} -> {args.output} "
        f"({total_bytes:,} bytes, dtype={args.dtype.upper()}"
        f"{', embeddings=FP32' if (args.dtype == 'q4_0' and args.keep_embeddings_fp32) else ''})"
    )
    if err_stats is not None and err_stats.num_elements > 0:
        print(
            f"[Q4_0 ERROR METRICS] elements={err_stats.num_elements:,} | "
            f"MAE={err_stats.mean_abs_error:.6e} | RMSE={err_stats.rmse:.6e} | "
            f"MaxAbsErr={err_stats.max_abs_error:.6e} | RelL2={err_stats.relative_l2_error:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
