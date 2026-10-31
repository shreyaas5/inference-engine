# `.ie` Model Binary Format Specification (Version 1)

## 1. Design Principles

1. **Self-Describing:** Contains magic bytes, format version, full model hyperparameters, and a complete tensor directory.
2. **No Native Struct Dumps:** Every integer and floating-point field is serialized and parsed explicitly in **little-endian** byte order (`read_u32_le`, `read_u64_le`, `write_u32_le`, `write_u64_le`). Compiler padding and host ABI differences cannot affect the on-disk format.
3. **64-Byte Payload Alignment:** Every tensor payload begins at a file byte offset divisible by `64` (`payload_offset % 64 == 0`), enabling zero-copy `mmap` and aligned AVX2 / NEON vector loads.
4. **Defensive Validation:** The loader validates all arithmetic with checked 64-bit integer operations (`checked_add_u64`, `checked_mul_u64`), verifies exact file size, checks that no payload overlaps the header/directory or any other payload, and verifies exact byte count per tensor.

## 2. File Layout Overview

```text
+-----------------------------------------------------------------------+
| Offset 0..63: Fixed 64-Byte File Header (Little-Endian)               |
+-----------------------------------------------------------------------+
| Offset 64..D-1: Variable-Length Tensor Directory (tensor_count items) |
+-----------------------------------------------------------------------+
| Zero Padding to 64-Byte Boundary                                      |
+-----------------------------------------------------------------------+
| Tensor 0 Payload (aligned to 64 bytes)                                |
+-----------------------------------------------------------------------+
| Zero Padding to 64-Byte Boundary                                      |
+-----------------------------------------------------------------------+
| Tensor 1 Payload (aligned to 64 bytes)                                |
| ...                                                                   |
+-----------------------------------------------------------------------+
```

## 3. Fixed 64-Byte Header Layout

| Byte Offset | Type | Field Name | Description |
|---|---|---|---|
| `0..3` | `u32` | `magic` | `0x464D4549` (ASCII `"IEMF"` in little-endian) |
| `4..7` | `u32` | `version` | `1` (`kIeFormatVersion`) |
| `8..11` | `u32` | `tensor_count` | Number of tensors in directory ($\le 4096$) |
| `12..15` | `u32` | `weight_dtype` | Primary weight `DataType` (`0`=FP32, `1`=FP16, `2`=Q4_0, `3`=I32) |
| `16..19` | `u32` | `num_layers` | Transformer layer count (`22` for TinyLlama) |
| `20..23` | `u32` | `hidden_dim` | Hidden size (`2048`) |
| `24..27` | `u32` | `intermediate_dim` | FFN intermediate size (`5632`) |
| `28..31` | `u32` | `num_query_heads` | Query attention head count (`32`) |
| `32..35` | `u32` | `num_kv_heads` | Key/Value head count (`4`) |
| `36..39` | `u32` | `head_dim` | Per-head dimension (`64`) |
| `40..43` | `u32` | `vocab_size` | Tokenizer vocabulary size (`32000`) |
| `44..47` | `u32` | `max_seq_len` | Maximum sequence length (`2048`) |
| `48..51` | `f32` | `rms_norm_eps` | IEEE 754 binary32 epsilon (`1e-5`) |
| `52..55` | `f32` | `rope_theta` | IEEE 754 binary32 RoPE base (`10000.0`) |
| `56..63` | `u64` | `total_file_bytes` | Exact total byte length of the `.ie` file |

## 4. Tensor Directory Entry Layout

Immediately following byte `64`, `tensor_count` entries are stored back-to-back:

| Field | Type | Size (Bytes) | Description |
|---|---|---|---|
| `name_len` | `u32` | `4` | Length of UTF-8 tensor name in bytes ($1 \le \text{name\_len} \le 256$) |
| `name` | `u8[name_len]` | `name_len` | UTF-8 tensor identifier (no null terminator stored) |
| `dtype` | `u32` | `4` | `0` = `FP32`, `1` = `FP16`, `2` = `Q4_0`, `3` = `I32` |
| `ndim` | `u32` | `4` | Number of active dimensions ($1 \le \text{ndim} \le 4$) |
| `shape[0..3]` | `u64[4]` | `32` | Tensor dimensions; `shape[i]` for $i < \text{ndim}$ must be $>0$, and $i \ge \text{ndim}$ must be `0` |
| `payload_offset` | `u64` | `8` | Absolute byte offset of payload (`payload_offset % 64 == 0`) |
| `payload_nbytes` | `u64` | `8` | Exact byte length of payload |

## 5. Validation Invariants Enforced by `ModelLoader`

1. `magic == 0x464D4549` and `version == 1`.
2. `total_file_bytes == actual_buffer_or_file_size`.
3. `ModelConfig::validate()` passes on header hyperparameters.
4. Every tensor name is unique and non-empty.
5. `num_elements = prod(shape[0..ndim-1])` computed with `checked_mul_u64` does not overflow.
6. For `Q4_0` tensors, the innermost dimension `shape[ndim - 1]` is divisible by `32`, and `payload_nbytes == (num_elements / 32) * 18`.
7. For `FP32` tensors, `payload_nbytes == num_elements * 4`.
8. `payload_offset % 64 == 0` and `payload_offset + payload_nbytes <= total_file_bytes` (via `checked_add_u64`).
9. Sorting all `[payload_offset, payload_offset + payload_nbytes)` intervals confirms `payload_ranges[0].begin >= align_up(directory_end, 64)` and `payload_ranges[i].begin >= payload_ranges[i - 1].end` (zero overlap).
10. `payload_ranges.back().end == total_file_bytes` (no unreferenced trailing bytes).

## 6. Mixed-Precision `Q4_0` Model Policy & Exact `TinyLlama-1.1B-Chat-v1.0` Storage Breakdown

When exporting a `TinyLlama-1.1B-Chat-v1.0` checkpoint (`201` tensors, `1,100,048,384` parameters) with `tools/convert_to_inference_engine.py`:

| Format Mode | Header `weight_dtype` | 1D RMSNorm Tensors (`45`) | 2D Layer Projections (`154`) | `embed_tokens` & `lm_head` (`2`) | Payload Bytes | Total `.ie` File Bytes |
|---|---|---|---|---|---|---|
| `--dtype fp32` | `FP32` (`0`) | `FP32` (`368,640` B) | `FP32` (`3,875,536,896` B) | `FP32` (`524,288,000` B) | `4,400,193,536` | **`4,400,212,992`** (~4.40 GB) |
| `--dtype q4_0` (Default) | `Q4_0` (`2`) | `FP32` (`368,640` B) | `Q4_0` (`544,997,376` B) | `Q4_0` (`73,728,000` B) | `619,094,016` | **`619,113,472`** (~619.11 MB) |
| `--dtype q4_0 --keep-embeddings-fp32` | `Q4_0` (`2`) | `FP32` (`368,640` B) | `Q4_0` (`544,997,376` B) | `FP32` (`524,288,000` B) | `1,069,654,016` | **`1,069,673,472`** (~1.07 GB) |

- **1D RMSNorm weights** (`model.layers.{0..21}.input_layernorm.weight`, `model.layers.{0..21}.post_attention_layernorm.weight`, `model.norm.weight`) are **always stored in `FP32`** to preserve per-channel normalization fidelity at negligible storage cost (`368.64 KB`).
- **2D weight matrices** (`[out_features, in_features]`) have innermost dimension `in_features` $\in \{2048, 5632\}$, which are exact multiples of `32`, and are quantised to 18-byte `BlockQ4_0` records (`0.5625` bytes/weight).
