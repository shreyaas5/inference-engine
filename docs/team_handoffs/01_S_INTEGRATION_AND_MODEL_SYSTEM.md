# `01_S_INTEGRATION_AND_MODEL_SYSTEM.md` — Member S Module Handoff

**Owner:** Member S (Project Lead, Model Format, Checkpoint Converter, `ModelLoader`, & Cross-Module Integration)
**Consolidated Branch:** `integration-ps08-team-sync-2026-10-31`
**Merged Feature Branches:**
- `origin/feature/model-loader` (`aa40368`) — M1 `.ie` v1 loader, `mmap`, and `FP32` converter
- `origin/feature/model-format-q4` (`cb176b3`) — M2 `Q4_0` & mixed-precision converter, `TensorView::data_q4_0()`, unified `TransformerWeightBindings`
- `origin/feature/real-checkpoint-validation` (`4515579`) — Real `TinyLlama-1.1B-Chat-v1.0` 201-tensor header inspection and 9-role `Q4_0` quantisation error evaluation (`--eval-real-quant-error`)

---

## 1. Owned Files & Responsibilities

| File Path | Role | Status |
|---|---|---|
| [`include/inference_engine/common.hpp`](../../include/inference_engine/common.hpp), [`src/common/common.cpp`](../../src/common/common.cpp) | `Status`/`StatusCode`, `DataType` block traits (`dtype_block_bytes`, `dtype_block_elements`), checked 64-bit arithmetic (`checked_add_u64`, `checked_mul_u64`, `align_up_u64`) | `TESTED (SYNTHETIC)` |
| [`include/inference_engine/config.hpp`](../../include/inference_engine/config.hpp) | Frozen `TinyLlama-1.1B` constants, `ModelConfig` (`validate_for_tinyllama`), `RuntimeConfig` | `REFERENCE VERIFIED` |
| [`include/inference_engine/tensor.hpp`](../../include/inference_engine/tensor.hpp), [`src/common/tensor.cpp`](../../src/common/tensor.cpp) | `TensorDescriptor` (shape/offset/byte-size validator), `TensorView` (`data_f32()`, `data_q4_0()`, `num_elements()`, `num_blocks()`) | `TESTED (SYNTHETIC)` |
| [`include/inference_engine/loader.hpp`](../../include/inference_engine/loader.hpp), [`src/loader/loader.cpp`](../../src/loader/loader.cpp) | `.ie` v1 binary header/directory parser, POSIX `mmap` / buffer loader, `TransformerWeightBindings`, `ModelLoader::bind_transformer_weights()` | `TESTED (SYNTHETIC)` |
| [`tools/inspect_checkpoint.py`](../../tools/inspect_checkpoint.py) | Offline `.safetensors` inspector, live HF HTTP Range header verifier (`--inspect-remote-hf`), real-weight `Q4_0` error evaluator (`--eval-real-quant-error`) | `REFERENCE VERIFIED` |
| [`tools/convert_to_inference_engine.py`](../../tools/convert_to_inference_engine.py) | Two-pass streaming `.safetensors` (`BF16`/`FP16`/`FP32`) to `.ie` (`FP32`, `Q4_0`, or `--keep-embeddings-fp32`) converter with `--report-error` & `--self-test` | `REFERENCE VERIFIED` |
| [`tests/test_loader_validation.cpp`](../../tests/test_loader_validation.cpp) | 15 malformed `.ie` rejection tests + 21-tensor `FP32`, `Q4_0`, and mixed-precision `mmap` reload, binding, and `TransformerEngine` forward tests | `TESTED (8/8 CTest)` |

---

## 2. Frozen Technical Contracts Owned by S

### 2.1 `.ie` v1 Binary File Format
- **Endianness:** Little-endian throughout; serialized field-by-field (never raw struct dumps).
- **64-Byte Fixed Header (`ModelFileHeader`):**
  - `0x00`: `magic = 0x464D4549` (`"IEMF"`)
  - `0x04`: `version = 1`
  - `0x08`: `num_tensors` (`uint32_t`, `201` for 22-layer TinyLlama)
  - `0x0C`: `vocab_size = 32000`
  - `0x10`: `hidden_dim = 2048`
  - `0x14`: `intermediate_dim = 5632`
  - `0x18`: `num_layers` (`22` for full model; `1..256` accepted in `validate_for_tinyllama(true)`)
  - `0x1C`: `num_heads = 32`
  - `0x20`: `num_kv_heads = 4`
  - `0x24`: `head_dim = 64`
  - `0x28`: `max_seq_len = 2048`
  - `0x2C`: `rms_norm_eps = 1e-5f` (`uint32_t` IEEE 754 bit pattern)
  - `0x30`: `rope_theta = 10000.0f` (`uint32_t` IEEE 754 bit pattern)
  - `0x34`: `primary_dtype` (`0` = `FP32`, `1` = `Q4_0`)
  - `0x38`: `total_file_bytes` (`uint64_t`, must equal exact file size on disk)
- **Tensor Directory & Payload Alignment:**
  - Each directory entry stores `name_len` (`uint32_t`), UTF-8 `name`, `dtype` (`uint32_t`), `ndim` (`uint32_t`, `1..4`), `shape[4]` (`uint64_t[4]`), `payload_offset` (`uint64_t`), and `payload_nbytes` (`uint64_t`).
  - Every tensor's `payload_offset` is 64-byte aligned (`payload_offset % 64 == 0`), lies after the end of the tensor directory, does not overlap any other tensor, and matches `expected_payload_bytes(dtype, ndim, shape)`.

### 2.2 `ModelLoader::bind_transformer_weights()` Contract
- Validates all $3 + 9 \times \text{num\_layers}$ tensors by exact name, rank, shape, and dtype:
  - 1D RMSNorm tensors (`input_layernorm`, `post_attention_layernorm`, `model.norm`) **must always be `DataType::FP32`** (`[2048]`).
  - 2D linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) must all share a single consistent `linear_dtype` (`FP32` or `Q4_0`) matching `header.primary_dtype`.
  - `model.embed_tokens.weight` and `lm_head.weight` (`[32000, 2048]`) may be either `linear_dtype` (`FP32` or `Q4_0`) or `FP32` when `linear_dtype == Q4_0` (supporting `--keep-embeddings-fp32`).

### 2.3 Real TinyLlama-1.1B `Q4_0` Quantisation Error Measurements
Measured directly on live `TinyLlama/TinyLlama-1.1B-Chat-v1.0` `BF16` tensor slices via `python3 tools/inspect_checkpoint.py --eval-real-quant-error`:

| Tensor Role (`layer 0` / global) | Mean Abs Error (`MAE`) | Root Mean Sq Error (`RMSE`) | Relative $L_2$ Error (`RelL2`) |
|---|---|---|---|
| `embed_tokens.weight` | `5.356e-04` | `7.175e-04` | `8.80%` |
| `self_attn.q_proj.weight` | `9.190e-04` | `1.298e-03` | `11.71%` |
| `self_attn.k_proj.weight` | `2.592e-03` | `3.597e-03` | `12.86%` |
| `self_attn.v_proj.weight` | `8.170e-04` | `1.096e-03` | `9.04%` |
| `self_attn.o_proj.weight` | `5.387e-04` | `7.277e-03` | `9.18%` |
| `mlp.gate_proj.weight` | `1.188e-03` | `1.591e-03` | `8.84%` |
| `mlp.up_proj.weight` | `1.210e-03` | `1.630e-03` | `8.70%` |
| `mlp.down_proj.weight` | `1.210e-03` | `1.627e-03` | `8.88%` |
| `lm_head.weight` | `1.847e-03` | `2.463e-03` | `11.43%` |

---

## 3. Ready-to-Paste Antigravity Continuation Prompt for Member S

```markdown
You are acting as Member S (Project Lead, Model Format, Checkpoint Converter & Integration Architect) for `inference-engine` (BIOS V2 — PS08) in `/Users/shreyaas/Developer/inference-engine`.

Before making any changes, read:
1. `AGENTS.md`
2. `docs/team_handoffs/00_MASTER_PROJECT_CONTEXT.md`
3. `docs/team_handoffs/01_S_INTEGRATION_AND_MODEL_SYSTEM.md`
4. `docs/team_handoffs/06_CURRENT_STATUS_AND_NEXT_ACTIONS.md`

Current branch: `integration-ps08-team-sync-2026-10-31` (consolidates S `4515579`, U `ce1026a`, I `58c604d`, and P `ce769c8`; all 8/8 CTest suites pass).

Your next engineering tasks:
1. Complete the local download of `models/TinyLlama-1.1B-Chat-v1.0/model.safetensors` (`2,200,120,064` bytes):
   `curl -C - -L -o models/TinyLlama-1.1B-Chat-v1.0/model.safetensors "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"`
2. Inspect and verify the downloaded checkpoint:
   `python3 tools/inspect_checkpoint.py --checkpoint models/TinyLlama-1.1B-Chat-v1.0`
3. Convert the real 22-layer checkpoint into both `.ie` formats:
   - Pure `Q4_0` (`619.11 MB`):
     `/usr/bin/python3 tools/convert_to_inference_engine.py --checkpoint models/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama-1.1b-q4_0.ie --dtype q4_0 --report-error`
   - Pure `FP32` (`4.40 GB`):
     `/usr/bin/python3 tools/convert_to_inference_engine.py --checkpoint models/TinyLlama-1.1B-Chat-v1.0 --output models/tinyllama-1.1b-fp32.ie --dtype fp32`
4. Run end-to-end CLI inference and benchmarks on both `.ie` models using `./build/inference-engine` and record the measured results in `docs/BENCHMARKING.md` and `CURRENT_STATUS.md`.
```
