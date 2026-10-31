# Q4_0 Quantisation Contract (`inference-engine`)

## 1. Block Format Specification

`Q4_0` is a 4-bit block-wise symmetric quantisation format with an unsigned offset of `8`, compatible with upstream GGML / `llama.cpp` `block_q4_0`:

- **Block size (`kQ4_0BlockElements`):** `32` contiguous weights along the innermost matrix dimension.
- **Storage per block (`kQ4_0BlockBytes`):** `18` bytes:
  - Bytes `0..1`: `uint16_t d` — IEEE 754 binary16 (FP16) scale factor stored in little-endian byte order.
  - Bytes `2..17`: `uint8_t qs[16]` — 16 packed bytes containing 32 4-bit unsigned quantised nibbles $q_i \in [0, 15]$.

```text
+-------------------+---------------------------------------------------+
| Bytes 0..1        | Bytes 2..17 (qs[0..15])                           |
| FP16 scale `d`    | qs[j] low nibble  = weight j      (indices 0..15) |
| (little-endian)   | qs[j] high nibble = weight j + 16 (indices 16..31)|
+-------------------+---------------------------------------------------+
```

## 2. Nibble Packing Layout (Split-Half Order)

Within each 32-element block $x_0, x_1, \dots, x_{31}$, for byte index $j \in \{0, 1, \dots, 15\}$:

- **Low nibble** (`qs[j] & 0x0F`) stores quantised weight $q_j$ corresponding to element **$j$** (indices `0..15`).
- **High nibble** (`(qs[j] >> 4) & 0x0F`) stores quantised weight $q_{j+16}$ corresponding to element **$j + 16$** (indices `16..31`).

> **Important:** Elements are **not** paired as $(2j, 2j+1)$. They are split into the first half (`0..15` in low nibbles) and second half (`16..31` in high nibbles) to allow direct 128-bit SIMD vector byte loads (`vld1q_u8` in NEON / `_mm_loadu_si128` in AVX2) followed by bitwise nibble separation (`& 0x0F` and `>> 4`).

## 3. Exact Upstream-Compatible Quantisation Procedure

Given a block of 32 `FP32` weights $x_0, \dots, x_{31}$:

1. **Signed Maximum Selection:**
   Scan $i = 0 \dots 31$ tracking `amax` and `signed_max`:
   ```cpp
   float amax = 0.0f;
   float signed_max = 0.0f;
   for (size_t i = 0; i < 32; ++i) {
       const float v = block_src[i];
       const float av = std::fabs(v);
       if (amax < av) {
           amax = av;
           signed_max = v;
       }
   }
   ```
2. **Scale Calculation:**
   $$d = \frac{\text{signed\_max}}{-8.0}, \qquad \text{id} = \begin{cases} \frac{1.0}{d} & \text{if } d \neq 0.0 \\ 0.0 & \text{if } d = 0.0 \end{cases}$$
   Convert $d$ to IEEE 754 FP16 (`d_fp16 = fp32_to_fp16(d)`).
3. **Rounding & Clamping:**
   For each $j \in \{0, \dots, 15\}$:
   $$x^{(0)}_j = x_j \cdot \text{id}, \qquad x^{(1)}_j = x_{j+16} \cdot \text{id}$$
   $$q_j = \text{clamp}\!\left(\lfloor x^{(0)}_j + 8.5 \rfloor, \, 0, \, 15\right), \qquad q_{j+16} = \text{clamp}\!\left(\lfloor x^{(1)}_j + 8.5 \rfloor, \, 0, \, 15\right)$$
   Pack into byte $j$:
   $$\text{qs}[j] = (q_j \mathbin{\&} \text{0x0F}) \mathbin{|} ((q_{j+16} \mathbin{\&} \text{0x0F}) \ll 4)$$
4. **Zero-Block Handling:**
   When all 32 inputs in a block are `0.0f`, `amax = 0.0f`, `signed_max = 0.0f`, `d = 0.0f`, and `id = 0.0f`. Consequently, every nibble evaluates to $\lfloor 0.0 + 8.5 \rfloor = 8$, and dequantisation yields $(8 - 8) \times 0.0 = 0.0$ without `NaN` or division by zero.

## 4. Dequantisation Procedure

Given `BlockQ4_0`:
1. Convert `d` from FP16 to FP32: $d_{\text{f32}} = \text{fp16\_to\_fp32}(\text{block.d})$.
2. For each $j \in \{0, \dots, 15\}$:
   $$q_{\text{low}} = \text{ block.qs}[j] \mathbin{\&} \text{0x0F}, \qquad q_{\text{high}} = (\text{block.qs}[j] \gg 4) \mathbin{\&} \text{0x0F}$$
   $$\hat{x}_j = (\text{int}(q_{\text{low}}) - 8) \cdot d_{\text{f32}}, \qquad \hat{x}_{j+16} = (\text{int}(q_{\text{high}}) - 8) \cdot d_{\text{f32}}$$

## 5. Offline `.safetensors -> .ie` `Q4_0` Export & Error Metrics

[`tools/convert_to_inference_engine.py`](../tools/convert_to_inference_engine.py) implements bit-exact `Q4_0` export (`--dtype q4_0`) matching [`src/kernels/quant_q4_0.cpp`](../src/kernels/quant_q4_0.cpp) and upstream GGML `block_q4_0`:

- **Mixed-Precision Policy:**
  - All 45 1D RMSNorm tensors (`[2048]`) remain in `DataType::FP32` (`368,640` bytes total).
  - By default, all 156 2D weight tensors (`model.embed_tokens.weight`, `lm_head.weight`, and all 22 $\times$ 7 decoder projections) are quantised to `DataType::Q4_0` (`618,725,376` bytes total), producing a **`619,113,472`-byte (~619.11 MB)** `.ie` file ($7.11\times$ smaller than the `4.40 GB` `FP32` `.ie` file).
  - Passing `--keep-embeddings-fp32` keeps `model.embed_tokens.weight` and `lm_head.weight` in `DataType::FP32` while quantising the 154 decoder layer projections to `Q4_0` (`1,069,673,472` bytes total).
- **Verified Real `TinyLlama-1.1B-Chat-v1.0` Tensor Quantisation Error (`--report-error`):**
  - `model.layers.0.self_attn.k_proj.weight [256, 2048]` (`524,288` weights): `MAE = 2.592e-03`, `RMSE = 4.085e-03`, `MaxAbsErr = 1.875e-01`, `RelL2 = 12.86%`.
  - `model.layers.0.self_attn.v_proj.weight [256, 2048]` (`524,288` weights): `MAE = 8.170e-04`, `RMSE = 9.958e-04`, `MaxAbsErr = 5.829e-03`, `RelL2 = 9.04%`.
