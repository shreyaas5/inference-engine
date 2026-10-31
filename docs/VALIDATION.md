# Correctness & Golden Validation Strategy

## 1. Golden Reference Generation (`tools/generate_golden_reference.py`)

To validate `inference-engine` without ambiguity, reference outputs are generated offline using Hugging Face `transformers` and `torch` under strict precision alignment:

1. **BF16-to-FP32 Weight Casting:**
   The upstream `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint stores weights in `bfloat16`. Before running either the Hugging Face reference model or exporting the `.ie` FP32 model, all checkpoint weights are cast once to `float32` (`torch.float32`). Both runtimes therefore operate on bit-identical FP32 weights.
2. **Intermediate Tensor Probes:**
   For canonical multi-token prompts, we capture:
   - Embedding lookup output (`pos = 0..N-1`)
   - Layer `0`, `11`, and `21` pre-attention RMSNorm outputs
   - Layer `0`, `11`, and `21` Q and K vectors before and after RoPE (specifically testing nonzero positions `pos > 0`)
   - Layer `0`, `11`, and `21` GQA attention outputs
   - Layer `0`, `11`, and `21` SwiGLU MLP outputs
   - Final RMSNorm output and full `[32000]` vocabulary logits

## 2. Multi-Token & Autoregressive Greedy Validation

Validation is performed across three progressive stages:

1. **Single-Token Position 0 Parity:**
   Verifies embedding lookup, RMSNorm, QKV matmul, SwiGLU, and LM head at `pos = 0`.
2. **Multi-Token Prompt Prefill (`pos = 0..N-1`):**
   Verifies nonzero RoPE rotation angles, causal masking ($t \le \text{pos}$), GQA head mapping (`32` Q heads to `4` KV heads), and KV cache `append()` + `commit_token()` semantics.
3. **Autoregressive Greedy Continuation:**
   Starting from a 12-token prompt (`pos = 0..11`), verifies that:
   - Logits at `pos = 11` select the exact same greedy argmax token ID as Hugging Face.
   - Feeding generated tokens at `pos = 12, 13, ...` maintains top-1 token agreement across greedy continuation steps.

## 3. Empirical Tolerance Calibration

Rather than assuming arbitrary fixed floating-point thresholds upfront:
- **FP32 Scalar vs FP32 Reference:** We measure maximum absolute error ($\max_i |x_i - y_i|$), mean absolute error, and cosine similarity at each layer depth (`0..21`) caused by FMA contraction and reduction ordering, and record the empirical bounds in our validation logs.
- **SIMD (NEON / AVX2) vs Scalar:** Tested per-kernel on identical inputs (`tests/test_kernels_simd.cpp`) to verify that vector horizontal reductions stay within tight FP32 rounding bounds ($\sim 10^{-5}$).
- **Q4_0 vs FP32:** Evaluated using logit RMSE, cosine similarity, top-1/top-5 logit agreement rate, and greedy continuation stability.
