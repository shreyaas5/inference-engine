# End-to-End Inference Data Flow & Position Indexing Contract

## 1. Forward Pass Data Flow (Single Token at Position `pos`)

Given input `token_id` $\in [0, 31999]$ and explicit sequence position `pos` $\in [0, 2047]$:

1. **Token Embedding Lookup:**
   - Copy row `token_id` from `model.embed_tokens.weight` into activation vector $x \in \mathbb{R}^{2048}$ (`FP32`).
2. **Decoder Layer Loop (`layer_idx = 0 .. 21`):**
   - **Pre-Attention RMSNorm:**
     $$u = \text{RMSNorm}(x, w_{\text{input\_layernorm}}, \epsilon = 10^{-5})$$
   - **QKV Linear Projections:**
     - $q = W_q u \in \mathbb{R}^{2048}$ (`32` query heads $\times$ `64` head_dim)
     - $k = W_k u \in \mathbb{R}^{256}$ (`4` KV heads $\times$ `64` head_dim)
     - $v = W_v u \in \mathbb{R}^{256}$ (`4` KV heads $\times$ `64` head_dim)
   - **Rotary Position Embedding (RoPE):**
     - Apply in-place RoPE rotation at position `pos` ($\theta = 10000.0$) to all 32 query heads in $q$ and all 4 key heads in $k$.
   - **KV Cache Write (`append`):**
     - Call `kv_cache.append(layer_idx, pos, k, v)`.
     - **Contract:** `append()` writes $k, v$ for `(layer_idx, pos)` and **does NOT advance** `kv_cache.seq_len()`.
   - **Causal Grouped Query Attention (GQA):**
     - Each KV head $h_{kv} \in [0, 3]$ serves 8 query heads $h_q \in [8 h_{kv}, 8 h_{kv} + 7]$.
     - For each query head $h_q$, compute scaled dot-product attention scores over cached keys at positions $t \in [0, \text{pos}]$:
       $$s_t = \frac{\langle q_{h_q}, k_{h_{kv}, t} \rangle}{\sqrt{64}}$$
     - Apply numerically stable `softmax_f32` over $s_{0..\text{pos}}$ and accumulate weighted values $v_{h_{kv}, t}$.
   - **Output Projection & First Residual Connection:**
     $$x \leftarrow x + W_o a$$
   - **Pre-MLP RMSNorm:**
     $$m = \text{RMSNorm}(x, w_{\text{post\_attention\_layernorm}}, \epsilon = 10^{-5})$$
   - **SwiGLU Feed-Forward Network & Second Residual Connection:**
     $$g = W_{\text{gate}} m \in \mathbb{R}^{5632}, \quad p = W_{\text{up}} m \in \mathbb{R}^{5632}$$
     $$h = \text{SiLU}(g) \odot p$$
     $$x \leftarrow x + W_{\text{down}} h$$
3. **KV Cache Commit (`commit_token`):**
   - Only after all 22 layers have completed for `pos`, call `kv_cache.commit_token()` to advance `seq_len()` from `pos` to `pos + 1`.
4. **Final RMSNorm & LM Head:**
   - $x_{\text{final}} = \text{RMSNorm}(x, w_{\text{final\_norm}}, \epsilon = 10^{-5})$
   - $\text{logits} = W_{\text{lm\_head}} x_{\text{final}} \in \mathbb{R}^{32000}$

## 2. Autoregressive Decoding & Position Indexing Contract

Careful position bookkeeping is mandatory to avoid off-by-one RoPE and KV cache bugs.

Consider a **12-token prompt** $(t_0, t_1, \dots, t_{11})$:

| Phase | Step | Input Token | RoPE / KV Position `pos` | `kv_cache.seq_len()` Before Commit | `kv_cache.seq_len()` After Commit | Output Logits Usage |
|---|---|---|---|---|---|---|
| Prefill | 0 | $t_0$ | `0` | `0` | `1` | Discarded (or used for perplexity) |
| Prefill | 1 | $t_1$ | `1` | `1` | `2` | Discarded |
| ... | ... | ... | ... | ... | ... | ... |
| Prefill | 11 | $t_{11}$ | `11` | `11` | `12` | **Predicts first generated token $\hat{t}_{12}$** |
| Decode | 0 | $\hat{t}_{12}$ | `12` | `12` | `13` | Predicts second generated token $\hat{t}_{13}$ |
| Decode | 1 | $\hat{t}_{13}$ | `13` | `13` | `14` | Predicts third generated token $\hat{t}_{14}$ |

Key invariants:
- A 12-token prompt occupies positions `0..11`.
- The logits computed at position `11` (the final prompt token) are sampled to produce the **first generated token** $\hat{t}_{12}$.
- $\hat{t}_{12}$ is never re-evaluated at position `11`; when decoding continues, $\hat{t}_{12}$ is processed at **position `12`**, followed by position `13`, `14`, and so on.
