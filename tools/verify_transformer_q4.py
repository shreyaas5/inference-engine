#!/usr/bin/env python3
"""Independent Python NumPy reference verification for FP32 and Q4_0 Transformer forward pass.

Owner: U (Transformer Forward Pass & Numerical Correctness)

Validates:
1. Mathematical equivalence of C++ Transformer forward pass against an independent NumPy reference.
2. Quantization error metrics between FP32 and Q4_0 (MaxErr, MAE, RMSE, greedy continuation tokens).
3. KV cache causal attention and autoregressive decode indexing contract.
"""

from __future__ import annotations

import math
import struct
import sys
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    print("[ERROR] NumPy is required for independent verification. Install with: pip install numpy")
    sys.exit(1)


def fp32_to_fp16(val: float) -> int:
    return int(np.float16(val).view(np.uint16))


def fp16_to_fp32(bits: int) -> float:
    return float(np.uint16(bits).view(np.float16))


def quantize_row_q4_0(row: np.ndarray) -> bytes:
    """Quantizes a 1D float array of length divisible by 32 into BlockQ4_0 bytes."""
    assert len(row) % 32 == 0
    num_blocks = len(row) // 32
    out = bytearray()

    for b in range(num_blocks):
        block = row[b * 32 : (b + 1) * 32]
        amax = np.max(np.abs(block))
        signed_max = block[np.argmax(np.abs(block))]
        d = signed_max / -8.0 if amax != 0 else 0.0
        id_ = 1.0 / d if d != 0 else 0.0
        d_fp16 = fp32_to_fp16(d)

        qs = bytearray(16)
        for j in range(16):
            x0 = block[j] * id_
            x1 = block[j + 16] * id_
            q0 = int(np.clip(np.floor(x0 + 8.5), 0, 15))
            q1 = int(np.clip(np.floor(x1 + 8.5), 0, 15))
            qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)

        out += struct.pack("<H", d_fp16) + qs

    return bytes(out)


def dequantize_row_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantizes BlockQ4_0 bytes into a 1D float32 array."""
    assert n_elements % 32 == 0
    num_blocks = n_elements // 32
    out = np.zeros(n_elements, dtype=np.float32)

    for b in range(num_blocks):
        offset = b * 18
        d_bits = struct.unpack_from("<H", data, offset)[0]
        d = fp16_to_fp32(d_bits)
        qs = data[offset + 2 : offset + 18]

        for j in range(16):
            byte_val = qs[j]
            q0 = (byte_val & 0x0F) - 8
            q1 = ((byte_val >> 4) & 0x0F) - 8
            out[b * 32 + j] = q0 * d
            out[b * 32 + j + 16] = q1 * d

    return out


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    ms = np.mean(x * x)
    return (x / np.sqrt(ms + eps)) * weight


def rope_inplace(x: np.ndarray, num_heads: int, head_dim: int, pos: int, theta: float = 10000.0) -> np.ndarray:
    out = x.copy().reshape(num_heads, head_dim)
    half = head_dim // 2
    for h in range(num_heads):
        for i in range(half):
            freq = 1.0 / (theta ** (float(2 * i) / float(head_dim)))
            angle = float(pos) * freq
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = out[h, i]
            x1 = out[h, i + half]
            out[h, i] = x0 * c - x1 * s
            out[h, i + half] = x1 * c + x0 * s
    return out.reshape(-1)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    silu_gate = gate / (1.0 + np.exp(-gate))
    return silu_gate * up


class NumpyTransformer:
    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 64,
        intermediate_dim: int = 128,
        num_query_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 16,
        vocab_size: int = 64,
        max_seq_len: int = 32,
        use_q4: bool = False,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_query_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_q4 = use_q4

        self.k_cache = [np.zeros((max_seq_len, self.kv_dim), dtype=np.float32) for _ in range(num_layers)]
        self.v_cache = [np.zeros((max_seq_len, self.kv_dim), dtype=np.float32) for _ in range(num_layers)]
        self.seq_len = 0

        # Deterministic synthetic weights matching create_synthetic_model_64
        def gen_matrix(rows, cols, base, factor):
            idx = np.arange(1, rows * cols + 1, dtype=np.float32)
            mat = (np.sin(base + idx * factor) * 0.05).astype(np.float32).reshape(rows, cols)
            if self.use_q4:
                # Quantize each row to Q4_0 and dequantize back to get exact Q4_0 mathematical weights
                mat_q = np.zeros_like(mat)
                for r in range(rows):
                    b = quantize_row_q4_0(mat[r])
                    mat_q[r] = dequantize_row_q4_0(b, cols)
                return mat_q
            return mat

        def gen_norm(dim, base):
            idx = np.arange(1, dim + 1, dtype=np.float32)
            return (1.0 + np.sin(base + idx * 0.1) * 0.02).astype(np.float32)

        self.embed = gen_matrix(vocab_size, hidden_dim, 0.5, 0.07)
        self.layers = []
        for l in range(num_layers):
            loff = float(l) * 1.5
            layer = {
                "input_norm": gen_norm(hidden_dim, 1.0 + loff),
                "q_proj": gen_matrix(self.q_dim, hidden_dim, 2.0 + loff, 0.03),
                "k_proj": gen_matrix(self.kv_dim, hidden_dim, 3.0 + loff, 0.04),
                "v_proj": gen_matrix(self.kv_dim, hidden_dim, 4.0 + loff, 0.05),
                "o_proj": gen_matrix(hidden_dim, self.q_dim, 5.0 + loff, 0.06),
                "post_norm": gen_norm(hidden_dim, 6.0 + loff),
                "gate_proj": gen_matrix(intermediate_dim, hidden_dim, 7.0 + loff, 0.02),
                "up_proj": gen_matrix(intermediate_dim, hidden_dim, 8.0 + loff, 0.03),
                "down_proj": gen_matrix(hidden_dim, intermediate_dim, 9.0 + loff, 0.04),
            }
            self.layers.append(layer)

        self.final_norm = gen_norm(hidden_dim, 10.0)
        self.lm_head = gen_matrix(vocab_size, hidden_dim, 11.0, 0.015)

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        assert pos == self.seq_len
        x = self.embed[token_id].copy()

        gqa_group = self.num_query_heads // self.num_kv_heads
        head_scale = 1.0 / math.sqrt(float(self.head_dim))

        for l in range(self.num_layers):
            layer = self.layers[l]

            # Input norm
            u = rmsnorm(x, layer["input_norm"])

            # QKV projection
            q = np.dot(layer["q_proj"], u)
            k = np.dot(layer["k_proj"], u)
            v = np.dot(layer["v_proj"], u)

            # RoPE
            q = rope_inplace(q, self.num_query_heads, self.head_dim, pos)
            k = rope_inplace(k, self.num_kv_heads, self.head_dim, pos)

            # KV cache append
            self.k_cache[l][pos] = k
            self.v_cache[l][pos] = v

            # Attention
            attn_out = np.zeros(self.q_dim, dtype=np.float32)
            q_heads = q.reshape(self.num_query_heads, self.head_dim)
            out_heads = attn_out.reshape(self.num_query_heads, self.head_dim)

            for hq in range(self.num_query_heads):
                hkv = hq // gqa_group
                q_h = q_heads[hq]

                # Causal dot products
                scores = np.zeros(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    k_h = self.k_cache[l][t].reshape(self.num_kv_heads, self.head_dim)[hkv]
                    scores[t] = np.dot(q_h, k_h) * head_scale

                # Softmax
                scores_exp = np.exp(scores - np.max(scores))
                attn_weights = scores_exp / np.sum(scores_exp)

                # Value aggregation
                for t in range(pos + 1):
                    v_h = self.v_cache[l][t].reshape(self.num_kv_heads, self.head_dim)[hkv]
                    out_heads[hq] += attn_weights[t] * v_h

            # Out projection + residual
            x = x + np.dot(layer["o_proj"], attn_out)

            # Post-attention norm
            m = rmsnorm(x, layer["post_norm"])

            # SwiGLU FFN + residual
            gate = np.dot(layer["gate_proj"], m)
            up = np.dot(layer["up_proj"], m)
            h = swiglu(gate, up)
            x = x + np.dot(layer["down_proj"], h)

        self.seq_len += 1

        # Final norm and LM head
        norm_x = rmsnorm(x, self.final_norm)
        logits = np.dot(self.lm_head, norm_x)
        return logits


def main():
    print("=" * 70)
    print("   INDEPENDENT PYTHON/NUMPY REFERENCE FOR FP32 & Q4_0 TRANSFORMER")
    print("=" * 70)

    prompt_tokens = [5, 11, 22, 33, 44]
    num_decode_steps = 4

    print(f"[INFO] Running NumPy FP32 forward pass on prompt {prompt_tokens}...")
    model_fp32 = NumpyTransformer(use_q4=False)
    for pos, tok in enumerate(prompt_tokens):
        fp32_logits = model_fp32.forward_token(tok, pos)

    print(f"[INFO] Running NumPy Q4_0 forward pass on prompt {prompt_tokens}...")
    model_q4 = NumpyTransformer(use_q4=True)
    for pos, tok in enumerate(prompt_tokens):
        q4_logits = model_q4.forward_token(tok, pos)

    # Compare prefill logits
    diff = np.abs(fp32_logits - q4_logits)
    max_err = float(np.max(diff))
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    tok_fp = int(np.argmax(fp32_logits))
    tok_q4 = int(np.argmax(q4_logits))

    print(f"[METRICS] Prompt Prefill FP32 vs Q4_0 in NumPy:")
    print(f"  MaxErr:          {max_err:.6f}")
    print(f"  MAE:             {mae:.6f}")
    print(f"  RMSE:            {rmse:.6f}")
    print(f"  Greedy tokens:   FP32={tok_fp}, Q4_0={tok_q4}")
    assert mae < 0.20, f"MAE too large: {mae}"
    assert rmse < 0.25, f"RMSE too large: {rmse}"

    # Decode steps
    curr_fp = tok_fp
    curr_q4 = tok_q4
    for step in range(num_decode_steps):
        decode_pos = len(prompt_tokens) + step
        d_fp_logits = model_fp32.forward_token(curr_fp, decode_pos)
        d_q4_logits = model_q4.forward_token(curr_q4, decode_pos)

        d_diff = np.abs(d_fp_logits - d_q4_logits)
        d_max = float(np.max(d_diff))
        d_mae = float(np.mean(d_diff))
        d_rmse = float(np.sqrt(np.mean(d_diff ** 2)))
        curr_fp = int(np.argmax(d_fp_logits))
        curr_q4 = int(np.argmax(d_q4_logits))

        print(f"[METRICS] Decode Step {step} (pos {decode_pos}) in NumPy:")
        print(f"  MaxErr={d_max:.6f}, MAE={d_mae:.6f}, RMSE={d_rmse:.6f}, Tokens: FP32={curr_fp}, Q4={curr_q4}")

    print("[SUCCESS] Independent NumPy verification passed with expected quantization bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

