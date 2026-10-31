# Hardware Benchmarking Strategy & Methodology

## 1. Target Hardware Matrix

All four team laptops are first-class benchmark targets. All official benchmarks run **CPU-only inference** (discrete GPUs and NPUs disabled/unused).

| Member | Machine & CPU | Memory | Target SIMD Backend |
|---|---|---|---|
| **S** | Apple MacBook Air M4 | 24 GB unified memory | ARM64 `NEON` |
| **I** | AMD Ryzen 7 7745HX | 16 GB DDR5 | x86-64 `AVX2` + `FMA` |
| **U** | Intel Core Ultra 7 155H | 16 GB DDR5 | x86-64 `AVX2` + `FMA` |
| **P** | Intel Core i5-10210U | 8 GB DDR4 | x86-64 `AVX2` (verified via runtime CPU feature detection) |

> **Policy:** Never fabricate, extrapolate, or estimate benchmark numbers. All tables in reports remain blank or marked `Pending M3/M4 measurement` until measured on the actual hardware under controlled conditions.

## 2. Required Metrics

Every official benchmark run must record:

1. **Model Loading Time (ms):** Wall-clock time from `load_from_file()` invocation to ready state.
2. **Time to First Token — TTFT (ms):** Latency to process the full prompt prefill and sample the first generated token.
3. **Prompt Prefill Throughput (tokens/sec):** $\frac{N_{\text{prompt}}}{\text{Prefill Duration (s)}}$.
4. **Autoregressive Decode Throughput (tokens/sec):** $\frac{N_{\text{generated}} - 1}{\text{Decode Duration (s)}}$ (excluding prefill).
5. **Peak Resident Set Size — Peak RSS (MB):** Measured via OS process statistics (`getrusage(RUSAGE_SELF)` on macOS/Linux, `GetProcessMemoryInfo` on Windows).
6. **CPU Utilisation & Thread Scaling:** Speedup curve across `threads = {1, 2, 4, 8, max_physical_cores}`.
7. **Effective Memory Bandwidth (GB/s):** Estimated from bytes of weights + KV cache read per decode token divided by mean token latency.
8. **Thermal / Sustained Stability:** Comparison of decode tokens/sec across a 5-run cold/warm sequence to capture laptop thermal throttling behaviour.

The CLI's `--benchmark` mode currently measures model-load time, prefill TTFT,
prompt throughput, decode throughput, peak resident memory, and process CPU
utilization when supported. Prompt throughput includes the encoded prompt
(including BOS and any chat-template pieces). Decode throughput excludes the
first generated token, which is sampled from the final prompt-position logits;
the generated-token count includes EOS when EOS is sampled. A run with fewer
than two generated tokens reports decode throughput as unavailable.

`--threads` is accepted by the CLI, but the current FP32 `TransformerEngine`
does not yet use the thread pool. It emits a warning for values above one;
those runs must not be presented as thread-scaling results.

## 3. Baseline Comparison Against `llama.cpp`

In Milestone M4, `inference-engine` is benchmarked head-to-head against `llama.cpp` (`llama-cli` / `llama-bench`) on all four laptops under identical conditions:
- Identical model (`TinyLlama-1.1B-Chat-v1.0`) and quantisation (`Q4_0`).
- Identical tokenised prompt lengths (`32`, `128`, `512` prompt tokens) and generation lengths (`128` generated tokens).
- CPU-only execution (`--n-gpu-layers 0` in `llama.cpp`) with matching thread counts (`-t 1, 2, 4, 8`) and greedy decoding (`--temp 0`).
