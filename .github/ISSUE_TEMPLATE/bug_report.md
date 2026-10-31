---
name: Bug Report
about: Report a correctness bug, numerical discrepancy, memory error, or build failure
title: "[BUG] "
labels: ["bug"]
assignees: []
---

## Summary

Provide a clear and concise description of the bug.

## Component

- [ ] Loader / `.ie` Binary Format (`src/loader/`, `tools/`)
- [ ] Q4_0 Quantisation / Dequantisation (`src/kernels/quant_q4_0.cpp`)
- [ ] Scalar Kernels (`src/kernels/scalar/`)
- [ ] SIMD Kernels (`src/kernels/neon/`, `src/kernels/avx2/`)
- [ ] Thread Pool / Dispatch (`src/kernels/threading/`, `src/kernels/dispatch.cpp`)
- [ ] KV Cache / Transformer Forward (`src/core/`, `src/transformer/`)
- [ ] Tokenizer / Sampler / CLI (`src/tokenizer/`, `src/sampler/`, `src/app/`)
- [ ] Build / CI (`CMakeLists.txt`, `.github/workflows/ci.yml`)

## Hardware & Environment

- Machine / Member Profile (S / I / U / P):
- OS & Architecture:
- Compiler & Version:
- SIMD Backend (`Scalar` / `NEON` / `AVX2`):

## Steps to Reproduce

1.
2.
3.

## Expected vs Actual Behaviour

- **Expected:**
- **Actual:**

## Numerical / Diagnostic Output

```text
Paste relevant CTest output, error logs, or tensor divergence metrics here.
```
