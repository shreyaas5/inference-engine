# Test Fixtures Directory (`fixtures/`)

This directory holds small, deterministic synthetic test inputs or local validation metadata used during development and testing.

## Rules

1. **Never commit model checkpoints or large binary artifacts.**
   Files matching `*.ie`, `*.bin`, `*.safetensors`, `*.pt`, `*.pth`, `*.gguf`, `*.model`, `*.npy`, and `*.npz` are excluded via [`.gitignore`](../.gitignore).
2. **CI Independence.**
   Continuous Integration (`.github/workflows/ci.yml`) must never download the 2.2 GB `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint. All CI unit tests generate synthetic in-memory `.ie` images and deterministic tensors directly inside C++ test suites.
3. **Local Golden Fixtures.**
   When running offline golden validation locally (`tools/generate_golden_reference.py`), place generated reference outputs under `fixtures/golden/` (which is git-ignored).
