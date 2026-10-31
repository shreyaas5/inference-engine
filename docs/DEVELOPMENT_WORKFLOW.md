# Development & Git Workflow

## 1. Branch Architecture

- `main` — Protected, stable branch containing only validated milestone checkpoints.
- `integration` — Team integration branch where feature branches are merged and cross-module tests are validated.
- Feature branches:
  - `feature/model-loader` (Owner: **S**)
  - `feature/quantisation-simd` (Owner: **I**)
  - `feature/transformer` (Owner: **U**)
  - `feature/runtime-tokenizer` (Owner: **P**)

## 2. Engineering Rules

1. **Never commit directly to `main`.**
2. **Pull Request Workflow:** All feature work is developed on a `feature/*` branch and opened as a Pull Request targeting `integration`.
3. **CI Gate:** Every PR must pass the GitHub Actions CI matrix (`.github/workflows/ci.yml` on Ubuntu x86-64 and macOS ARM64) and `ctest --output-on-failure` before merge.
4. **Peer Code Review:** Every PR requires review and approval from at least one other team member.
5. **Milestone Checkpoints:** `integration` is merged into `main` only when a milestone (`m0-foundation`, `m1-fp32`, `m2-q4`, `m3-optimised`) passes all acceptance tests.
6. **Conventional Commits:** Use standard prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`, `chore:`).
7. **No Weights in Git:** Never commit `.ie`, `.safetensors`, `.bin`, `.gguf`, or `.model` files.

## 3. Standard Build & Test Commands

```bash
# Configure Debug or Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile library, CLI executable, and unit tests
cmake --build build --parallel 4

# Run all CTest unit suites
ctest --test-dir build --output-on-failure

# Run CLI smoke check
./build/inference-engine --info
```

## 4. Inviting Collaborators on GitHub

To avoid inventing GitHub usernames during repository initialisation, the repository administrator (**S**, `shreyaas5`) can invite team members **I**, **U**, and **P** via GitHub CLI once their exact GitHub handles are confirmed:

```bash
gh api --method PUT repos/shreyaas5/inference-engine/collaborators/<github-username> -f permission=push
```
