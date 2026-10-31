---
name: Engineering Task
about: Milestone implementation task with explicit ownership and acceptance criteria
title: "[M0/M1/M2/M3/M4] "
labels: ["task"]
assignees: []
---

## Milestone & Owner

- **Milestone:** `M0` / `M1` / `M2` / `M3` / `M4`
- **Owner:** `S` / `I` / `U` / `P`
- **Feature Branch:** `feature/...`

## Objective

Describe the engineering deliverable and how it fits into the `inference-engine` architecture.

## Scope & Files Touched

- `include/inference_engine/...`
- `src/...`
- `tests/...`

## Contract & Invariants

List the architectural or numerical contracts this task must satisfy (e.g., 64-byte payload alignment, explicit KV cache commit, Q4_0 nibble layout, golden reference tolerance).

## Acceptance Criteria

- [ ] Compiles cleanly under C++17 with `-Wall -Wextra -Wpedantic` (and `/W4` on MSVC).
- [ ] Deterministic unit tests added and passing via `ctest --output-on-failure`.
- [ ] CI passes on both Ubuntu x86-64 and macOS ARM64.
- [ ] Reviewed by at least one team member before merging into `integration`.
