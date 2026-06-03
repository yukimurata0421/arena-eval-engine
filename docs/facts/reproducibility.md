# Public Reproducibility (Smoke Sample)

## Purpose

This document defines **public smoke reproducibility for the release layer**. The smoke sample
provides deterministic, lightweight reproducibility checks for the ARENA public release layer. It is
designed to validate:

- command execution
- release-layer flow wiring
- artifact generation
- freeze/verify workflow

## Version Scope

- Current source version in this repository is `0.4.0`.
- Reproducibility checks in this document are intended for the current repository state; use
  `CHANGELOG.md` for release-by-release details.

## What It Guarantees

- A fixed synthetic input fixture can be generated deterministically.
- Expected outputs can be frozen from a deterministic public CLI flow.
- Current outputs can be verified against frozen expected outputs with explicit diff reporting.

## What It Does Not Guarantee

- Scientific validity of ADS-B analysis conclusions
- Real-world receiver performance representativeness
- Statistical claims about production telemetry

This sample is for release reproducibility, not research validity.

## Two Reproducibility Layers

- Public smoke reproducibility:
command execution, release-layer flow wiring, artifact generation, freeze/verify checks.
- Research/statistical reproducibility:
requires real data scope, analysis windows, domain assumptions, and statistical controls.

## Quickstart

Run from the repository root (`<repo-root>`).

### 1) Build deterministic smoke input

```powershell
python scripts/tools/sample_data/build_public_sample.py
```

### 2) Freeze expected outputs

```powershell
python scripts/tools/sample_data/freeze_expected_outputs.py --force
```

### 3) Verify against frozen expected outputs

```powershell
python scripts/tools/sample_data/verify_sample_outputs.py
```

Exit code is non-zero when verification fails.

## Synthesis Sample Validation

You can also validate the synthesis pipeline with public fixture data:

```powershell
python -m arena.cli synthesis run `
  --path sample_data/synthesis/raw `
  --db ./tmp/synthesis-smoke/synthesis.sqlite3 `
  --enriched-dir ./tmp/synthesis-smoke/enriched `
  --review-dir ./tmp/synthesis-smoke/review `
  --raw-original-dir ./tmp/synthesis-smoke/raw_original `
  --raw-repaired-dir ./tmp/synthesis-smoke/raw_repaired `
  --repair-log-dir ./tmp/synthesis-smoke/repair_logs
```

## Notes on Determinism

- Input generation uses a fixed seed by default.
- Input file mtimes are fixed to a constant epoch.
- The freeze/verify flow runs `artifact run` in deterministic legacy mode.
- Output comparison uses explicit normalization for absolute-path and zip metadata stability.
- `manifest.normalized.csv` is part of versioned expected outputs to verify file-inventory and
  normalized-path stability explicitly.
- `arena.lib.platform_setup` contains environment-dependent GPU/CUDA branches.
Public tests cover deterministic CPU/fallback paths; hardware-specific paths are intentionally
limited.

## GitHub Actions Integration

- `verify-smoke` workflow (`.github/workflows/verify-smoke.yml`):
lightweight release-layer smoke verification on `push`, `pull_request`, and `workflow_dispatch`. It
runs deterministic sample build in CI temp storage and verifies against repository-fixed expected
outputs in `sample_data/smoke/expected`. Expected outputs are repository-fixed
(`sample_data/smoke/expected`), while CI temp build validates generator behavior. It intentionally
does not run freeze in normal CI.
- `artifacts-verify-replay` workflow (`.github/workflows/artifacts-verify-replay.yml`):
heavier manual audit/revalidation path on `workflow_dispatch` only. It generates an artifact bundle,
runs `arena artifacts verify`, runs `arena artifacts replay`, and uploads logs plus bundle outputs.

Both workflows are release-layer checks. They do not establish research/statistical reproducibility.

## Release Checklist

- `pytest` passes in this repository tree.
- Coverage boundary is fixed to this release tree (`.coveragerc`: `src/arena`, `scripts`).
- Coverage total remains at or above the release target (currently 86%+).
- README Quickstart commands are copy-paste verified.
- Smoke sample flow passes:
`build_public_sample.py` -> `freeze_expected_outputs.py` -> `verify_sample_outputs.py`.
- Import boundary check confirms local resolution:
`arena.__file__` / `arena.artifacts.__file__` point to this repository.
- `CHANGELOG.md` includes release-layer isolation and reproducibility updates.

## README Link Snippet

Recommended short pointer text:

> For public reproducibility checks (synthetic smoke sample, freeze, verify), see
`docs/facts/reproducibility.md`.

