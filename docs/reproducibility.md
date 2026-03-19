# Public Reproducibility (Smoke Sample)

## Purpose

This document defines **public smoke reproducibility for the release layer**.
The smoke sample provides deterministic, lightweight reproducibility checks for the ARENA public release layer.
It is designed to validate:

- command execution
- release-layer flow wiring
- artifact generation
- freeze/verify workflow

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

Run from repository root (`E:\arena_release`).

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

## Notes on Determinism

- Input generation uses a fixed seed by default.
- Input file mtimes are fixed to a constant epoch.
- The freeze/verify flow runs `artifact run` in deterministic legacy mode.
- Output comparison uses explicit normalization for absolute-path and zip metadata stability.
- `arena.lib.platform_setup` contains environment-dependent GPU/CUDA branches.
  Public tests cover deterministic CPU/fallback paths; hardware-specific paths are intentionally limited.

## Release Checklist

- `pytest` passes in this repository tree.
- Coverage boundary is fixed to this release tree (`.coveragerc`: `src/arena`, `scripts`).
- Coverage total remains at or above the release target (currently 80%+).
- README Quickstart commands are copy-paste verified.
- Smoke sample flow passes:
  `build_public_sample.py` -> `freeze_expected_outputs.py` -> `verify_sample_outputs.py`.
- Import boundary check confirms local resolution:
  `arena.__file__` / `arena.artifacts.__file__` point to this repository.
- `CHANGELOG.md` includes release-layer isolation and reproducibility updates.

## README Link Snippet

Recommended short pointer text:

> For public reproducibility checks (synthetic smoke sample, freeze, verify), see `docs/reproducibility.md`.
