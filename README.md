# ARENA Public Release

ARENA is a reproducible ADS-B evaluation toolkit.
This repository contains the public execution layer for pipeline orchestration, CLI operation, and evaluation scripts.

## Highlights
- 8-stage pipeline (`arena.pipeline`) for aggregation, evaluation, reporting, and comparisons.
- Failure-resilient execution model: stage-level continuation and explicit error reporting.
- Append-only JSONL execution logging (`output/performance/pipeline_runs.jsonl`) for auditability.
- `PipelineBuildOptions` for environment-separated pipeline construction (feature toggles and optional stages).
- CLI-first operation (`python -m arena.cli ...`) with deterministic path/env resolution.

## Repository Layout
- `src/arena/`: core CLI, pipeline, and shared runtime modules.
- `scripts/`: stage implementation scripts and compatibility shims.
- `tests/`: pytest-based contract and regression tests.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev]
```

## Environment Setup
Use one of the following setup levels:

1. Test/validation setup (minimum public baseline):
```bash
pip install -e .[dev]
```
2. Full Stage 1-8 runtime setup (adds Stage 4/5 probabilistic stack):
```bash
pip install -e .[dev]
pip install "jax>=0.4.30" "jaxlib>=0.4.30" "numpyro>=0.15.0" "pymc>=5.0.0" "arviz>=0.17.0"
```

Sanity checks:
```bash
python -m arena.cli --help
python -m arena.cli validate
pytest
coverage run -m pytest -q && coverage report -m
```

## Windows / WSL Quick Setup
Windows PowerShell (minimum public baseline):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[dev]
python -m arena.cli --help
python -m arena.cli validate
pytest
```

WSL (Ubuntu, full Stage 1-8 runtime):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
pip install "jax>=0.4.30" "jaxlib>=0.4.30" "numpyro>=0.15.0" "pymc>=5.0.0" "arviz>=0.17.0"
python -m arena.cli --help
python -m arena.cli validate
pytest
coverage run -m pytest -q && coverage report -m
```

## Runtime Environment Variables
Set these when running outside defaults:
- `ARENA_SCRIPTS_ROOT`
- `ARENA_DATA_DIR`
- `ARENA_OUTPUT_DIR`
- `ARENA_SETTINGS`
- `ARENA_PHASE_CONFIG`

OpenSky integration credentials (optional, only when fetching traffic data):
- `OPENSKY_CLIENT_ID`
- `OPENSKY_CLIENT_SECRET`

## CLI Examples
```bash
python -m arena.cli --help
python -m arena.cli validate
python -m arena.cli run --only 1 --no-gpu
```

Optional overrides:
```bash
python -m arena.cli run --only 1 --no-gpu \
  --data-dir ./data \
  --output-dir ./output \
  --settings ./scripts/config/settings.toml
```

## Testing
```bash
pytest
```

## Docker
```bash
docker compose -f docker/docker-compose.yml run --rm arena-tests
docker compose -f docker/docker-compose.yml run --rm arena-validate
```

Real-data Stage 1 / full pipeline execution (opt-in) is documented in `docs/real-data-smoke.md`.
It uses host-mounted paths and keeps private data/credentials outside this repository.

## Clean Public ZIP
Build a distribution zip that excludes VCS/caches/build artifacts:
```bash
python scripts/tools/build_public_zip.py --output arena_public_source.zip
```
The generated zip excludes at least:
- `.git/`
- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `*.egg-info/`
- `.coverage`
- `htmlcov/`
- `build/`
- `dist/`
- `output/`
- `scripts/tools/debug/tmp_*`
- `scripts/structure`

## Notes for Public Usage
- Secrets, credential files, and private runtime artifacts are intentionally excluded.
- Absolute local paths are not required; use environment variables or repo-relative defaults.
- Generated data/log/output directories are not part of this release baseline.
- `scripts/tools/debug/` contains developer diagnostics; `tmp_*` debug helpers are excluded from public zip artifacts.
