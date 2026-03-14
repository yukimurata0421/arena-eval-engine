# Real-Data Smoke Validation

The public repository keeps real-data validation opt-in.

CI uses only `data/sample/` and `output/sample/`.
Local validation can point to private datasets without hardcoding host paths.

## Required Environment

- `ARENA_REAL_DATA_ROOT`: root directory containing private telemetry such as `pos_*.jsonl` and `dist_1m*.jsonl`

Optional:

- `ARENA_REAL_OUTPUT_ROOT`: scratch output directory for dry-run validation
- `ARENA_REAL_ARTIFACT_BASE`: existing output directory to export/verify instead of `output/sample`
- `ARENA_REAL_BUNDLE_ROOT`: where the temporary artifact bundle is written

## PowerShell

```powershell
$env:ARENA_REAL_DATA_ROOT="E:\arena\data"
$env:ARENA_REAL_ARTIFACT_BASE="E:\arena\output"
python scripts/dev/run_real_data_smoke.py
```

or

```powershell
$env:ARENA_REAL_DATA_ROOT="E:\arena\data"
scripts/dev/run_real_data_smoke.ps1
```

## POSIX Shell

```bash
export ARENA_REAL_DATA_ROOT=/workspace/data
export ARENA_REAL_ARTIFACT_BASE=/workspace/output
python scripts/dev/run_real_data_smoke.py
```

## What It Checks

- path resolution through public CLI and runtime settings
- required input contract visibility for `pos_*.jsonl` and `dist_1m*.jsonl`
- minimal pipeline dry-run against the private data root
- artifact export, verify, and replay flows

If `ARENA_REAL_ARTIFACT_BASE` is not set, artifact verify/replay falls back to `output/sample/`.
That still exercises the public artifact contract, but it does not validate private output contents.
