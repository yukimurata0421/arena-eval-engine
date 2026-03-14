$ErrorActionPreference = "Stop"

if (-not $env:ARENA_REAL_DATA_ROOT) {
    throw "ARENA_REAL_DATA_ROOT is required."
}

python scripts/dev/run_real_data_smoke.py
