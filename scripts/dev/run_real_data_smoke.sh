#!/usr/bin/env sh
set -eu

if [ -z "${ARENA_REAL_DATA_ROOT:-}" ]; then
  echo "ARENA_REAL_DATA_ROOT is required." >&2
  exit 1
fi

python scripts/dev/run_real_data_smoke.py
