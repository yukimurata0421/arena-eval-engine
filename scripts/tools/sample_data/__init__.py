from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

DEFAULT_SAMPLE_ROOT = ROOT / "sample_data" / "smoke"
DEFAULT_SAMPLE_NAME = "arena_public_smoke"
DEFAULT_SAMPLE_TYPE = "smoke"
DEFAULT_MODE = "synthetic"
SUPPORTED_MODES = ("synthetic", "from-real")
DEFAULT_RANDOM_SEED = 20260320
DEFAULT_GENERATION_TIMESTAMP = "2026-01-01T00:00:00Z"
DEFAULT_FIXED_MTIME_EPOCH = 1704067200

MANIFEST_NAME = "manifest.json"
INPUT_DIRNAME = "input"
EXPECTED_DIRNAME = "expected"

