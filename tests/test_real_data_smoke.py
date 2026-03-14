from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.real_data
def test_real_data_smoke_command() -> None:
    if not os.environ.get("ARENA_REAL_DATA_ROOT"):
        pytest.skip("ARENA_REAL_DATA_ROOT is not set")

    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo_root / "src"), str(repo_root)])}
    result = subprocess.run(
        [sys.executable, "scripts/dev/run_real_data_smoke.py"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
