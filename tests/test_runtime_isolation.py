from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_plain_python_resolves_arena_from_release_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import arena, arena.artifacts, pathlib;"
                "print(pathlib.Path(arena.__file__).resolve());"
                "print(pathlib.Path(arena.artifacts.__file__).resolve())"
            ),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, f"unexpected subprocess output: {proc.stdout!r}"
    arena_file = Path(lines[0])
    artifacts_file = Path(lines[1])
    assert repo_root in arena_file.parents, f"unexpected arena import path: {arena_file}"
    assert repo_root in artifacts_file.parents, f"unexpected arena.artifacts import path: {artifacts_file}"
