from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(ROOT)])}
    sample_data_dir = ROOT / "data"
    sample_output_dir = ROOT / "output"
    bundle_root = ROOT / ".tmp_sample_smoke" / "artifacts"

    _run(
        [
            sys.executable,
            "-m",
            "arena.cli",
            "validate",
            "--data-dir",
            str(sample_data_dir),
            "--output-dir",
            str(sample_output_dir),
            "--create-dirs",
        ],
        env,
    )
    _run(
        [
            sys.executable,
            "-m",
            "arena.cli",
            "run",
            "--only",
            "1",
            "--dry-run",
            "--no-gpu",
            "--skip-plao",
            "--backend",
            "native",
            "--data-dir",
            str(sample_data_dir),
            "--output-dir",
            str(sample_output_dir),
            "--log-jsonl-mode",
            "overwrite",
        ],
        env,
    )
    _run(
        [
            sys.executable,
            "scripts/tools/merge_output_for_ai/merge_output_for_ai.py",
            "--base",
            str(ROOT / "output" / "sample"),
            "--out",
            str(bundle_root),
            "--export-ai-folder",
            "--deterministic",
        ],
        env,
    )

    bundles = sorted(bundle_root.glob("_ai_review_*"))
    if not bundles:
        raise SystemExit("artifact bundle was not generated")
    bundle_dir = bundles[-1]

    _run([sys.executable, "-m", "arena.cli", "artifacts", "verify", str(bundle_dir)], env)
    _run([sys.executable, "-m", "arena.cli", "artifacts", "replay", str(bundle_dir)], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
