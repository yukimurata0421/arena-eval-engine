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


def _require_env_path(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise SystemExit(f"{name} is required")
    path = Path(raw)
    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    return path


def _count_matching_files(root: Path, patterns: list[str]) -> int:
    count = 0
    for pattern in patterns:
        count += sum(1 for _ in root.rglob(pattern))
    return count


def main() -> int:
    real_data_root = _require_env_path("ARENA_REAL_DATA_ROOT")
    artifact_base = Path(os.environ.get("ARENA_REAL_ARTIFACT_BASE", str(ROOT / "output" / "sample")))
    output_root = Path(
        os.environ.get("ARENA_REAL_OUTPUT_ROOT", str(ROOT / ".tmp_real_data_smoke" / "output"))
    )
    bundle_root = Path(
        os.environ.get("ARENA_REAL_BUNDLE_ROOT", str(ROOT / ".tmp_real_data_smoke" / "bundle"))
    )

    pos_count = _count_matching_files(real_data_root, ["pos_*.jsonl"])
    dist_count = _count_matching_files(real_data_root, ["dist_1m.jsonl", "dist_1m*.jsonl"])
    print(f"[INFO] real_data_root={real_data_root}")
    print(f"[INFO] pos_files={pos_count} dist_files={dist_count}")
    if pos_count == 0 or dist_count == 0:
        raise SystemExit(
            "required input contract check failed: expected at least one pos_*.jsonl and dist_1m*.jsonl"
        )

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(SRC), str(ROOT)]),
        "ARENA_DATA_DIR": str(real_data_root),
        "ARENA_OUTPUT_DIR": str(output_root),
    }

    _run(
        [
            sys.executable,
            "-m",
            "arena.cli",
            "validate",
            "--data-dir",
            str(real_data_root),
            "--output-dir",
            str(output_root),
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
            str(real_data_root),
            "--output-dir",
            str(output_root),
            "--log-jsonl-mode",
            "overwrite",
        ],
        env,
    )

    if not artifact_base.exists():
        raise SystemExit(f"artifact base does not exist: {artifact_base}")
    if artifact_base.resolve() == (ROOT / "output" / "sample").resolve():
        print("[INFO] ARENA_REAL_ARTIFACT_BASE not set; using public sample artifacts for verify/replay.")

    _run(
        [
            sys.executable,
            "scripts/tools/merge_output_for_ai/merge_output_for_ai.py",
            "--base",
            str(artifact_base),
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
