from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.tools.sample_data.build_public_sample import main as build_main


def _tree_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def test_public_sample_build_creates_manifest_and_inputs(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    rc = build_main(["--sample-root", str(sample_root)])
    assert rc == 0

    manifest_path = sample_root / "manifest.json"
    input_dir = sample_root / "input"
    assert manifest_path.exists()
    assert input_dir.exists()
    assert (input_dir / "daily_metrics.csv").exists()
    assert (input_dir / "signal_events.jsonl").exists()
    assert (input_dir / "README.txt").exists()


def test_public_sample_build_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    sample_a = tmp_path / "sample_a"
    sample_b = tmp_path / "sample_b"
    argv = [
        "--seed",
        "7",
        "--generation-timestamp",
        "2026-01-01T00:00:00Z",
    ]
    rc_a = build_main(["--sample-root", str(sample_a), *argv])
    rc_b = build_main(["--sample-root", str(sample_b), *argv])
    assert rc_a == 0
    assert rc_b == 0

    assert _tree_hash(sample_a) == _tree_hash(sample_b)

