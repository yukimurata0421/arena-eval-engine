from __future__ import annotations

from pathlib import Path

from scripts.tools.sample_data.build_public_sample import main as build_main
from scripts.tools.sample_data.freeze_expected_outputs import main as freeze_main
from scripts.tools.sample_data.verify_sample_outputs import main as verify_main


def test_public_sample_verify_succeeds_when_expected_matches(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    assert build_main(["--sample-root", str(sample_root)]) == 0
    assert freeze_main(["--sample-root", str(sample_root)]) == 0
    assert verify_main(["--sample-root", str(sample_root)]) == 0


def test_public_sample_verify_fails_on_difference(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    assert build_main(["--sample-root", str(sample_root)]) == 0
    assert freeze_main(["--sample-root", str(sample_root)]) == 0

    expected_file = sample_root / "expected" / "manifest.normalized.csv"
    expected_file.write_text(expected_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    assert verify_main(["--sample-root", str(sample_root)]) == 1

