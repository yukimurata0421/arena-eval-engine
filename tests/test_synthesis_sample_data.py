from __future__ import annotations

from pathlib import Path

from arena.synthesis import cli as synthesis_cli


def test_synthesis_sample_data_end_to_end(tmp_path: Path) -> None:
    sample_root = Path("sample_data") / "synthesis" / "raw"
    db_path = tmp_path / "synthesis.sqlite3"
    enriched_dir = tmp_path / "enriched"
    review_dir = tmp_path / "review"
    raw_original_dir = tmp_path / "raw_original"
    raw_repaired_dir = tmp_path / "raw_repaired"
    repair_log_dir = tmp_path / "repair_logs"

    rc = synthesis_cli.main(
        [
            "run",
            "--path",
            str(sample_root),
            "--db",
            str(db_path),
            "--enriched-dir",
            str(enriched_dir),
            "--review-dir",
            str(review_dir),
            "--raw-original-dir",
            str(raw_original_dir),
            "--raw-repaired-dir",
            str(raw_repaired_dir),
            "--repair-log-dir",
            str(repair_log_dir),
        ]
    )

    assert rc == 0
    assert db_path.exists()
    assert (enriched_dir / "claude.json").exists()
    assert (enriched_dir / "gpt.json").exists()
    assert (review_dir / "triage").exists()
    assert (review_dir / "queue").exists()


def test_synthesis_sample_data_ingest_only_with_db_override(tmp_path: Path) -> None:
    sample_root = Path("sample_data") / "synthesis" / "raw"
    db_path = tmp_path / "ingest.sqlite3"
    raw_original_dir = tmp_path / "raw_original"
    raw_repaired_dir = tmp_path / "raw_repaired"
    repair_log_dir = tmp_path / "repair_logs"

    rc = synthesis_cli.main(
        [
            "ingest",
            "--path",
            str(sample_root),
            "--db",
            str(db_path),
            "--raw-original-dir",
            str(raw_original_dir),
            "--raw-repaired-dir",
            str(raw_repaired_dir),
            "--repair-log-dir",
            str(repair_log_dir),
        ]
    )

    assert rc == 0
    assert db_path.exists()
    assert (repair_log_dir / "ingest_repair.jsonl").exists()
