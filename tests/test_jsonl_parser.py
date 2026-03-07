import json
from pathlib import Path

import pytest

from arena import pipeline


def _parse_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"invalid JSONL record type at line {line_no}")
        records.append(rec)
    return records


def _write_valid_settings(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 36.0",
                "lon = 140.0",
                "",
                "[quality]",
                "min_auc_n_used = 5000",
                "min_minutes_covered = 1200",
                "",
                "[distance_bins]",
                "km = [0, 25, 50]",
            ]
        ),
        encoding="utf-8",
    )


def _write_minimal_phase(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[events]",
                "2026-01-01 = Baseline | rtl-sdr | blue",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_pipeline_jsonl(tmp_path: Path, monkeypatch) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    log_path = output_root / "performance" / "pipeline_runs.jsonl"
    settings_path = tmp_path / "settings.toml"
    phase_path = tmp_path / "phases.txt"
    (scripts_root / "adsb").mkdir(parents=True, exist_ok=True)
    _write_valid_settings(settings_path)
    _write_minimal_phase(phase_path)
    monkeypatch.setenv("ARENA_SETTINGS", str(settings_path))

    cfg = pipeline.RunConfig(
        stage=1,
        only=1,
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-11",
        phase_config=str(phase_path),
        validate=False,
        log_jsonl=str(log_path),
        skip_plao=True,
    )
    assert pipeline.run(cfg) == 0

    records = _parse_jsonl(log_path)
    statuses = {str(rec.get("status")) for rec in records}
    assert "CONFIG" in statuses
    assert "DRY" in statuses


def test_parse_jsonl_reports_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        _parse_jsonl(bad)
