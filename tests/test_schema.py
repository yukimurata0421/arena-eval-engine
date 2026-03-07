from pathlib import Path

from arena import cli


def _write_invalid_settings(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 36.0",
                "lon = 140.0",
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


def test_validate_fails_on_missing_required_sections(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SCRIPTS_ROOT", raising=False)
    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARENA_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("ARENA_SETTINGS", raising=False)
    monkeypatch.delenv("ARENA_PHASE_CONFIG", raising=False)

    scripts_root = tmp_path / "scripts"
    (scripts_root / "adsb").mkdir(parents=True, exist_ok=True)
    settings_path = tmp_path / "settings.toml"
    phase_path = tmp_path / "phases.txt"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_invalid_settings(settings_path)
    _write_minimal_phase(phase_path)

    code = cli.main(
        [
            "validate",
            "--scripts-root",
            str(scripts_root),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--settings",
            str(settings_path),
            "--phase-config",
            str(phase_path),
            "--create-dirs",
        ]
    )
    assert code == 1
