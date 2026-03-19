from __future__ import annotations

from pathlib import Path

from arena.lib import phase_config as pc


def test_load_phase_config_uses_defaults_when_file_missing(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "phases.txt"
    cfg = pc.load_phase_config(str(missing))
    out = capsys.readouterr().out
    assert "using default value" in out
    assert cfg.intervention_date


def test_load_phase_config_parses_events_and_settings(tmp_path: Path) -> None:
    path = tmp_path / "phases.txt"
    path.write_text(
        "\n".join(
            [
                "[events]",
                "2026-01-01 = Init | rtl-sdr |",
                "2026-02-01 = CableFix | airspy_mini_plus_cable | #ff0000",
                "",
                "[settings]",
                "post_change_date = 2026-01-10",
                "intervention_date = 2026-02-11",
                "report_start_date = 2026-01-08",
                "fringe_boundary = 2026-01-31, 2026-02-14",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = pc.load_phase_config(str(path))
    assert cfg.config_path.endswith("phases.txt")
    assert len(cfg.events) == 2
    assert cfg.events[0].date == "2026-01-01"
    assert cfg.events[1].hardware == "airspy_mini_plus_cable"
    assert cfg.events[1].color == "#ff0000"
    assert cfg._fringe_boundary == ["2026-01-31", "2026-02-14"]
    assert cfg.fringe_phase("2026-01-15") == "1_Old_Settings"
    assert cfg.fringe_phase("2026-02-01") == "2_New_Filter"
    assert cfg.fringe_phase("2026-03-01") == "3_Post_Cable_Fix"


def test_get_config_caches_and_force_reload(tmp_path: Path) -> None:
    pc.clear_config_cache()
    path = tmp_path / "phases.txt"
    path.write_text("[events]\n2026-01-01 = Init | rtl-sdr |\n", encoding="utf-8")

    cfg1 = pc.get_config(str(path), force_reload=True)
    cfg2 = pc.get_config(str(path), force_reload=False)
    assert cfg1 is cfg2

    # Modify file and force reload to get a new object.
    path.write_text("[events]\n2026-01-01 = Init | rtl-sdr |\n2026-02-01 = X | airspy_mini |\n", encoding="utf-8")
    cfg3 = pc.get_config(str(path), force_reload=True)
    assert cfg3 is not cfg2
    assert len(cfg3.events) == 2


def test_hardware_views_and_phase_names(tmp_path: Path) -> None:
    path = tmp_path / "phases.txt"
    path.write_text(
        "\n".join(
            [
                "[events]",
                "2026-01-01 = Init | rtl-sdr |",
                "2026-02-01 = Airspy | airspy_mini |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = pc.load_phase_config(str(path))
    assert cfg.default_hardware == "rtl-sdr"
    assert cfg.hardware_at("2026-01-15") == "rtl-sdr"
    assert cfg.hardware_at("2026-02-15") == "airspy_mini"
    assert cfg.hardware_transitions == [("2026-02-01", "airspy_mini")]
    assert cfg.phase_names[cfg.hardware_map["rtl-sdr"]] == "RTL-SDR"
