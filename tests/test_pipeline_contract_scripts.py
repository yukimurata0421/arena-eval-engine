from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


def _write_settings(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 36.0",
                "lon = 140.0",
                "",
                "[quality]",
                "min_auc_n_used = 10",
                "min_minutes_covered = 5",
                "",
                "[distance_bins]",
                "km = [0, 25, 50]",
            ]
        ),
        encoding="utf-8",
    )


def _write_phase_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[events]",
                "2026-01-01 = Baseline | rtl-sdr | #2c3e50",
                "2026-01-03 = Gain + Filter | airspy_mini | #e74c3c",
            ]
        ),
        encoding="utf-8",
    )


def test_phase_timebin_export_writes_contract_outputs(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "output"
    time_resolved_dir = output_root / "time_resolved"
    settings_path = tmp_path / "settings.toml"
    phase_path = tmp_path / "phases.txt"
    time_resolved_dir.mkdir(parents=True)
    _write_settings(settings_path)
    _write_phase_config(phase_path)

    pd.DataFrame(
        [
            {"date": "2026-01-01", "auc_n_used": 100, "minutes_covered": 60, "local_traffic_proxy": 50},
            {"date": "2026-01-03", "auc_n_used": 120, "minutes_covered": 70, "local_traffic_proxy": 55},
        ]
    ).to_csv(output_root / "adsb_daily_summary_v2.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "time_bin": "00:00-00:30", "auc_sum": 100, "minutes": 30, "capture_ratio": 0.5},
            {"date": "2026-01-03", "time_bin": "00:00-00:30", "auc_sum": 120, "minutes": 30, "capture_ratio": 0.7},
        ]
    ).to_csv(time_resolved_dir / "adsb_timebin_summary.csv", index=False)

    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("ARENA_SETTINGS", str(settings_path))
    monkeypatch.setenv("ARENA_PHASE_CONFIG", str(phase_path))

    module = importlib.import_module("scripts.adsb.analysis.phase.adsb_phase_timebin_export")
    module = importlib.reload(module)

    assert module.main() == 0
    assert (output_root / "performance" / "phase_config_daily_mapping.csv").exists()
    assert (output_root / "performance" / "phase_timebin_summary.csv").exists()
    assert (output_root / "performance" / "phase_timebin_export_report.txt").exists()


def test_change_point_scripts_write_reports_when_source_missing(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "output"
    settings_path = tmp_path / "settings.toml"
    output_root.mkdir(parents=True)
    _write_settings(settings_path)

    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("ARENA_SETTINGS", str(settings_path))

    single = importlib.import_module("scripts.adsb.analysis.change_points.adsb_detect_change_point")
    multi = importlib.import_module("scripts.adsb.analysis.change_points.adsb_detect_multi_change_points")
    single = importlib.reload(single)
    multi = importlib.reload(multi)

    single.run_discovery_analysis()
    multi.run_multi_discovery_analysis()

    change_dir = output_root / "change_point"
    assert (change_dir / "change_point_report.txt").exists()
    assert (change_dir / "change_point_result.json").exists()
    assert (change_dir / "multi_change_points_report.txt").exists()
    assert (change_dir / "multi_change_points_result.json").exists()
    assert "source_csv_not_found" in (change_dir / "change_point_report.txt").read_text(encoding="utf-8")


def test_bayesian_dynamic_quick_mode_runs_without_sampling(monkeypatch, capsys) -> None:
    module = importlib.import_module("scripts.adsb.analysis.bayesian.adsb_bayesian_dynamic_eval")
    module = importlib.reload(module)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]),
            "auc_n_used": [100.0, 110.0, 105.0, 140.0, 150.0, 145.0],
            "local_traffic_proxy": [10, 10, 11, 12, 12, 12],
            "log_traffic": [1.0, 1.0, 1.1, 1.2, 1.2, 1.2],
        }
    )
    monkeypatch.setenv("ADSB_BAYES_DYNAMIC_MODE", "quick")
    monkeypatch.setattr(module, "get_quality_thresholds", lambda: (10, 5))
    monkeypatch.setattr(module, "load_summary", lambda min_auc, min_minutes: df.copy())
    monkeypatch.setattr(module, "prompt_intervention_date", lambda default: pd.Timestamp("2026-01-04"))

    module.run_bayesian_analysis()
    out = capsys.readouterr().out
    assert "FAST" in out
    assert "Probability the change was effective" in out


def test_signal_stats_aggregator_keeps_legacy_daily_summary(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    data_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    sample_path = data_root / "20260101_signal.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                '{"ts": 1767225600, "buckets": {"150-175km": {"n_samples": 2, "avg_signal": -10.0, "avg_snr": 5.0}}}',
                '{"ts": 1767229200, "buckets": {"150-175km": {"n_samples": 1, "avg_signal": -13.0, "avg_snr": 6.0}}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ARENA_DATA_DIR", str(data_root))
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))

    module = importlib.import_module("scripts.signals.collectors.signal_stats_aggregator")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "SEARCH_DIRS", [str(data_root)])
    monkeypatch.setattr(module, "OUTPUT_FILE", str(output_root / "adsb_signal_range_summary.csv"))
    monkeypatch.setattr(module, "LEGACY_OUTPUT_FILE", str(output_root / "adsb_signal_daily_summary.csv"))
    module.aggregate_signal_ranges()

    assert (output_root / "adsb_signal_range_summary.csv").exists()
    legacy_path = output_root / "adsb_signal_daily_summary.csv"
    assert legacy_path.exists()
    legacy_df = pd.read_csv(legacy_path)
    assert list(legacy_df.columns) == ["date", "sig_150_175"]
