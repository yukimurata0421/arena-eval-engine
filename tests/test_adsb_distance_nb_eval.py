from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_distance_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "adsb" / "analysis" / "stats" / "adsb_distance_nb_eval.py"
    spec = importlib.util.spec_from_file_location("arena_test_adsb_distance_nb_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distance_analysis_writes_hodges_lehmann_columns(monkeypatch, tmp_path: Path) -> None:
    fringe_csv = tmp_path / "fringe_decoding_stats.csv"
    rows = []
    for idx, base in enumerate([10, 12, 14], start=1):
        rows.append(
            {
                "date": f"2026-01-0{idx}",
                "phase": "1_Old_Settings",
                "total": 100,
                "dist_0_100": base,
                "dist_100_200": base + 5,
                "dist_200_300": base + 10,
                "dist_300_plus": base + 15,
            }
        )
    for idx, base in enumerate([20, 22, 24], start=4):
        rows.append(
            {
                "date": f"2026-01-0{idx}",
                "phase": "2_New_Filter",
                "total": 100,
                "dist_0_100": base,
                "dist_100_200": base + 5,
                "dist_200_300": base + 10,
                "dist_300_plus": base + 15,
            }
        )
    pd.DataFrame(rows).to_csv(fringe_csv, index=False)

    module = _load_distance_module()
    out_dir = tmp_path / "performance"
    monkeypatch.setattr(module, "FRINGE_CSV", str(fringe_csv))
    monkeypatch.setattr(module, "OUTPUT_DIR", str(out_dir))

    module.run_distance_analysis()

    summary = pd.read_csv(out_dir / "distance_performance_summary.csv")
    expected_cols = {
        "Comparison_Method",
        "Mann_Whitney_U_Target_vs_Baseline",
        "Rank_Biserial_Effect",
        "HL_Diff_Pct",
        "HL_95CI_Low_Pct",
        "HL_95CI_High_Pct",
        "HL_95CI_Pct",
    }
    assert expected_cols.issubset(summary.columns)
    assert set(summary["Comparison_Method"]) == {"MWU+Hodges-Lehmann"}
    assert (summary["HL_Diff_Pct"] > 0).all()
