from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from arena.lib import data_loader as dl


def _write_summary_csv(path: Path, *, with_proxy: bool = True) -> None:
    rows = [
        {"date": "2026-01-01", "auc_n_used": 15, "minutes_covered": 70, "local_traffic_proxy": 3},
        {"date": "2026-01-02", "auc_n_used": 20, "minutes_covered": 90, "local_traffic_proxy": 0},
        {"date": "2026-01-03", "auc_n_used": 25, "minutes_covered": 120, "local_traffic_proxy": np.nan},
    ]
    df = pd.DataFrame(rows)
    if not with_proxy:
        df = df.drop(columns=["local_traffic_proxy"])
    df.to_csv(path, index=False, encoding="utf-8")


def test_load_summary_applies_common_preprocessing(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "adsb_daily_summary_v2.csv"
    _write_summary_csv(csv_path, with_proxy=True)

    monkeypatch.setattr(dl, "get_quality_thresholds", lambda: (10, 60))
    monkeypatch.setattr(dl, "get_config", lambda: SimpleNamespace(intervention_date="2026-01-02"))

    df = dl.load_summary(path=str(csv_path), require_proxy=True, min_auc=None, min_minutes=None, post_date="2026-01-02")
    assert df is not None
    assert "log_traffic" in df.columns
    assert "post" in df.columns
    assert set(df["post"].unique()).issubset({0, 1})
    assert (df["local_traffic_proxy"] > 0).all()


def test_load_summary_returns_none_when_proxy_required_but_missing(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "adsb_daily_summary_v2.csv"
    _write_summary_csv(csv_path, with_proxy=False)

    monkeypatch.setattr(dl, "get_quality_thresholds", lambda: (0, 0))
    monkeypatch.setattr(dl, "get_config", lambda: SimpleNamespace(intervention_date="2026-01-02"))

    assert dl.load_summary(path=str(csv_path), require_proxy=True, min_auc=0, min_minutes=0) is None


def test_check_proxy_endogeneity_returns_float(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "adsb_daily_summary_v2.csv"
    _write_summary_csv(csv_path, with_proxy=True)

    monkeypatch.setattr(dl, "get_quality_thresholds", lambda: (0, 0))
    monkeypatch.setattr(dl, "get_config", lambda: SimpleNamespace(intervention_date="2026-01-02"))
    df = dl.load_summary(path=str(csv_path), require_proxy=True, min_auc=0, min_minutes=0)
    assert df is not None
    corr = dl.check_proxy_endogeneity(df)
    assert isinstance(corr, float)
