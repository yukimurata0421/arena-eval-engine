from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrafficLoadResult:
    frame: pd.DataFrame
    traffic_column: str
    warnings: list[str]


def load_daily_traffic(traffic_csv: Path) -> TrafficLoadResult:
    warnings: list[str] = []
    if not traffic_csv.exists():
        warnings.append(f"traffic_csv_not_found:{traffic_csv}")
        return TrafficLoadResult(frame=pd.DataFrame(columns=["date", "traffic_count"]), traffic_column="", warnings=warnings)
    try:
        df = pd.read_csv(traffic_csv)
    except Exception as exc:
        warnings.append(f"traffic_csv_read_failed:{exc}")
        return TrafficLoadResult(frame=pd.DataFrame(columns=["date", "traffic_count"]), traffic_column="", warnings=warnings)

    if "date" not in df.columns:
        warnings.append("traffic_csv_missing_date")
        return TrafficLoadResult(frame=pd.DataFrame(columns=["date", "traffic_count"]), traffic_column="", warnings=warnings)

    traffic_column = ""
    if "hnd_nrt_movements" in df.columns:
        traffic_column = "hnd_nrt_movements"
        traffic = pd.to_numeric(df["hnd_nrt_movements"], errors="coerce")
    elif {"hnd_arr", "hnd_dep", "nrt_arr", "nrt_dep"}.issubset(set(df.columns)):
        traffic_column = "hnd_arr+hnd_dep+nrt_arr+nrt_dep"
        traffic = (
            pd.to_numeric(df["hnd_arr"], errors="coerce")
            + pd.to_numeric(df["hnd_dep"], errors="coerce")
            + pd.to_numeric(df["nrt_arr"], errors="coerce")
            + pd.to_numeric(df["nrt_dep"], errors="coerce")
        )
    else:
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            traffic_column = num_cols[0]
            traffic = pd.to_numeric(df[traffic_column], errors="coerce")
            warnings.append(f"traffic_column_fallback:{traffic_column}")
        else:
            warnings.append("traffic_column_not_found")
            return TrafficLoadResult(frame=pd.DataFrame(columns=["date", "traffic_count"]), traffic_column="", warnings=warnings)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "traffic_count": traffic,
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return TrafficLoadResult(frame=out, traffic_column=traffic_column, warnings=warnings)


def _fit_log_linear_adjustment(
    y: pd.Series,
    traffic: pd.Series,
    min_points: int,
    clip_min_traffic: float,
) -> tuple[pd.Series, dict[str, Any], str]:
    valid = y.notna() & traffic.notna()
    if int(valid.sum()) < min_points:
        return y.copy(), {"status": "skipped", "reason": "insufficient_points"}, "insufficient_points"

    x = np.log1p(np.maximum(traffic.astype(float), clip_min_traffic))
    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    if np.nanstd(xv) <= 1e-9:
        return y.copy(), {"status": "skipped", "reason": "traffic_zero_variance"}, "traffic_zero_variance"
    if np.nanstd(yv) <= 1e-9:
        return y.copy(), {"status": "skipped", "reason": "target_zero_variance"}, "target_zero_variance"

    slope, intercept = np.polyfit(xv, yv, deg=1)
    x_center = float(np.nanmedian(xv))
    adjusted = y.astype(float) - slope * (x - x_center)
    meta = {
        "status": "ok",
        "slope": float(slope),
        "intercept": float(intercept),
        "x_center": x_center,
        "n_valid": int(valid.sum()),
    }
    return adjusted, meta, ""


def apply_traffic_adjustment(
    metric_df: pd.DataFrame,
    target_columns: list[str],
    *,
    min_points: int = 20,
    clip_min_traffic: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    out = metric_df.copy()
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    if "traffic_count" not in out.columns:
        warnings.append("traffic_count_missing_adjustment_skipped")
        return out, metadata, warnings

    for col in target_columns:
        if col not in out.columns:
            metadata.append({"column": col, "status": "skipped", "reason": "column_missing"})
            continue
        adjusted, meta, warn = _fit_log_linear_adjustment(
            y=out[col],
            traffic=out["traffic_count"],
            min_points=min_points,
            clip_min_traffic=clip_min_traffic,
        )
        out[f"traffic_adjusted_{col}"] = adjusted
        meta["column"] = col
        metadata.append(meta)
        if warn:
            warnings.append(f"{col}:{warn}")
    return out, metadata, warnings
