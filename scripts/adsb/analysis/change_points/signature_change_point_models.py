from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass
class ChangePointResult:
    series_name: str
    series_type: str
    adjustment: str
    detected_change_date: str
    detection_method: str
    score: float | None
    notes: str
    split_index: int | None
    n_obs: int

    def to_record(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "series_type": self.series_type,
            "adjustment": self.adjustment,
            "detected_change_date": self.detected_change_date,
            "detection_method": self.detection_method,
            "score": self.score,
            "notes": self.notes,
            "split_index": self.split_index,
            "n_obs": self.n_obs,
        }


def _rank_scan_single(values: np.ndarray, min_segment_days: int) -> tuple[int | None, float | None, str]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n, m = arr.shape
    if m == 0:
        return None, None, "no_feature_columns"
    if n < max(2 * min_segment_days, 6):
        return None, None, "insufficient_length_for_split"

    ranked = np.column_stack([rankdata(arr[:, i], method="average") for i in range(m)])
    med = np.median(ranked, axis=0)
    mad = np.median(np.abs(ranked - med), axis=0)
    mad = np.where(mad < 1e-9, 1.0, mad)
    z = (ranked - med) / mad
    prefix = np.cumsum(z, axis=0)
    total = prefix[-1]

    best_idx: int | None = None
    best_score = -np.inf
    for split in range(min_segment_days, n - min_segment_days + 1):
        n1 = split
        n2 = n - split
        if n1 < min_segment_days or n2 < min_segment_days:
            continue
        left = prefix[split - 1] / n1
        right = (total - prefix[split - 1]) / n2
        delta = right - left
        score = float(np.linalg.norm(delta, ord=2) * np.sqrt((n1 * n2) / n))
        if score > best_score:
            best_score = score
            best_idx = split

    if best_idx is None:
        return None, None, "no_valid_split"
    return best_idx, float(best_score), "ok"


def _detect_single_change(
    *,
    dates: pd.Series,
    values: np.ndarray,
    cp_model: str,
    min_segment_days: int,
) -> tuple[int | None, str, float | None, str]:
    method = cp_model
    if cp_model.startswith("ruptures"):
        try:
            import ruptures as rpt

            model = "rbf"
            signal = np.asarray(values, dtype=float)
            if signal.ndim == 1:
                signal = signal[:, None]
            algo = rpt.Binseg(model=model).fit(signal)
            bkps = algo.predict(n_bkps=1)
            split = int(bkps[0]) if bkps else None
            if split is not None and split >= len(signal):
                split = None
            if split is None:
                return None, "ruptures_binseg", None, "ruptures_no_split"
            return split, "ruptures_binseg", float(np.nan), "ok"
        except Exception as exc:
            method = "rank_scan"
            note = f"ruptures_unavailable_fallback:{exc}"
            split, score, status = _rank_scan_single(values=values, min_segment_days=min_segment_days)
            return split, method, score, f"{status};{note}"

    split, score, status = _rank_scan_single(values=values, min_segment_days=min_segment_days)
    return split, method, score, status


def detect_change_point_for_columns(
    *,
    metric_df: pd.DataFrame,
    columns: list[str],
    series_name: str,
    series_type: str,
    adjustment: str,
    cp_model: str,
    min_segment_days: int,
    min_days: int = 0,
) -> ChangePointResult:
    if not columns:
        n_obs = 0
        if "date" in metric_df.columns:
            n_obs = int(pd.to_datetime(metric_df["date"], errors="coerce").notna().sum())
        return ChangePointResult(
            series_name=series_name,
            series_type=series_type,
            adjustment=adjustment,
            detected_change_date="",
            detection_method=cp_model,
            score=None,
            notes="no_columns_requested",
            split_index=None,
            n_obs=n_obs,
        )

    missing = [c for c in columns if c not in metric_df.columns]
    if missing:
        return ChangePointResult(
            series_name=series_name,
            series_type=series_type,
            adjustment=adjustment,
            detected_change_date="",
            detection_method=cp_model,
            score=None,
            notes=f"missing_columns:{','.join(missing)}",
            split_index=None,
            n_obs=0,
        )
    needed = ["date"] + columns
    sub = metric_df.loc[:, needed].dropna().sort_values("date").reset_index(drop=True)
    n_obs = int(len(sub))
    if n_obs == 0:
        return ChangePointResult(
            series_name=series_name,
            series_type=series_type,
            adjustment=adjustment,
            detected_change_date="",
            detection_method=cp_model,
            score=None,
            notes="no_observations_after_dropna",
            split_index=None,
            n_obs=0,
        )
    if min_days > 0 and n_obs < min_days:
        return ChangePointResult(
            series_name=series_name,
            series_type=series_type,
            adjustment=adjustment,
            detected_change_date="",
            detection_method=cp_model,
            score=None,
            notes=f"insufficient_days_for_model:n_obs={n_obs}<min_days={min_days}",
            split_index=None,
            n_obs=n_obs,
        )

    values = sub[columns].to_numpy(dtype=float)
    split, method_used, score, status = _detect_single_change(
        dates=sub["date"],
        values=values,
        cp_model=cp_model,
        min_segment_days=min_segment_days,
    )
    change_date = ""
    if split is not None and split < len(sub):
        change_date = pd.Timestamp(sub.iloc[split]["date"]).strftime("%Y-%m-%d")

    return ChangePointResult(
        series_name=series_name,
        series_type=series_type,
        adjustment=adjustment,
        detected_change_date=change_date,
        detection_method=method_used,
        score=score,
        notes=f"{status};valid_days={n_obs};columns={','.join(columns)}",
        split_index=split,
        n_obs=n_obs,
    )
