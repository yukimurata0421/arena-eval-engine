from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def _bootstrap_hl_ci(
    diffs: np.ndarray,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    if diffs.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    sample_size = min(2000, diffs.size)
    medians = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.choice(diffs, size=sample_size, replace=True)
        medians[i] = float(np.median(sampled))
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def compare_before_after(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    x = np.asarray(before, dtype=float)
    y = np.asarray(after, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n_before = int(x.size)
    n_after = int(y.size)
    if n_before < 2 or n_after < 2:
        return {
            "comparison_method": "MWU+HL",
            "p_value": np.nan,
            "effect_size": np.nan,
            "hl_or_pim_shift": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "notes": "insufficient_samples",
            "n_before": n_before,
            "n_after": n_after,
        }

    # U computed as after vs before, so effect sign maps to post-pre.
    mwu = mannwhitneyu(y, x, alternative="two-sided", method="auto")
    u_stat = float(mwu.statistic)
    p_value = float(mwu.pvalue)
    rank_biserial = float((2.0 * u_stat / (n_after * n_before)) - 1.0)

    diffs = (y[:, None] - x[None, :]).ravel()
    hl_shift = float(np.median(diffs))
    ci_low, ci_high = _bootstrap_hl_ci(diffs, n_bootstrap=n_bootstrap, random_seed=random_seed)
    return {
        "comparison_method": "MWU+HL",
        "p_value": p_value,
        "effect_size": rank_biserial,
        "hl_or_pim_shift": hl_shift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "notes": "ok",
        "n_before": n_before,
        "n_after": n_after,
    }


def split_series_by_change_date(
    df: pd.DataFrame,
    *,
    value_column: str,
    change_date: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if value_column not in df.columns:
        return np.array([]), np.array([]), "column_missing"
    if not change_date:
        return np.array([]), np.array([]), "change_date_missing"
    cpd = pd.to_datetime(change_date, errors="coerce")
    if pd.isna(cpd):
        return np.array([]), np.array([]), "change_date_invalid"

    sub = df.loc[:, ["date", value_column]].dropna().sort_values("date").reset_index(drop=True)
    if sub.empty:
        return np.array([]), np.array([]), "no_observation"
    before = sub.loc[sub["date"] < cpd, value_column].to_numpy(dtype=float)
    after = sub.loc[sub["date"] >= cpd, value_column].to_numpy(dtype=float)
    if before.size == 0 or after.size == 0:
        return before, after, "empty_segment"
    return before, after, "ok"


def build_series_comparison_rows(
    *,
    metric_df: pd.DataFrame,
    change_date: str,
    columns: list[str],
    series_prefix: str,
    n_bootstrap: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in columns:
        before, after, split_status = split_series_by_change_date(
            metric_df, value_column=col, change_date=change_date
        )
        stats = compare_before_after(
            before=before,
            after=after,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        rows.append(
            {
                "series_name": f"{series_prefix}.{col}",
                "detected_change_date": change_date,
                "n_before": stats["n_before"],
                "n_after": stats["n_after"],
                "comparison_method": stats["comparison_method"],
                "p_value": stats["p_value"],
                "effect_size": stats["effect_size"],
                "hl_or_pim_shift": stats["hl_or_pim_shift"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "notes": f"{stats['notes']};split={split_status}",
            }
        )
    return rows
