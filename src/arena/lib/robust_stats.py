from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass(frozen=True)
class HodgesLehmannResult:
    """Hodges-Lehmann shift estimate for treatment - control."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_control: int
    n_treatment: int
    n_pairwise: int
    ci_method: str
    notes: str


@dataclass(frozen=True)
class MannWhitneyHLResult:
    """Mann-Whitney U test paired with a robust HL shift estimate."""

    u_statistic: float
    p_value: float
    rank_biserial: float
    hl_estimate: float
    hl_ci_low: float
    hl_ci_high: float
    confidence_level: float
    n_control: int
    n_treatment: int
    n_pairwise: int
    comparison_method: str
    notes: str


def _finite_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _validate_confidence_level(confidence_level: float) -> None:
    if not isfinite(float(confidence_level)) or not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def _count_pairwise_leq(control_sorted: np.ndarray, treatment_sorted: np.ndarray, value: float) -> int:
    """Count pairs where treatment - control <= value without materializing pairs."""
    if control_sorted.size <= treatment_sorted.size:
        counts = np.searchsorted(treatment_sorted, control_sorted + value, side="right")
        return int(np.sum(counts, dtype=np.int64))
    starts = np.searchsorted(control_sorted, treatment_sorted - value, side="left")
    return int(np.sum(control_sorted.size - starts, dtype=np.int64))


def _select_pairwise_binary(control_sorted: np.ndarray, treatment_sorted: np.ndarray, rank_1based: int) -> float:
    lo = float(treatment_sorted[0] - control_sorted[-1])
    hi = float(treatment_sorted[-1] - control_sorted[0])
    if lo == hi:
        return lo

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if _count_pairwise_leq(control_sorted, treatment_sorted, mid) >= rank_1based:
            hi = mid
        else:
            lo = mid
    return hi


def _rank_ci_bounds(n_control: int, n_treatment: int, confidence_level: float) -> tuple[int, int]:
    n_pairwise = n_control * n_treatment
    alpha = 1.0 - confidence_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    mean_u = n_pairwise / 2.0
    sd_u = sqrt(n_control * n_treatment * (n_control + n_treatment + 1) / 12.0)
    critical = ceil(mean_u - z * sd_u)

    if critical <= 0:
        return 1, n_pairwise
    lower_rank = max(1, critical)
    upper_rank = min(n_pairwise, n_pairwise - critical + 1)
    if lower_rank > upper_rank:
        return 1, n_pairwise
    return lower_rank, upper_rank


def _format_notes(base: str, *, selection: str) -> str:
    return f"{base};ci=rank_normal;selection={selection}"


def hodges_lehmann_shift(
    control: Any,
    treatment: Any,
    *,
    confidence_level: float = 0.95,
    min_samples: int = 2,
    exact_pair_limit: int = 5_000_000,
) -> HodgesLehmannResult:
    """Estimate the median pairwise shift ``treatment - control`` with a rank-based CI.

    The estimate is the median of all pairwise differences. For large samples,
    order statistics are selected by binary search over the sorted samples so
    the full pairwise matrix does not need to be materialized.
    """
    _validate_confidence_level(confidence_level)
    x = _finite_1d(control)
    y = _finite_1d(treatment)
    n_control = int(x.size)
    n_treatment = int(y.size)
    n_pairwise = n_control * n_treatment

    if n_control < min_samples or n_treatment < min_samples:
        return HodgesLehmannResult(
            estimate=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            confidence_level=confidence_level,
            n_control=n_control,
            n_treatment=n_treatment,
            n_pairwise=n_pairwise,
            ci_method="rank_normal",
            notes="insufficient_samples",
        )

    median_rank_low = (n_pairwise + 1) // 2
    median_rank_high = (n_pairwise + 2) // 2
    ci_rank_low, ci_rank_high = _rank_ci_bounds(n_control, n_treatment, confidence_level)

    if n_pairwise <= exact_pair_limit:
        diffs = (y[:, None] - x[None, :]).reshape(-1)
        ranks = sorted({median_rank_low - 1, median_rank_high - 1, ci_rank_low - 1, ci_rank_high - 1})
        selected = np.partition(diffs, ranks)
        values = {rank + 1: float(selected[rank]) for rank in ranks}
        notes = _format_notes("ok", selection="exact")
    else:
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        needed_ranks = {median_rank_low, median_rank_high, ci_rank_low, ci_rank_high}
        values = {
            rank: _select_pairwise_binary(x_sorted, y_sorted, rank)
            for rank in needed_ranks
        }
        notes = _format_notes("ok", selection="binary")

    estimate = (values[median_rank_low] + values[median_rank_high]) / 2.0
    return HodgesLehmannResult(
        estimate=float(estimate),
        ci_low=float(values[ci_rank_low]),
        ci_high=float(values[ci_rank_high]),
        confidence_level=confidence_level,
        n_control=n_control,
        n_treatment=n_treatment,
        n_pairwise=n_pairwise,
        ci_method="rank_normal",
        notes=notes,
    )


def mann_whitney_hodges_lehmann(
    control: Any,
    treatment: Any,
    *,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
    min_samples: int = 2,
    exact_pair_limit: int = 5_000_000,
) -> MannWhitneyHLResult:
    """Run MWU and report HL shift for ``treatment - control``."""
    x = _finite_1d(control)
    y = _finite_1d(treatment)
    n_control = int(x.size)
    n_treatment = int(y.size)
    n_pairwise = n_control * n_treatment

    if n_control < min_samples or n_treatment < min_samples:
        return MannWhitneyHLResult(
            u_statistic=np.nan,
            p_value=np.nan,
            rank_biserial=np.nan,
            hl_estimate=np.nan,
            hl_ci_low=np.nan,
            hl_ci_high=np.nan,
            confidence_level=confidence_level,
            n_control=n_control,
            n_treatment=n_treatment,
            n_pairwise=n_pairwise,
            comparison_method="MWU+Hodges-Lehmann",
            notes="insufficient_samples",
        )

    # U is computed as treatment vs control so positive rank-biserial means
    # the treatment group tends to be larger than the control group.
    mwu = mannwhitneyu(y, x, alternative=alternative, method="auto")
    u_stat = float(mwu.statistic)
    p_value = float(mwu.pvalue)
    rank_biserial = float((2.0 * u_stat / n_pairwise) - 1.0)
    hl = hodges_lehmann_shift(
        x,
        y,
        confidence_level=confidence_level,
        min_samples=min_samples,
        exact_pair_limit=exact_pair_limit,
    )
    return MannWhitneyHLResult(
        u_statistic=u_stat,
        p_value=p_value,
        rank_biserial=rank_biserial,
        hl_estimate=hl.estimate,
        hl_ci_low=hl.ci_low,
        hl_ci_high=hl.ci_high,
        confidence_level=confidence_level,
        n_control=n_control,
        n_treatment=n_treatment,
        n_pairwise=n_pairwise,
        comparison_method="MWU+Hodges-Lehmann",
        notes=hl.notes,
    )


def format_estimate_ci(estimate: float, ci_low: float, ci_high: float, *, digits: int = 4) -> str:
    if not (np.isfinite(estimate) and np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "---"
    return f"{estimate:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"
