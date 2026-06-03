from __future__ import annotations

import math

import numpy as np
import pytest

from arena.lib.robust_stats import format_estimate_ci, hodges_lehmann_shift, mann_whitney_hodges_lehmann


def test_hodges_lehmann_shift_uses_median_pairwise_difference() -> None:
    control = np.array([1.0, 2.0, 3.0])
    treatment = np.array([3.0, 4.0, 5.0])

    result = hodges_lehmann_shift(control, treatment, min_samples=2)

    assert result.estimate == pytest.approx(2.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(4.0)
    assert result.n_pairwise == 9
    assert result.notes.startswith("ok;")


def test_mann_whitney_hodges_lehmann_reports_directional_effect() -> None:
    result = mann_whitney_hodges_lehmann([1, 2, 3, math.nan], [4, 5, 6], min_samples=2)

    assert result.p_value <= 0.1
    assert result.rank_biserial > 0
    assert result.hl_estimate == pytest.approx(3.0)
    assert result.comparison_method == "MWU+Hodges-Lehmann"


def test_hodges_lehmann_binary_selection_matches_exact_selection() -> None:
    control = np.linspace(0.0, 10.0, 40)
    treatment = np.linspace(2.0, 12.0, 35)

    exact = hodges_lehmann_shift(control, treatment, exact_pair_limit=10_000)
    binary = hodges_lehmann_shift(control, treatment, exact_pair_limit=10)

    assert binary.estimate == pytest.approx(exact.estimate, abs=1e-8)
    assert binary.ci_low == pytest.approx(exact.ci_low, abs=1e-8)
    assert binary.ci_high == pytest.approx(exact.ci_high, abs=1e-8)
    assert "selection=binary" in binary.notes


def test_format_estimate_ci_handles_missing_values() -> None:
    assert format_estimate_ci(1.23456, 0.5, 2.0, digits=2) == "1.23 [0.50, 2.00]"
    assert format_estimate_ci(float("nan"), 0.5, 2.0) == "---"
