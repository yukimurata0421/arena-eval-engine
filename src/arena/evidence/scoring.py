from __future__ import annotations

import math

from arena.evidence.schema import EvidenceRow

ROLE_VALUES = {
    "primary",
    "corroborating",
    "diagnostic",
    "proxy",
    "contradictory",
    "reference_only",
    "invalid",
}


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def effective_n_two_group(n_a: int | None, n_b: int | None) -> float | None:
    if n_a is None or n_b is None or n_a <= 0 or n_b <= 0:
        return None
    return float(2 * n_a * n_b / (n_a + n_b))


def n_reliability_tag(n_effective: float | None) -> str:
    if n_effective is None:
        return "reference_only"
    if n_effective < 3:
        return "reference_only"
    if n_effective < 7:
        return "trend_only"
    if n_effective < 14:
        return "usable"
    if n_effective < 21:
        return "likely"
    return "strong"


def n_score(n_effective: float | None) -> float:
    tag = n_reliability_tag(n_effective)
    return {
        "reference_only": 0.15,
        "trend_only": 0.35,
        "usable": 0.6,
        "likely": 0.78,
        "strong": 0.95,
    }[tag]


def precision_score(row: EvidenceRow) -> float:
    if row.role == "invalid":
        return 0.0
    score = 0.55
    if row.prob_positive is not None:
        if row.prob_positive >= 95 or row.prob_positive <= 5:
            score = max(score, 0.9)
        elif row.prob_positive >= 80 or row.prob_positive <= 20:
            score = max(score, 0.72)
        else:
            score = min(score, 0.45)
    if row.p_value is not None:
        if row.p_value < 0.01:
            score = max(score, 0.85)
        elif row.p_value < 0.05:
            score = max(score, 0.75)
        else:
            score = min(score, 0.45)
    if row.ci_low is not None and row.ci_high is not None:
        if row.ci_low <= 0 <= row.ci_high:
            score = min(score, 0.5)
        elif row.effect is not None:
            width = abs(row.ci_high - row.ci_low)
            magnitude = max(abs(row.effect), 1.0)
            if width / magnitude > 5:
                score = min(score, 0.45)
    return clamp01(score)


def assumption_score(row: EvidenceRow) -> float:
    if row.role == "invalid":
        return 0.0
    score = 1.0
    warning_text = " ".join(row.warnings).casefold()
    caveat_text = " ".join(row.caveats).casefold()
    combined = f"{warning_text} {caveat_text}"
    if "complete_separation" in combined or "model_failed" in combined:
        return 0.0
    if "minute_level_non_independence" in combined or "autocorrelation" in combined:
        score *= 0.55
    if "prior_sensitivity_large" in combined:
        score *= 0.4
    elif "prior_sensitivity_medium" in combined:
        score *= 0.7
    if "target_ci_not_delta_ci" in combined:
        score *= 0.75
    return clamp01(score)


def proxy_score(row: EvidenceRow) -> float:
    if row.role == "invalid":
        return 0.0
    if row.metric_family == "capture_ratio":
        return 0.45
    if row.metric_family == "distance_bin":
        return 0.68
    if row.metric_family == "coverage_auc":
        return 0.9
    if row.metric_family == "coverage_signature":
        return 0.74
    if row.metric_family == "traffic_proxy":
        return 0.48
    if row.metric_family == "time_bin":
        return 0.72
    return 0.65


def practical_effect_score(row: EvidenceRow) -> float:
    if row.role == "invalid" or row.effect is None:
        return 0.0
    magnitude = abs(row.effect)
    if row.effect_scale == "rank_probability":
        if 0.0 <= row.effect <= 1.0:
            centered = min(abs(row.effect - 0.5) * 2.0, 1.0)
        else:
            centered = min(magnitude, 1.0)
        return clamp01(0.4 + centered * 0.6)
    if row.effect_scale == "rank_biserial":
        return clamp01(0.4 + min(magnitude, 1.0) * 0.6)
    if row.effect_scale in {"coverage_auc_delta_pct", "posterior_ratio", "mean_delta_pct", "location_shift_pct"}:
        if magnitude >= 20:
            return 0.9
        if magnitude >= 5:
            return 0.7
        return 0.45
    if row.effect_scale in {"location_shift", "mean_delta", "capture_ratio_delta"}:
        return 0.65 if magnitude > 0 else 0.2
    if row.effect_scale == "change_score":
        return 0.55
    return 0.6


def role_for_row(row: EvidenceRow) -> str:
    if row.role in ROLE_VALUES and row.role != "diagnostic":
        return row.role
    if row.model_family == "invalid":
        return "invalid"
    if row.metric_family == "coverage_auc" and row.model_name == "bayes_phase":
        return "primary"
    if row.metric_family == "coverage_auc":
        return "corroborating"
    if row.metric_family == "capture_ratio":
        return "proxy"
    if row.metric_family == "traffic_proxy":
        return "proxy"
    if row.metric_family == "distance_bin":
        return "diagnostic"
    return "diagnostic"


def score_evidence_row(row: EvidenceRow) -> EvidenceRow:
    if row.n_effective is None:
        row.n_effective = effective_n_two_group(row.n_a, row.n_b)
    if row.n_effective is None and row.n_b is not None:
        row.n_effective = float(row.n_b)
    row.role = role_for_row(row)
    if row.role == "invalid":
        row.reliability_tag = "invalid"
        row.evidence_score = 0.0
        row.score_components = {
            "applicability_score": 0.0,
            "precision_score": 0.0,
            "assumption_score": 0.0,
            "proxy_score": 0.0,
            "practical_effect_score": 0.0,
        }
        return row

    components = {
        "applicability_score": n_score(row.n_effective),
        "precision_score": precision_score(row),
        "assumption_score": assumption_score(row),
        "proxy_score": proxy_score(row),
        "practical_effect_score": practical_effect_score(row),
    }
    score = 1.0
    for value in components.values():
        score *= clamp01(value)
    row.score_components = {key: round(value, 4) for key, value in components.items()}
    row.evidence_score = round(clamp01(score), 4)

    n_tag = n_reliability_tag(row.n_effective)
    if n_tag in {"reference_only", "trend_only"}:
        row.reliability_tag = n_tag
    elif row.evidence_score >= 0.55:
        row.reliability_tag = "strong" if row.evidence_score >= 0.7 else "likely"
    else:
        row.reliability_tag = "usable"
    if row.role == "reference_only":
        row.reliability_tag = "reference_only"
    return row
