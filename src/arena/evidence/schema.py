from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

CSV_JSON_FIELDS = {
    "score_components",
    "diagnostics",
    "assumptions",
    "warnings",
    "caveats",
    "next_data_needed",
}


EVIDENCE_CSV_FIELDS = [
    "comparison_id",
    "source_file",
    "source_metric",
    "comparison_window",
    "baseline_id",
    "phase_a",
    "phase_b",
    "model_name",
    "model_family",
    "metric_family",
    "effect_scale",
    "effect",
    "effect_direction",
    "ci_low",
    "ci_high",
    "ci_level",
    "p_value",
    "prob_positive",
    "n_a",
    "n_b",
    "n_effective",
    "independence_unit",
    "autocorrelation_risk",
    "role",
    "reliability_tag",
    "evidence_score",
    "review_priority",
    "conflict_group",
    "score_components",
    "diagnostics",
    "assumptions",
    "warnings",
    "caveats",
    "next_data_needed",
]


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.casefold() in {"nan", "none", "null", "---"}:
            return None
        stripped = stripped.replace("%", "")
        try:
            value = float(stripped)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clean_int(value: Any) -> int | None:
    number = clean_float(value)
    if number is None:
        return None
    return int(number)


def effect_direction(effect: float | None) -> str:
    if effect is None:
        return "unknown"
    if effect > 0:
        return "positive"
    if effect < 0:
        return "negative"
    return "neutral"


@dataclass
class EvidenceRow:
    comparison_id: str
    model_name: str
    model_family: str
    metric_family: str
    effect_scale: str
    effect: float | None = None
    effect_direction: str = "unknown"
    source_file: str = ""
    source_metric: str = ""
    comparison_window: str = ""
    baseline_id: str = ""
    phase_a: str = ""
    phase_b: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    p_value: float | None = None
    prob_positive: float | None = None
    n_a: int | None = None
    n_b: int | None = None
    n_effective: float | None = None
    independence_unit: str = "unknown"
    autocorrelation_risk: str = "unknown"
    role: str = "diagnostic"
    reliability_tag: str = "reference_only"
    evidence_score: float = 0.0
    review_priority: int = 0
    conflict_group: str = ""
    score_components: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    next_data_needed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.effect = clean_float(self.effect)
        self.ci_low = clean_float(self.ci_low)
        self.ci_high = clean_float(self.ci_high)
        self.ci_level = clean_float(self.ci_level)
        self.p_value = clean_float(self.p_value)
        self.prob_positive = clean_float(self.prob_positive)
        self.n_a = clean_int(self.n_a)
        self.n_b = clean_int(self.n_b)
        self.n_effective = clean_float(self.n_effective)
        if self.effect_direction == "unknown":
            self.effect_direction = effect_direction(self.effect)
        if not self.conflict_group:
            self.conflict_group = "|".join(
                [
                    self.baseline_id or "unknown_baseline",
                    self.phase_a or "unknown_a",
                    self.phase_b or "unknown_b",
                    self.comparison_window or "default_window",
                ]
            )

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in EVIDENCE_CSV_FIELDS}

    def to_csv_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        for key in CSV_JSON_FIELDS:
            row[key] = json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
        return row
