from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from arena.evidence.schema import EvidenceRow
from arena.evidence.scoring import score_evidence_row


def _direction_sign(direction: str) -> int:
    if direction == "positive":
        return 1
    if direction == "negative":
        return -1
    return 0


def _row_ref(row: EvidenceRow) -> dict[str, Any]:
    return {
        "comparison_id": row.comparison_id,
        "model_name": row.model_name,
        "model_family": row.model_family,
        "metric_family": row.metric_family,
        "effect_scale": row.effect_scale,
        "effect": row.effect,
        "effect_direction": row.effect_direction,
        "reliability_tag": row.reliability_tag,
        "evidence_score": row.evidence_score,
        "role": row.role,
        "source_file": row.source_file,
        "caveats": row.caveats,
    }


def _claim_text(group_key: str, rows: list[EvidenceRow], sign: int) -> str:
    exemplar = rows[0]
    phase_a = exemplar.phase_a or "baseline"
    phase_b = exemplar.phase_b or "target"
    direction = "improved" if sign >= 0 else "worsened"
    metric = exemplar.metric_family.replace("_", " ")
    return f"{phase_b} shows {direction} {metric} versus {phase_a}"


def _route_status(
    *,
    support_score: float,
    has_conflict: bool,
    min_reliability: str,
) -> str:
    if has_conflict:
        return "supported_but_qualified" if support_score >= 0.2 else "conflicting"
    if min_reliability in {"reference_only", "trend_only"}:
        return "needs_more_data"
    if support_score >= 0.6:
        return "supported"
    if support_score >= 0.25:
        return "supported_but_qualified"
    return "insufficient_data"


def _next_data_needed(rows: list[EvidenceRow], counter: list[EvidenceRow]) -> list[str]:
    items: list[str] = []
    for row in rows + counter:
        for value in row.next_data_needed:
            if value and value not in items:
                items.append(value)
    if counter and "resolve metric-family disagreement" not in items:
        items.append("resolve metric-family disagreement")
    return items


def _mark_cross_metric_conflicts(groups: dict[str, list[EvidenceRow]]) -> list[dict[str, Any]]:
    validation_targets: list[dict[str, Any]] = []
    for group_key, rows in groups.items():
        usable = [row for row in rows if row.role != "invalid" and row.effect_direction in {"positive", "negative"}]
        if len(usable) < 2:
            continue
        by_metric: dict[str, set[int]] = defaultdict(set)
        for row in usable:
            by_metric[row.metric_family].add(_direction_sign(row.effect_direction))
        metric_signs = {metric: next(iter(signs)) for metric, signs in by_metric.items() if len(signs) == 1 and 0 not in signs}
        if len(set(metric_signs.values())) <= 1:
            continue

        primary_candidates = sorted(
            [row for row in usable if row.role in {"primary", "corroborating"}],
            key=lambda row: row.evidence_score,
            reverse=True,
        )
        primary_sign = _direction_sign(primary_candidates[0].effect_direction) if primary_candidates else next(iter(metric_signs.values()))
        contradictory = [row for row in usable if _direction_sign(row.effect_direction) not in {0, primary_sign}]
        for row in contradictory:
            if row.role != "invalid":
                row.role = "contradictory"
                row.review_priority = max(row.review_priority, 85)
        validation_targets.append(
            {
                "conflict_group": group_key,
                "reason": "metric_family_direction_conflict",
                "metric_signs": metric_signs,
                "counter_evidence": [_row_ref(row) for row in contradictory],
                "needed_data": "re-check proxy assumptions and distance-bin traffic normalization",
            }
        )
    return validation_targets


def build_claim_routes(rows: list[EvidenceRow]) -> dict[str, Any]:
    scored = [score_evidence_row(row) for row in rows]
    groups: dict[str, list[EvidenceRow]] = defaultdict(list)
    for row in scored:
        groups[row.conflict_group].append(row)

    validation_targets = _mark_cross_metric_conflicts(groups)
    claims: list[dict[str, Any]] = []
    invalid = [_row_ref(row) for row in scored if row.role == "invalid" or row.reliability_tag == "invalid"]

    reliability_rank = {
        "invalid": 0,
        "reference_only": 1,
        "trend_only": 2,
        "usable": 3,
        "likely": 4,
        "strong": 5,
    }

    for group_key, group_rows in sorted(groups.items()):
        usable = [row for row in group_rows if row.role not in {"invalid", "reference_only"}]
        if not usable:
            continue
        primary_rows = [row for row in usable if row.role in {"primary", "corroborating"}]
        if not primary_rows:
            primary_rows = [row for row in usable if row.role != "contradictory"]
        if not primary_rows:
            continue
        primary_rows.sort(key=lambda row: row.evidence_score, reverse=True)
        primary_sign = _direction_sign(primary_rows[0].effect_direction)
        if primary_sign == 0:
            continue
        support = [row for row in usable if _direction_sign(row.effect_direction) == primary_sign and row.role != "contradictory"]
        counter = [row for row in usable if row.role == "contradictory" or _direction_sign(row.effect_direction) not in {0, primary_sign}]
        support_score = sum(row.evidence_score for row in support)
        family_count = len({row.model_family for row in support})
        if family_count <= 1 and len(support) > 1:
            support_score *= 0.75
        min_rel = min((row.reliability_tag for row in support), key=lambda tag: reliability_rank.get(tag, 0), default="reference_only")
        has_conflict = bool(counter)
        caveats: list[str] = []
        for row in support + counter:
            for caveat in row.caveats:
                if caveat not in caveats:
                    caveats.append(caveat)
        claims.append(
            {
                "claim_id": f"CLAIM-EVIDENCE-{len(claims) + 1:03d}",
                "conflict_group": group_key,
                "claim": _claim_text(group_key, primary_rows, primary_sign),
                "status": _route_status(
                    support_score=support_score,
                    has_conflict=has_conflict,
                    min_reliability=min_rel,
                ),
                "primary_metric_family": primary_rows[0].metric_family,
                "support_strength": round(support_score, 4),
                "model_family_count": family_count,
                "support": [_row_ref(row) for row in support],
                "counter_evidence": [_row_ref(row) for row in counter],
                "caveats": caveats,
                "next_data_needed": _next_data_needed(support, counter),
            }
        )

    counts = Counter(row.role for row in scored)
    return {
        "claims": claims,
        "validation_targets": validation_targets,
        "invalid": invalid,
        "summary": {
            "evidence_row_count": len(scored),
            "claim_count": len(claims),
            "validation_target_count": len(validation_targets),
            "invalid_count": len(invalid),
            "role_counts": dict(sorted(counts.items())),
            "metric_family_counts": dict(sorted(Counter(row.metric_family for row in scored).items())),
            "model_family_counts": dict(sorted(Counter(row.model_family for row in scored).items())),
        },
    }


def render_disagreement_report(routes: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Model Evidence Disagreement Report",
        "",
        "This report preserves metric-family disagreements as validation targets instead of averaging them away.",
        "",
    ]
    targets = routes.get("validation_targets") or []
    if not targets:
        lines.append("No metric-family disagreements detected.")
        return "\n".join(lines) + "\n"

    for target in targets:
        lines.append(f"## {target.get('conflict_group')}")
        lines.append("")
        lines.append(f"- reason: {target.get('reason')}")
        lines.append(f"- needed_data: {target.get('needed_data')}")
        metric_signs = target.get("metric_signs") or {}
        for metric, sign in sorted(metric_signs.items()):
            direction = "positive" if int(sign) > 0 else "negative"
            lines.append(f"- {metric}: {direction}")
        counter = target.get("counter_evidence") or []
        if counter:
            lines.append("- counter_evidence:")
            for row in counter:
                lines.append(
                    f"  - {row.get('model_name')} {row.get('metric_family')} "
                    f"{row.get('effect_direction')} effect={row.get('effect')}"
                )
        lines.append("")
    return "\n".join(lines)
