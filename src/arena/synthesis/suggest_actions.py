from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.synthesis.report_baselines import build_baseline_report

_SEVERITY_RANK = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


def _severity_from_score(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _decorate_action_item(
    *,
    item: dict[str, Any],
    score: int,
    recommended_next_step: str,
    why_this_matters: str,
) -> dict[str, Any]:
    normalized_score = _clamp_score(score)
    enriched = dict(item)
    enriched["severity"] = _severity_from_score(normalized_score)
    enriched["score"] = normalized_score
    enriched["recommended_next_step"] = recommended_next_step
    enriched["why_this_matters"] = why_this_matters
    return enriched


def _severity_rank(value: Any) -> int:
    normalized = str(value or "low").casefold()
    return int(_SEVERITY_RANK.get(normalized, 1))


def _passes_min_severity(item: dict[str, Any], min_severity: str | None) -> bool:
    if min_severity is None:
        return True
    return _severity_rank(item.get("severity")) >= _severity_rank(min_severity)


def _sort_action_items(items: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "severity":
        return sorted(items, key=lambda row: (_severity_rank(row.get("severity")), int(row.get("score", 0))), reverse=True)
    return sorted(items, key=lambda row: (int(row.get("score", 0)), _severity_rank(row.get("severity"))), reverse=True)


def _apply_action_filters(
    items: list[dict[str, Any]],
    *,
    min_severity: str | None,
    sort_by: str,
) -> list[dict[str, Any]]:
    filtered = [row for row in items if _passes_min_severity(row, min_severity=min_severity)]
    return _sort_action_items(filtered, sort_by=sort_by)


def _build_topic_health_summary(
    *,
    topic_distribution: list[dict[str, Any]],
    weak_baselines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    weak_by_topic: dict[str, int] = {}
    for row in weak_baselines:
        topic = str(row.get("topic") or "__null__")
        weak_by_topic[topic] = weak_by_topic.get(topic, 0) + 1

    summary: list[dict[str, Any]] = []
    for row in topic_distribution:
        topic = str(row.get("topic") or "__null__")
        claim_count = int(row.get("claim_count", 0))
        cluster_count = int(row.get("baseline_cluster_count", 0))
        missing_baseline_high_count = int(row.get("high_priority_no_baseline_count", 0))
        mixed_cluster_count = int(row.get("mixed_cluster_count", 0))
        weak_baseline_count = int(weak_by_topic.get(topic, 0))
        fragmentation_flag = cluster_count > 2

        # health_score is "risk score": larger means worse health.
        health_score = _clamp_score(
            missing_baseline_high_count * 30
            + mixed_cluster_count * 20
            + weak_baseline_count * 10
            + (15 if fragmentation_flag else 0)
        )
        if health_score >= 80:
            health_status = "critical"
        elif health_score >= 50:
            health_status = "needs_attention"
        elif health_score >= 20:
            health_status = "watch"
        else:
            health_status = "healthy"

        summary.append(
            {
                "topic": topic,
                "claim_count": claim_count,
                "cluster_count": cluster_count,
                "missing_baseline_high_count": missing_baseline_high_count,
                "mixed_cluster_count": mixed_cluster_count,
                "weak_baseline_count": weak_baseline_count,
                "fragmentation_flag": fragmentation_flag,
                "health_score": health_score,
                "health_status": health_status,
            }
        )
    return sorted(summary, key=lambda x: (-int(x["health_score"]), -int(x["claim_count"]), str(x["topic"])))


def build_action_suggestions(
    *,
    db_path: Path | None = None,
    limit: int = 20,
    topic: str | None = None,
    only_high: bool = False,
    only_conflicts: bool = False,
    min_severity: str | None = None,
    sort_by: str = "score",
) -> dict[str, list[dict[str, Any]]]:
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    if min_severity is not None and str(min_severity).casefold() not in _SEVERITY_RANK:
        raise ValueError("min_severity must be one of: high, medium, low.")
    if sort_by not in {"score", "severity"}:
        raise ValueError("sort_by must be one of: score, severity.")

    report = build_baseline_report(
        db_path=db_path,
        limit=10**9,
        topic=topic,
        include_cluster_details=True,
    )

    bottlenecks: list[dict[str, Any]] = []
    for item in report.get("high_priority_without_baseline", []):
        topic_name = item.get("topic")
        review_status = str(item.get("review_status") or "")
        score = 88
        if topic_name in (None, "__null__"):
            score -= 5
        if review_status == "needs_review":
            score += 5
        bottlenecks.append(
            _decorate_action_item(
                item={
                    "claim_id": int(item["hypothesis_id"]),
                    "claim": item.get("claim"),
                    "topic": topic_name,
                    "priority": item.get("priority_hint"),
                    "reason": "missing baseline",
                },
                score=score,
                recommended_next_step="Re-check baseline candidates and add explicit comparison wording or required evidence files.",
                why_this_matters="Priority is high, but baseline reference is still unresolved, so interpretation cannot converge.",
            )
        )

    conflicts: list[dict[str, Any]] = []
    for item in report.get("mixed_clusters", []):
        claim_ids = item.get("claim_ids") or []
        claim_count = int(item.get("claim_count", len(claim_ids)))
        topic_count = int(item.get("topic_count", 0))
        score = 70 + max(0, claim_count - 2) * 5 + (10 if topic_count == 1 and claim_count > 0 else 0)
        conflicts.append(
            _decorate_action_item(
                item={
                    "cluster_id": int(item["cluster_id"]),
                    "label": item.get("label"),
                    "claim_ids": claim_ids,
                    "summary": "supported and negative claims coexist in one baseline cluster",
                },
                score=score,
                recommended_next_step="Review fixed_conditions and variable_conditions to decide whether cluster split or condition expansion is needed.",
                why_this_matters="Opposing claims coexist under one baseline, which may indicate missing conditions or over-merged clustering.",
            )
        )

    fragmentation: list[dict[str, Any]] = []
    topic_claim_counts = {
        str(row.get("topic") or "__null__"): int(row.get("claim_count", 0))
        for row in report.get("topic_distribution", [])
    }
    topic_to_labels: dict[str, set[str]] = {}
    for detail in report.get("cluster_details", []):
        label = detail.get("label")
        topics = detail.get("topics") or []
        for t in topics:
            if t not in topic_to_labels:
                topic_to_labels[t] = set()
            if label:
                topic_to_labels[t].add(str(label))
    for t, labels in sorted(topic_to_labels.items()):
        if len(labels) <= 2:
            continue
        claim_count = int(topic_claim_counts.get(t, 0))
        score = 50 + (len(labels) - 2) * 10 + min(20, max(0, claim_count - 3) * 2)
        fragmentation.append(
            _decorate_action_item(
                item={
                    "topic": t,
                    "cluster_count": len(labels),
                    "cluster_labels": sorted(labels),
                },
                score=score,
                recommended_next_step="Review baseline label variance and synonym mapping to identify mergeable clusters.",
                why_this_matters="Baseline references are fragmented within the topic, so a shared interpretation is not established.",
            )
        )

    weak_baselines: list[dict[str, Any]] = []
    for detail in report.get("cluster_details", []):
        if int(detail.get("claim_count", 0)) != 1:
            continue
        if str(detail.get("confidence") or "").casefold() != "low":
            continue
        topics = detail.get("topics") or []
        high_priority_count = int((detail.get("priority_hint_breakdown") or {}).get("high", 0))
        score = 40 + (20 if high_priority_count > 0 else 0)
        weak_baselines.append(
            _decorate_action_item(
                item={
                    "cluster_id": int(detail["cluster_id"]),
                    "label": detail.get("label"),
                    "topic": topics[0] if topics else "__null__",
                },
                score=score,
                recommended_next_step="Check for additional related claims and decide whether this baseline is an isolated case or an unresolved candidate.",
                why_this_matters="This baseline is singleton and low-confidence, so it is not yet stable as a comparison reference.",
            )
        )

    topic_health_summary = _build_topic_health_summary(
        topic_distribution=list(report.get("topic_distribution", [])),
        weak_baselines=weak_baselines,
    )

    if only_conflicts:
        bottlenecks = []
        fragmentation = []
        weak_baselines = []
    elif only_high:
        conflicts = []
        fragmentation = []
        weak_baselines = []

    bottlenecks = _apply_action_filters(
        bottlenecks,
        min_severity=min_severity,
        sort_by=sort_by,
    )
    conflicts = _apply_action_filters(
        conflicts,
        min_severity=min_severity,
        sort_by=sort_by,
    )
    fragmentation = _apply_action_filters(
        fragmentation,
        min_severity=min_severity,
        sort_by=sort_by,
    )
    weak_baselines = _apply_action_filters(
        weak_baselines,
        min_severity=min_severity,
        sort_by=sort_by,
    )

    def _limit(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return items[:limit]

    return {
        "bottlenecks": _limit(bottlenecks),
        "conflicts": _limit(conflicts),
        "fragmentation": _limit(fragmentation),
        "weak_baselines": _limit(weak_baselines),
        "topic_health_summary": _limit(topic_health_summary),
    }


def render_action_suggestions_text(suggestions: dict[str, list[dict[str, Any]]]) -> str:
    def _sev_label(value: Any) -> str:
        sev = str(value or "low").casefold()
        return sev.upper()

    lines: list[str] = []
    if suggestions.get("bottlenecks"):
        lines.append("[HIGH] Missing baseline (priority=high)")
        for row in suggestions["bottlenecks"]:
            lines.append(
                f"- [{_sev_label(row.get('severity'))}] score={row.get('score')} "
                f"claim_id={row['claim_id']} topic={row.get('topic')} reason={row.get('reason')}"
            )
            lines.append(f"  next_step={row.get('recommended_next_step')}")
    if suggestions.get("conflicts"):
        lines.append("[MEDIUM] Mixed cluster detected")
        for row in suggestions["conflicts"]:
            lines.append(
                f"- [{_sev_label(row.get('severity'))}] score={row.get('score')} "
                f"cluster_id={row['cluster_id']} label={row.get('label')} claim_ids={row.get('claim_ids')}"
            )
            lines.append(f"  next_step={row.get('recommended_next_step')}")
    if suggestions.get("fragmentation"):
        lines.append("[LOW] Fragmented baseline")
        for row in suggestions["fragmentation"]:
            lines.append(
                f"- [{_sev_label(row.get('severity'))}] score={row.get('score')} "
                f"topic={row['topic']} cluster_count={row['cluster_count']} labels={row['cluster_labels']}"
            )
            lines.append(f"  next_step={row.get('recommended_next_step')}")
    if suggestions.get("weak_baselines"):
        lines.append("[LOW] Weak baseline")
        for row in suggestions["weak_baselines"]:
            lines.append(
                f"- [{_sev_label(row.get('severity'))}] score={row.get('score')} "
                f"cluster_id={row['cluster_id']} label={row.get('label')} topic={row.get('topic')}"
            )
            lines.append(f"  next_step={row.get('recommended_next_step')}")
    if suggestions.get("topic_health_summary"):
        lines.append("Topic Health Summary:")
        for row in suggestions["topic_health_summary"]:
            lines.append(
                f"- topic={row.get('topic')} health_status={row.get('health_status')} health_score={row.get('health_score')} "
                f"missing_high={row.get('missing_baseline_high_count')} mixed={row.get('mixed_cluster_count')} "
                f"weak={row.get('weak_baseline_count')} fragmentation={int(bool(row.get('fragmentation_flag')))}"
            )
    if not lines:
        lines.append("No suggested actions.")
    return "\n".join(lines)


def render_action_suggestions_json(suggestions: dict[str, list[dict[str, Any]]]) -> str:
    return json.dumps(suggestions, ensure_ascii=False, indent=2)
