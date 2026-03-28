from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.synthesis.db import DB_PATH, connect, ensure_schema
from arena.synthesis.normalization import normalize_optional_text

TOPIC_TAXONOMY: tuple[str, ...] = (
    "gain_tuning",
    "threshold_tuning",
    "antenna_change",
    "cable_change",
    "filter_change",
    "receiver_config",
    "traffic_adjustment",
    "coverage_auc",
    "distance_distribution",
    "sector_pattern",
    "dropout_behavior",
    "baseline_definition",
    "data_quality",
    "statistical_validity",
    "artifact_pipeline",
    "unknown",
)

BASELINE_TYPE_VOCAB: tuple[str, ...] = (
    "temporal",
    "hardware",
    "parameter",
    "metric_definition",
    "composite",
)

REVIEW_STATUS_VOCAB: tuple[str, ...] = (
    "auto_unreviewed",
    "auto_high_confidence",
    "needs_review",
    "human_confirmed",
    "human_corrected",
)

_HUMAN_LOCKED_STATUSES = {"human_confirmed", "human_corrected"}
_AUTO_MUTABLE_STATUSES = {None, "auto_unreviewed", "auto_high_confidence", "needs_review"}

_TOPIC_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "gain_tuning": {
        "keywords": ("receiver gain", "receiver_gain", "gain=", "gain ", "lna gain"),
        "axes": ("gain", "receiver_config"),
    },
    "threshold_tuning": {
        "keywords": ("threshold", "squelch", "sensitivity"),
        "axes": ("threshold",),
    },
    "antenna_change": {
        "keywords": ("antenna", "antennae"),
        "axes": ("antenna", "coverage_auc"),
    },
    "cable_change": {
        "keywords": ("cable swap", "cable change", "coax", "coaxial"),
        "axes": ("cable",),
    },
    "filter_change": {
        "keywords": ("filter", "lpf", "bandpass"),
        "axes": ("filter",),
    },
    "receiver_config": {
        "keywords": ("receiver config", "readsb", "rtlsdr", "rtl_sdr", "configuration"),
        "axes": ("receiver_config",),
    },
    "traffic_adjustment": {
        "keywords": ("traffic", "congestion", "load", "volume"),
        "axes": ("traffic",),
    },
    "coverage_auc": {
        "keywords": ("auc", "coverage", "area under curve"),
        "axes": ("coverage_auc",),
    },
    "distance_distribution": {
        "keywords": ("distance", "range", "distribution", "km"),
        "axes": ("distance",),
    },
    "sector_pattern": {
        "keywords": ("sector", "azimuth", "bearing", "heading"),
        "axes": ("sector",),
    },
    "dropout_behavior": {
        "keywords": ("dropout", "packet loss", "missing", "gap"),
        "axes": ("dropout",),
    },
    "baseline_definition": {
        "keywords": ("baseline", "reference", "compared to", "versus", "before", "after"),
        "axes": ("baseline",),
    },
    "data_quality": {
        "keywords": ("quality", "noise", "outlier", "missingness"),
        "axes": ("quality",),
    },
    "statistical_validity": {
        "keywords": ("p-value", "confidence interval", "significant", "bootstrap", "variance"),
        "axes": ("stats",),
    },
    "artifact_pipeline": {
        "keywords": ("ingest", "validator", "schema", "artifact", "pipeline"),
        "axes": ("pipeline",),
    },
}

_AXIS_RULES: dict[str, tuple[str, ...]] = {
    "gain": ("gain", "receiver_gain", "lna"),
    "threshold": ("threshold", "squelch", "sensitivity"),
    "antenna": ("antenna", "antennae"),
    "cable": ("cable", "coax", "coaxial"),
    "filter": ("filter", "lpf", "bandpass"),
    "receiver_config": ("receiver", "readsb", "rtl"),
    "traffic": ("traffic", "load", "volume", "congestion"),
    "coverage_auc": ("auc", "coverage", "area under curve"),
    "distance": ("distance", "range", "km"),
    "sector": ("sector", "azimuth", "bearing"),
    "dropout": ("dropout", "packet loss", "missing", "gap"),
    "baseline": ("baseline", "reference", "versus", "compared to", "before", "after"),
    "quality": ("quality", "noise", "outlier", "missingness"),
    "stats": ("p-value", "confidence interval", "significant", "bootstrap", "variance"),
    "pipeline": ("ingest", "validator", "schema", "artifact", "pipeline"),
}

_COMPARE_MARKERS = (
    "compared to",
    "versus",
    "vs ",
    "baseline",
    "reference",
    "before",
    "after",
    "pre-change",
    "post-change",
)


@dataclass(frozen=True, slots=True)
class TopicResult:
    primary_topic: str | None
    topic_confidence: str | None
    topic_method: str | None
    topic_reason: str | None
    exploration_axes: list[str] | None


@dataclass(frozen=True, slots=True)
class BaselineResult:
    baseline_label: str | None
    baseline_type: str | None
    baseline_confidence: str | None
    baseline_method: str | None
    baseline_reason: str | None


@dataclass(frozen=True, slots=True)
class EnrichReport:
    scanned_claims: int
    updated_claims: int
    skipped_human_locked: int
    topic_assigned: int
    baseline_assigned: int
    needs_review_count: int
    high_confidence_count: int
    dry_run: bool


def _connect_for_enrich(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _safe_json_loads_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()]


def _normalize_confidence(value: str | None) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if lowered in {"high", "medium", "low"}:
        return lowered
    return normalized


def _normalize_label_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", text.casefold())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _collect_context(record: dict[str, Any]) -> dict[str, str]:
    claim = normalize_optional_text(record.get("claim")) or ""
    basis = normalize_optional_text(record.get("basis")) or ""
    raw_text = normalize_optional_text(record.get("raw_text")) or ""
    evidence_refs = _safe_json_loads_list(record.get("evidence_refs"))
    required_names = record.get("required_file_names") or []
    if not isinstance(required_names, list):
        required_names = []
    required_names = [name.strip() for name in required_names if isinstance(name, str) and name.strip()]

    claim_basis = " ".join([claim, basis]).casefold()
    auxiliary = " ".join([raw_text, " ".join(evidence_refs), " ".join(required_names)]).casefold()
    combined = " ".join([claim_basis, auxiliary]).strip()
    return {
        "claim_basis": claim_basis,
        "auxiliary": auxiliary,
        "combined": combined,
    }


def infer_exploration_axes(record: dict[str, Any], primary_topic: str | None = None) -> list[str] | None:
    context = _collect_context(record)
    combined = context["combined"]
    axes: set[str] = set()

    for axis, keywords in _AXIS_RULES.items():
        if any(keyword in combined for keyword in keywords):
            axes.add(axis)

    if primary_topic and primary_topic in _TOPIC_RULES:
        axes.update(_TOPIC_RULES[primary_topic]["axes"])

    if not axes:
        return None
    return sorted(axes)


def infer_primary_topic(record: dict[str, Any]) -> TopicResult:
    context = _collect_context(record)
    claim_basis = context["claim_basis"]
    auxiliary = context["auxiliary"]

    scores: dict[str, int] = {topic: 0 for topic in _TOPIC_RULES}
    matched: dict[str, list[str]] = defaultdict(list)

    for topic, rule in _TOPIC_RULES.items():
        for keyword in rule["keywords"]:
            if keyword in claim_basis:
                scores[topic] += 2
                matched[topic].append(keyword)
            elif keyword in auxiliary:
                scores[topic] += 1
                matched[topic].append(keyword)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_topic, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score

    if top_score <= 0:
        return TopicResult(
            primary_topic=None,
            topic_confidence=None,
            topic_method="rule",
            topic_reason="no reliable keyword match",
            exploration_axes=infer_exploration_axes(record, primary_topic=None),
        )

    if margin == 0 and top_score <= 2:
        return TopicResult(
            primary_topic="unknown",
            topic_confidence="low",
            topic_method="rule",
            topic_reason="ambiguous topic signals",
            exploration_axes=infer_exploration_axes(record, primary_topic="unknown"),
        )

    if top_score >= 5 and margin >= 2:
        confidence = "high"
    elif top_score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    reason_keywords = ", ".join(sorted(set(matched[top_topic]))[:4])
    reason = f"matched keywords: {reason_keywords}" if reason_keywords else "rule match"

    return TopicResult(
        primary_topic=top_topic,
        topic_confidence=confidence,
        topic_method="rule",
        topic_reason=reason,
        exploration_axes=infer_exploration_axes(record, primary_topic=top_topic),
    )


def infer_baseline(record: dict[str, Any]) -> BaselineResult:
    context = _collect_context(record)
    text = context["combined"]
    if not text:
        return BaselineResult(None, None, None, None, None)

    has_compare = any(marker in text for marker in _COMPARE_MARKERS)
    has_baseline_word = "baseline" in text or "reference" in text

    gain_match = re.search(r"\bgain\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if gain_match and (has_compare or has_baseline_word):
        value = gain_match.group(1).replace(".", "_")
        return BaselineResult(
            baseline_label=f"gain_{value}",
            baseline_type="parameter",
            baseline_confidence="high" if has_compare else "medium",
            baseline_method="rule",
            baseline_reason="explicit compared-to phrase with parameter value",
        )

    threshold_match = re.search(r"\bthreshold\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if threshold_match and (has_compare or has_baseline_word):
        value = threshold_match.group(1).replace(".", "_")
        return BaselineResult(
            baseline_label=f"threshold_{value}",
            baseline_type="parameter",
            baseline_confidence="high" if has_compare else "medium",
            baseline_method="rule",
            baseline_reason="threshold candidate detected with comparison wording",
        )

    for label, pattern in (
        ("pre_cable_swap", r"\b(before|pre[- ]change|pre)\b.{0,24}\bcable\b"),
        ("post_cable_swap", r"\b(after|post[- ]change|post)\b.{0,24}\bcable\b"),
        ("pre_antenna_change", r"\b(before|pre[- ]change|pre)\b.{0,24}\bantenna\b"),
        ("post_antenna_change", r"\b(after|post[- ]change|post)\b.{0,24}\bantenna\b"),
        ("pre_filter", r"\b(before|pre[- ]change|pre)\b.{0,24}\bfilter\b"),
        ("post_filter", r"\b(after|post[- ]change|post)\b.{0,24}\bfilter\b"),
    ):
        if re.search(pattern, text):
            return BaselineResult(
                baseline_label=label,
                baseline_type="hardware",
                baseline_confidence="high",
                baseline_method="rule",
                baseline_reason="pre/post hardware wording detected",
            )

    temporal_match = re.search(r"\b(reference|baseline)\s+(?:window\s*)?(\d{4}[-_/]\d{2})\b", text)
    if temporal_match:
        ym = temporal_match.group(2).replace("-", "_").replace("/", "_")
        return BaselineResult(
            baseline_label=f"reference_window_{ym}",
            baseline_type="temporal",
            baseline_confidence="medium",
            baseline_method="rule",
            baseline_reason="reference window wording detected",
        )

    if has_compare:
        return BaselineResult(
            baseline_label=None,
            baseline_type="composite",
            baseline_confidence="low",
            baseline_method="rule",
            baseline_reason="comparison wording exists but baseline label is not safely extractable",
        )

    return BaselineResult(
        baseline_label=None,
        baseline_type=None,
        baseline_confidence=None,
        baseline_method="rule",
        baseline_reason="no explicit baseline markers",
    )


def should_skip_auto_enrich(record: dict[str, Any]) -> bool:
    status = normalize_optional_text(record.get("review_status"))
    return status in _HUMAN_LOCKED_STATUSES


def _derive_review_status(
    *,
    topic: str | None,
    topic_confidence: str | None,
    baseline_label: str | None,
    baseline_confidence: str | None,
) -> str:
    has_topic = normalize_optional_text(topic) is not None and topic != "unknown"
    has_baseline = normalize_optional_text(baseline_label) is not None
    if has_topic or has_baseline:
        if _normalize_confidence(topic_confidence) == "high" or _normalize_confidence(baseline_confidence) == "high":
            return "auto_high_confidence"
        return "needs_review"
    return "auto_unreviewed"


def enrich_claim_record(
    record: dict[str, Any],
    *,
    topic_only: bool = False,
    baseline_only: bool = False,
) -> dict[str, Any]:
    current_topic = normalize_optional_text(record.get("topic"))
    current_topic_confidence = normalize_optional_text(record.get("topic_confidence"))
    current_baseline_label = normalize_optional_text(record.get("baseline_label"))
    current_baseline_confidence = normalize_optional_text(record.get("baseline_confidence"))

    updates: dict[str, Any] = {}
    topic_result: TopicResult | None = None
    baseline_result: BaselineResult | None = None

    if not baseline_only:
        topic_result = infer_primary_topic(record)
        updates["topic"] = topic_result.primary_topic
        updates["topic_confidence"] = _normalize_confidence(topic_result.topic_confidence)
        updates["topic_method"] = topic_result.topic_method
        updates["topic_reason"] = topic_result.topic_reason
        updates["exploration_axes_json"] = (
            json.dumps(topic_result.exploration_axes, ensure_ascii=False)
            if topic_result.exploration_axes is not None
            else None
        )

    if not topic_only:
        baseline_result = infer_baseline(record)
        updates["baseline_label"] = (
            _normalize_label_token(baseline_result.baseline_label)
            if baseline_result.baseline_label is not None
            else None
        )
        updates["baseline_type"] = baseline_result.baseline_type
        updates["baseline_confidence"] = _normalize_confidence(baseline_result.baseline_confidence)
        updates["baseline_method"] = baseline_result.baseline_method
        updates["baseline_reason"] = baseline_result.baseline_reason

    topic_for_status = updates.get("topic", current_topic)
    topic_conf_for_status = updates.get("topic_confidence", current_topic_confidence)
    baseline_for_status = updates.get("baseline_label", current_baseline_label)
    baseline_conf_for_status = updates.get("baseline_confidence", current_baseline_confidence)
    updates["review_status"] = _derive_review_status(
        topic=topic_for_status,
        topic_confidence=topic_conf_for_status,
        baseline_label=baseline_for_status,
        baseline_confidence=baseline_conf_for_status,
    )

    return updates


def _fetch_claims(
    conn: sqlite3.Connection,
    *,
    ids: Sequence[int] | None,
    where_review_status: str | None,
    only_unreviewed: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          id,
          claim,
          basis,
          raw_text,
          evidence_refs,
          topic,
          topic_confidence,
          topic_method,
          topic_reason,
          baseline_label,
          baseline_type,
          baseline_confidence,
          baseline_method,
          baseline_reason,
          exploration_axes_json,
          fixed_conditions_json,
          variable_conditions_json,
          validity_scope,
          review_status
        FROM claims
        ORDER BY id
        """
    ).fetchall()

    required_files_by_claim: dict[int, list[str]] = defaultdict(list)
    for rf in conn.execute("SELECT claim_id, file_name FROM claim_required_files ORDER BY id").fetchall():
        required_files_by_claim[int(rf["claim_id"])].append(str(rf["file_name"]))

    id_filter = set(ids or [])
    normalized_where_status = normalize_optional_text(where_review_status)
    selected: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["required_file_names"] = required_files_by_claim.get(int(item["id"]), [])

        if id_filter and int(item["id"]) not in id_filter:
            continue

        status = normalize_optional_text(item.get("review_status"))
        if normalized_where_status is not None and status != normalized_where_status:
            continue

        if only_unreviewed and status not in _AUTO_MUTABLE_STATUSES:
            continue

        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break

    return selected


def _prune_non_destructive_updates(record: dict[str, Any], candidate_updates: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key, value in candidate_updates.items():
        current = record.get(key)
        if value is None and current is not None:
            continue
        if value == current:
            continue
        updates[key] = value
    return updates


def _apply_updates(conn: sqlite3.Connection, claim_id: int, updates: dict[str, Any]) -> None:
    assignments = ", ".join(f"{name} = ?" for name in updates)
    values = [updates[name] for name in updates]
    values.append(claim_id)
    conn.execute(f"UPDATE claims SET {assignments} WHERE id = ?", values)


def enrich_claims(
    *,
    db_path: Path | None = None,
    topic_only: bool = False,
    baseline_only: bool = False,
    limit: int | None = None,
    where_review_status: str | None = None,
    dry_run: bool = False,
    ids: Sequence[int] | None = None,
    only_unreviewed: bool = False,
) -> EnrichReport:
    if topic_only and baseline_only:
        raise ValueError("topic_only and baseline_only cannot both be true.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when specified.")

    scanned_claims = 0
    updated_claims = 0
    skipped_human_locked = 0
    topic_assigned = 0
    baseline_assigned = 0
    needs_review_count = 0
    high_confidence_count = 0

    resolved_db_path = db_path or DB_PATH
    with _connect_for_enrich(db_path) as conn:
        ensure_schema(conn)
        records = _fetch_claims(
            conn,
            ids=ids,
            where_review_status=where_review_status,
            only_unreviewed=only_unreviewed,
            limit=limit,
        )
        scanned_claims = len(records)

        for record in records:
            if should_skip_auto_enrich(record):
                skipped_human_locked += 1
                continue

            candidate = enrich_claim_record(
                record,
                topic_only=topic_only,
                baseline_only=baseline_only,
            )
            updates = _prune_non_destructive_updates(record, candidate)
            if not updates:
                continue

            topic_value = updates.get("topic", record.get("topic"))
            if normalize_optional_text(topic_value) is not None and topic_value != "unknown":
                topic_assigned += 1

            baseline_value = updates.get("baseline_label", record.get("baseline_label"))
            if normalize_optional_text(baseline_value) is not None:
                baseline_assigned += 1

            status_value = updates.get("review_status", record.get("review_status"))
            normalized_status = normalize_optional_text(status_value)
            if normalized_status == "needs_review":
                needs_review_count += 1
            if normalized_status == "auto_high_confidence":
                high_confidence_count += 1

            updated_claims += 1
            if not dry_run:
                _apply_updates(conn, claim_id=int(record["id"]), updates=updates)

        if not dry_run:
            conn.commit()

    _ = resolved_db_path  # Keep value for future audit logging extension.
    return EnrichReport(
        scanned_claims=scanned_claims,
        updated_claims=updated_claims,
        skipped_human_locked=skipped_human_locked,
        topic_assigned=topic_assigned,
        baseline_assigned=baseline_assigned,
        needs_review_count=needs_review_count,
        high_confidence_count=high_confidence_count,
        dry_run=dry_run,
    )
