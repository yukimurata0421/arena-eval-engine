from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from arena.synthesis.cluster_baselines import normalize_baseline_label
from arena.synthesis.normalization import normalize_optional_text, normalize_priority_hint
from arena.synthesis.paths import DB_PATH

_HUMAN_LOCKED_STATUSES = {"human_confirmed", "human_corrected"}
_NULL_BUCKET = "__null__"


def _connect_readonly(db_path: Path | None = None) -> sqlite3.Connection:
    target = db_path or DB_PATH
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_json_loads_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]
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
    return [str(item).strip() for item in parsed if isinstance(item, str) and str(item).strip()]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _bucket(value: Any) -> str:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return _NULL_BUCKET
    return normalized


def summarize_topic_distribution(
    claims: list[dict[str, Any]],
    claim_to_clusters: dict[int, set[int]],
    mixed_cluster_ids: set[int],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    topic_cluster_sets: dict[str, set[int]] = defaultdict(set)

    for claim in claims:
        topic = _bucket(claim.get("topic"))
        row = rows.setdefault(
            topic,
            {
                "topic": topic,
                "claim_count": 0,
                "baseline_cluster_count": 0,
                "no_baseline_count": 0,
                "high_priority_no_baseline_count": 0,
                "mixed_cluster_count": 0,
            },
        )
        row["claim_count"] += 1

        claim_id = int(claim["id"])
        clusters = claim_to_clusters.get(claim_id, set())
        topic_cluster_sets[topic].update(clusters)

        missing_baseline = bool(claim.get("is_missing_baseline"))
        if missing_baseline:
            row["no_baseline_count"] += 1

        if bool(claim.get("is_high_priority_missing_baseline")):
            row["high_priority_no_baseline_count"] += 1

    for topic, row in rows.items():
        topic_clusters = topic_cluster_sets.get(topic, set())
        row["baseline_cluster_count"] = len(topic_clusters)
        row["mixed_cluster_count"] = len(topic_clusters & mixed_cluster_ids)

    return sorted(rows.values(), key=lambda x: (-int(x["claim_count"]), str(x["topic"])))


def find_mixed_clusters(cluster_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [detail for detail in cluster_details if bool(detail.get("has_mixed_claim_types"))]


def find_high_priority_missing_baseline(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": int(claim["id"]),
            "claim": claim.get("claim"),
            "topic": claim.get("topic"),
            "priority_hint": claim.get("priority_hint"),
            "review_status": claim.get("review_status"),
            "evidence_files": claim.get("evidence_files_summary"),
            "created_at": claim.get("created_at"),
        }
        for claim in claims
        if bool(claim.get("is_high_priority_missing_baseline"))
    ]


def build_baseline_report(
    *,
    db_path: Path | None = None,
    limit: int = 20,
    topic: str | None = None,
    only_mixed: bool = False,
    only_high_priority_missing: bool = False,
    include_cluster_details: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be >= 1.")

    normalized_topic_filter = normalize_optional_text(topic)

    with _connect_readonly(db_path=db_path) as conn:
        claims_columns = _table_columns(conn, "claims")

        def _col(name: str, alias: str | None = None) -> str:
            column_alias = alias or name
            if name in claims_columns:
                return f"h.{name} AS {column_alias}"
            return f"NULL AS {column_alias}"

        claim_query = """
            SELECT
              h.id AS id,
              h.claim AS claim,
              {claim_type_col},
              {topic_col},
              {priority_hint_col},
              {review_status_col},
              {baseline_label_col},
              {created_at_col},
              {evidence_refs_col}
            FROM claims h
        """.format(
            claim_type_col=_col("claim_type"),
            topic_col=_col("topic"),
            priority_hint_col=_col("priority_hint"),
            review_status_col=_col("review_status"),
            baseline_label_col=_col("baseline_label"),
            created_at_col=_col("created_at"),
            evidence_refs_col=_col("evidence_refs"),
        )
        params: list[Any] = []
        if normalized_topic_filter is not None:
            claim_query += " WHERE h.topic = ?"
            params.append(normalized_topic_filter)
        claim_query += " ORDER BY h.id"
        claim_rows = conn.execute(claim_query, params).fetchall()

        claims: list[dict[str, Any]] = [dict(row) for row in claim_rows]
        claims_by_id = {int(claim["id"]): claim for claim in claims}

        required_files_by_claim: dict[int, list[str]] = defaultdict(list)
        required_params: list[Any] = []
        if "claim_required_files" in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
            required_query = """
                SELECT rf.claim_id, rf.file_name
                FROM claim_required_files rf
                JOIN claims h ON h.id = rf.claim_id
            """
            if normalized_topic_filter is not None:
                required_query += " WHERE h.topic = ?"
                required_params.append(normalized_topic_filter)
            for row in conn.execute(required_query, required_params).fetchall():
                required_files_by_claim[int(row["claim_id"])].append(str(row["file_name"]))

        link_query = """
            SELECT
              l.claim_id,
              l.baseline_cluster_id,
              l.link_confidence,
              bc.label,
              bc.type,
              bc.confidence
            FROM claim_baseline_links l
            JOIN baseline_clusters bc ON bc.id = l.baseline_cluster_id
            JOIN claims h ON h.id = l.claim_id
        """
        link_params: list[Any] = []
        if normalized_topic_filter is not None:
            link_query += " WHERE h.topic = ?"
            link_params.append(normalized_topic_filter)
        link_query += " ORDER BY l.baseline_cluster_id, l.claim_id"
        link_rows = conn.execute(link_query, link_params).fetchall()

        claim_to_clusters: dict[int, set[int]] = defaultdict(set)
        cluster_links: dict[int, list[dict[str, Any]]] = defaultdict(list)
        clusters_meta: dict[int, dict[str, Any]] = {}
        for row in link_rows:
            claim_id = int(row["claim_id"])
            cluster_id = int(row["baseline_cluster_id"])
            claim_to_clusters[claim_id].add(cluster_id)
            cluster_links[cluster_id].append(
                {
                    "claim_id": claim_id,
                    "link_confidence": normalize_optional_text(row["link_confidence"]),
                }
            )
            clusters_meta[cluster_id] = {
                "cluster_id": cluster_id,
                "label": row["label"],
                "type": row["type"],
                "confidence": row["confidence"],
            }

        for claim in claims:
            claim_id = int(claim["id"])
            normalized_label = normalize_baseline_label(claim.get("baseline_label"))
            has_link = claim_id in claim_to_clusters and bool(claim_to_clusters[claim_id])
            is_missing_baseline = (not has_link) or normalized_label is None
            priority_hint = normalize_priority_hint(claim.get("priority_hint"))
            review_status = normalize_optional_text(claim.get("review_status"))
            is_high_priority_missing = (
                is_missing_baseline
                and priority_hint == "high"
                and review_status not in _HUMAN_LOCKED_STATUSES
            )
            evidence_refs = _safe_json_loads_list(claim.get("evidence_refs"))
            required_files = required_files_by_claim.get(claim_id, [])
            claim["is_missing_baseline"] = is_missing_baseline
            claim["is_high_priority_missing_baseline"] = is_high_priority_missing
            claim["priority_hint"] = priority_hint
            claim["evidence_files_summary"] = required_files or evidence_refs[:3]

        cluster_details: list[dict[str, Any]] = []
        for cluster_id, meta in sorted(clusters_meta.items(), key=lambda item: item[0]):
            linked = cluster_links.get(cluster_id, [])
            linked_ids = {int(x["claim_id"]) for x in linked}
            cluster_claims = [claims_by_id[claim_id] for claim_id in linked_ids if claim_id in claims_by_id]
            claim_type_counter = Counter(_bucket(claim.get("claim_type")) for claim in cluster_claims)
            priority_counter = Counter(_bucket(claim.get("priority_hint")) for claim in cluster_claims)
            review_counter = Counter(_bucket(claim.get("review_status")) for claim in cluster_claims)
            topic_set = {_bucket(claim.get("topic")) for claim in cluster_claims}
            has_mixed = claim_type_counter.get("supported", 0) > 0 and claim_type_counter.get("negative", 0) > 0

            cluster_details.append(
                {
                    "cluster_id": cluster_id,
                    "label": meta.get("label"),
                    "type": meta.get("type"),
                    "confidence": meta.get("confidence"),
                    "claim_count": len(cluster_claims),
                    "topic_count": len(topic_set),
                    "claim_ids": sorted(linked_ids),
                    "topics": sorted(topic_set),
                    "claim_type_breakdown": dict(sorted(claim_type_counter.items())),
                    "priority_hint_breakdown": dict(sorted(priority_counter.items())),
                    "review_status_breakdown": dict(sorted(review_counter.items())),
                    "has_mixed_claim_types": has_mixed,
                }
            )

        mixed_clusters = find_mixed_clusters(cluster_details)
        mixed_cluster_ids = {int(row["cluster_id"]) for row in mixed_clusters}
        topic_distribution = summarize_topic_distribution(claims, claim_to_clusters, mixed_cluster_ids)
        high_priority_missing = find_high_priority_missing_baseline(claims)

        if normalized_topic_filter is None:
            total_clusters = int(conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0])
        else:
            total_clusters = len({int(row["baseline_cluster_id"]) for row in link_rows})

        total_claim_links = len(link_rows)
        total_claims_with_baseline = sum(
            1
            for claim in claims
            if (int(claim["id"]) in claim_to_clusters and bool(claim_to_clusters[int(claim["id"])]))
            or normalize_baseline_label(claim.get("baseline_label")) is not None
        )
        total_claims_without_baseline = len(claims) - total_claims_with_baseline
        total_high_priority_without_baseline = len(high_priority_missing)
        total_mixed_clusters = len(mixed_clusters)
        total_auto_unreviewed_claims = sum(1 for claim in claims if _bucket(claim.get("review_status")) == "auto_unreviewed")
        total_needs_review_claims = sum(1 for claim in claims if _bucket(claim.get("review_status")) == "needs_review")

        summary = {
            "total_clusters": total_clusters,
            "total_claim_links": total_claim_links,
            "total_claims_with_baseline": total_claims_with_baseline,
            "total_claims_without_baseline": total_claims_without_baseline,
            "total_high_priority_without_baseline": total_high_priority_without_baseline,
            "total_mixed_clusters": total_mixed_clusters,
            "total_auto_unreviewed_claims": total_auto_unreviewed_claims,
            "total_needs_review_claims": total_needs_review_claims,
        }

        def _limit_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return items[:limit]

        output_topic_distribution = _limit_rows(topic_distribution)
        output_mixed_clusters = _limit_rows(sorted(mixed_clusters, key=lambda x: (-int(x["claim_count"]), int(x["cluster_id"]))))
        output_missing = _limit_rows(high_priority_missing)
        output_cluster_details = _limit_rows(sorted(cluster_details, key=lambda x: (-int(x["claim_count"]), int(x["cluster_id"]))))

        if only_mixed:
            output_missing = []
        if only_high_priority_missing:
            output_mixed_clusters = []
            if not include_cluster_details:
                output_cluster_details = []
        if not include_cluster_details:
            output_cluster_details = []

        return {
            "summary": summary,
            "topic_distribution": output_topic_distribution,
            "mixed_clusters": output_mixed_clusters,
            "high_priority_without_baseline": output_missing,
            "cluster_details": output_cluster_details,
            "filters": {
                "topic": normalized_topic_filter,
                "only_mixed": bool(only_mixed),
                "only_high_priority_missing": bool(only_high_priority_missing),
                "include_cluster_details": bool(include_cluster_details),
                "limit": int(limit),
            },
        }


def render_baseline_report_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines: list[str] = []
    lines.append("Baseline Report")
    lines.append(
        "summary: "
        + " ".join(
            f"{k}={summary.get(k, 0)}"
            for k in (
                "total_clusters",
                "total_claim_links",
                "total_claims_with_baseline",
                "total_claims_without_baseline",
                "total_high_priority_without_baseline",
                "total_mixed_clusters",
                "total_auto_unreviewed_claims",
                "total_needs_review_claims",
            )
        )
    )

    topic_distribution = report.get("topic_distribution", [])
    lines.append("topic_distribution:")
    if not topic_distribution:
        lines.append("- none")
    else:
        for row in topic_distribution:
            lines.append(
                f"- topic={row['topic']} claim_count={row['claim_count']} "
                f"baseline_cluster_count={row['baseline_cluster_count']} no_baseline_count={row['no_baseline_count']} "
                f"high_priority_no_baseline_count={row['high_priority_no_baseline_count']} mixed_cluster_count={row['mixed_cluster_count']}"
            )

    mixed_clusters = report.get("mixed_clusters", [])
    lines.append("mixed_clusters:")
    if not mixed_clusters:
        lines.append("- none")
    else:
        for row in mixed_clusters:
            lines.append(
                f"- cluster_id={row['cluster_id']} label={row.get('label')} type={row.get('type')} "
                f"claim_count={row['claim_count']} has_mixed_claim_types={int(bool(row.get('has_mixed_claim_types')))}"
            )

    missing = report.get("high_priority_without_baseline", [])
    lines.append("high_priority_without_baseline:")
    if not missing:
        lines.append("- none")
    else:
        for row in missing:
            lines.append(
                f"- hypothesis_id={row['hypothesis_id']} topic={_bucket(row.get('topic'))} "
                f"priority_hint={row.get('priority_hint')} review_status={row.get('review_status')} "
                f"created_at={row.get('created_at')}"
            )

    cluster_details = report.get("cluster_details", [])
    if cluster_details:
        lines.append("cluster_details:")
        for row in cluster_details:
            lines.append(
                f"- cluster_id={row['cluster_id']} label={row.get('label')} confidence={row.get('confidence')} "
                f"claim_type_breakdown={json.dumps(row.get('claim_type_breakdown', {}), ensure_ascii=False)}"
            )

    return "\n".join(lines)


def render_baseline_report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
