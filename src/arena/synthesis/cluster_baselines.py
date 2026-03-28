from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.synthesis.db import connect, ensure_schema
from arena.synthesis.normalization import normalize_optional_text

_CONFIDENCE_SCORE = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

_LABEL_SYNONYMS = {
    "pre_filter": "pre_filter",
    "prefilter": "pre_filter",
    "post_filter": "post_filter",
    "postfilter": "post_filter",
}


@dataclass(frozen=True, slots=True)
class BaselineClusterReport:
    scanned_claims: int
    cluster_count: int
    claim_link_count: int
    high_count: int
    medium_count: int
    low_count: int
    dry_run: bool


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_baseline_label(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    token = normalized.casefold()
    token = token.replace(".", "_").replace("-", "_").replace("/", "_")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return None
    return _LABEL_SYNONYMS.get(token, token)


def _normalize_confidence(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if lowered in _CONFIDENCE_SCORE:
        return lowered
    return None


def _compute_cluster_confidence(confidences: list[str | None]) -> str:
    if not confidences:
        return "low"

    scores = [_CONFIDENCE_SCORE.get(conf or "", 0) for conf in confidences]
    avg = (sum(scores) / len(scores)) if scores else 0.0
    count = len(confidences)
    if count >= 3 and avg >= _CONFIDENCE_SCORE["medium"]:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def _connect_for_cluster(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def cluster_baselines(
    *,
    db_path: Path | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
) -> BaselineClusterReport:
    with _connect_for_cluster(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT id, baseline_label, baseline_type, baseline_confidence
            FROM claims
            WHERE baseline_label IS NOT NULL AND TRIM(baseline_label) <> ''
            ORDER BY id
            """
        ).fetchall()

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            label = normalize_baseline_label(row["baseline_label"])
            if label is None:
                continue
            groups[label].append(
                {
                    "claim_id": int(row["id"]),
                    "baseline_type": normalize_optional_text(row["baseline_type"]),
                    "baseline_confidence": _normalize_confidence(row["baseline_confidence"]),
                }
            )

        cluster_specs: list[dict[str, Any]] = []
        high_count = 0
        medium_count = 0
        low_count = 0
        potential_link_count = 0

        for label in sorted(groups):
            members = groups[label]
            confidences = [m["baseline_confidence"] for m in members]
            cluster_confidence = _compute_cluster_confidence(confidences)
            type_counter = Counter(m["baseline_type"] for m in members if m["baseline_type"])
            cluster_type = type_counter.most_common(1)[0][0] if type_counter else None
            cluster_reason = f"normalized baseline label grouping across {len(members)} claim(s)"
            links = [
                {
                    "claim_id": m["claim_id"],
                    "link_confidence": m["baseline_confidence"] or cluster_confidence,
                }
                for m in members
            ]
            potential_link_count += len(links)

            if cluster_confidence == "high":
                high_count += 1
            elif cluster_confidence == "medium":
                medium_count += 1
            else:
                low_count += 1

            cluster_specs.append(
                {
                    "label": label,
                    "type": cluster_type,
                    "confidence": cluster_confidence,
                    "cluster_reason": cluster_reason,
                    "links": links,
                }
            )

        if dry_run:
            return BaselineClusterReport(
                scanned_claims=len(rows),
                cluster_count=len(cluster_specs),
                claim_link_count=potential_link_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                dry_run=True,
            )

        if rebuild:
            conn.execute("DELETE FROM claim_baseline_links")
            conn.execute("DELETE FROM baseline_clusters")

        existing_clusters = {
            str(row["label"]): int(row["id"])
            for row in conn.execute("SELECT id, label FROM baseline_clusters").fetchall()
        }

        inserted_links = 0
        for spec in cluster_specs:
            cluster_id = existing_clusters.get(spec["label"])
            if cluster_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO baseline_clusters (label, type, confidence, cluster_reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        spec["label"],
                        spec["type"],
                        spec["confidence"],
                        spec["cluster_reason"],
                        _now_iso(),
                    ),
                )
                cluster_id = int(cur.lastrowid)
                existing_clusters[spec["label"]] = cluster_id

            for link in spec["links"]:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO claim_baseline_links (claim_id, baseline_cluster_id, link_confidence)
                    VALUES (?, ?, ?)
                    """,
                    (link["claim_id"], cluster_id, link["link_confidence"]),
                )
                inserted_links += int(cur.rowcount)

        conn.commit()
        return BaselineClusterReport(
            scanned_claims=len(rows),
            cluster_count=len(cluster_specs),
            claim_link_count=inserted_links,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            dry_run=False,
        )
