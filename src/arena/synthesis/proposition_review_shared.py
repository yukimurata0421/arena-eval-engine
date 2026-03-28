from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.synthesis.db import connect
from arena.synthesis.normalization import normalize_optional_text

SUGGESTED_REVIEW_STATUS_VOCAB: tuple[str, ...] = (
    "auto_accept_candidate",
    "human_review_required",
    "hold",
    "reject_candidate",
)
VALIDITY_ASSESSMENT_VOCAB: tuple[str, ...] = ("strong", "moderate", "weak", "conflicting")
PROPOSITION_REVIEW_STATUS_VOCAB: tuple[str, ...] = (
    "pending",
    "triaged",
    "human_review_required",
    "human_confirmed",
    "human_corrected",
    "human_rejected",
    "on_hold",
)
CONSENSUS_CLASS_VOCAB: tuple[str, ...] = (
    "cross_model_supported",
    "same_model_repeated",
    "cross_model_conflicting",
    "weak_sparse",
)

HUMAN_LOCKED_STATUSES = {"human_confirmed", "human_corrected", "human_rejected"}
ASSERTIVE_TOKENS: tuple[str, ...] = (
    "will",
    "guarantee",
    "definitely",
    "proves",
    "clearly",
    "must",
)
IMPACT_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.35}
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"triaged", "human_review_required", "on_hold"},
    "triaged": {"human_review_required", "human_confirmed", "human_corrected", "human_rejected", "on_hold"},
    "human_review_required": {"human_confirmed", "human_corrected", "human_rejected", "on_hold"},
    "on_hold": {"triaged", "human_review_required", "human_confirmed", "human_corrected", "human_rejected"},
    "human_confirmed": set(),
    "human_corrected": set(),
    "human_rejected": set(),
}


@dataclass(frozen=True, slots=True)
class ClaimRef:
    source_ai: str
    source_file: str
    claim_id: str
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class PropositionSnapshot:
    proposition_id: str
    question: str
    target: str
    convergence: str
    resolved_type: str
    priority_hint: str
    caveat: str | None
    claim_refs: tuple[ClaimRef, ...]


@dataclass(frozen=True, slots=True)
class TriagePropositionsReport:
    triage_run_id: str
    scanned_propositions: int
    triage_records_written: int
    queue_candidates: int
    output_path: str
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ReviewQueueExportReport:
    exported_rows: int
    output_path: str
    format: str
    include_final: bool


@dataclass(frozen=True, slots=True)
class PropositionReviewStatusUpdateReport:
    requested_status: str
    matched_records: int
    updated_records: int
    unchanged_records: int
    skipped_locked_records: int
    skipped_missing_records: int
    skipped_transition_records: int
    history_records_written: int
    reason_used: str | None
    dry_run: bool
    updated_ids: tuple[str, ...]
    skipped_locked_ids: tuple[str, ...]
    skipped_missing_ids: tuple[str, ...]
    skipped_transition_ids: tuple[str, ...]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def now_compact_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def connect_for_update(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def safe_json_loads_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = normalize_optional_text(value)
        return [token] if token is not None else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        token = normalize_optional_text(item)
        if token is not None:
            result.append(token)
    return result


def normalize_claim_type(value: Any) -> str:
    token = normalize_optional_text(value)
    return token.casefold() if token is not None else "unknown"


def contains_assertive_text(text: str) -> bool:
    lowered = text.casefold()
    return any(token in lowered for token in ASSERTIVE_TOKENS)


def resolved_family(resolved_type: str) -> str:
    lowered = str(resolved_type).casefold()
    if lowered in {"supported", "supported_with_caveat"}:
        return "supportive"
    if lowered == "negative":
        return "negative"
    return "uncertain"


def collect_next_data_needed(payload: dict[str, Any]) -> list[str]:
    if "next_data_needed" not in payload:
        return []
    value = payload.get("next_data_needed")
    if isinstance(value, list):
        return normalize_string_list(value)
    token = normalize_optional_text(value)
    if token is None:
        return []
    return [token]


def derive_stance(claim_type: str) -> str:
    lowered = claim_type.casefold()
    if lowered == "supported":
        return "supported"
    if lowered == "negative":
        return "negative"
    if lowered == "future":
        return "future"
    return "unknown"


def normalize_metric_keys(value: Any) -> list[str]:
    metrics = safe_json_loads_list(value) if isinstance(value, str) else value
    if not isinstance(metrics, list):
        return []
    keys: set[str] = set()
    for item in metrics:
        if isinstance(item, dict):
            metric = normalize_optional_text(item.get("metric"))
            if metric is not None:
                keys.add(metric.casefold())
    return sorted(keys)


def build_evidence_signature(payload: dict[str, Any]) -> str:
    evidence_files = sorted({item.casefold() for item in normalize_string_list(payload.get("evidence_files"))})
    evidence_refs = sorted({item.casefold() for item in normalize_string_list(payload.get("evidence_refs"))})
    metric_keys = normalize_metric_keys(payload.get("metrics_used"))
    signature_payload = {
        "files": evidence_files,
        "refs": evidence_refs,
        "metrics": metric_keys,
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def same_model_stability_bonus(model_counts: dict[str, int]) -> float:
    bonus = 0.0
    for count in model_counts.values():
        repeats = max(0, count - 1)
        if repeats >= 1:
            bonus += 0.015
        if repeats >= 2:
            bonus += 0.01
    return min(0.04, bonus)


def load_integrated_claim_refs(enriched_dir: Path) -> tuple[dict[str, tuple[ClaimRef, ...]], list[str]]:
    if not enriched_dir.exists() or not enriched_dir.is_dir():
        raise NotADirectoryError(f"enriched directory does not exist or is not a directory: {enriched_dir}")

    proposition_claim_refs: dict[str, list[ClaimRef]] = defaultdict(list)
    loaded_files: list[str] = []
    for path in sorted(enriched_dir.glob("*.json")):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        meta = parsed.get("meta")
        claims = parsed.get("claims")
        if not isinstance(meta, dict) or not isinstance(claims, list):
            continue
        source_ai = normalize_optional_text(meta.get("source_ai")) or path.stem.casefold()
        loaded_files.append(str(path))
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = normalize_optional_text(claim.get("claim_id"))
            if claim_id is None:
                continue
            source_file = normalize_optional_text(claim.get("source_file")) or str(path.name)
            proposition_ids = normalize_string_list(claim.get("proposition_ids"))
            if not proposition_ids:
                continue
            ref = ClaimRef(
                source_ai=source_ai.casefold(),
                source_file=source_file,
                claim_id=claim_id,
                payload=claim,
            )
            for proposition_id in proposition_ids:
                proposition_claim_refs[proposition_id].append(ref)

    normalized_refs: dict[str, tuple[ClaimRef, ...]] = {}
    for proposition_id, refs in proposition_claim_refs.items():
        refs_sorted = sorted(refs, key=lambda item: (item.source_ai, item.source_file, item.claim_id))
        normalized_refs[proposition_id] = tuple(refs_sorted)
    return normalized_refs, loaded_files


def fetch_proposition_snapshots(
    conn: sqlite3.Connection,
    *,
    proposition_claim_refs: dict[str, tuple[ClaimRef, ...]],
) -> list[PropositionSnapshot]:
    rows = conn.execute(
        """
        SELECT
          p.proposition_id,
          p.question,
          p.target,
          p.convergence,
          p.resolved_type,
          p.priority_hint,
          p.caveat
        FROM propositions p
        ORDER BY p.proposition_id
        """
    ).fetchall()
    snapshots: list[PropositionSnapshot] = []
    for row in rows:
        proposition_id = str(row["proposition_id"])
        claim_refs = proposition_claim_refs.get(proposition_id, tuple())
        snapshots.append(
            PropositionSnapshot(
                proposition_id=proposition_id,
                question=str(row["question"]),
                target=str(row["target"]),
                convergence=str(row["convergence"]),
                resolved_type=str(row["resolved_type"]),
                priority_hint=str(row["priority_hint"]),
                caveat=normalize_optional_text(row["caveat"]),
                claim_refs=claim_refs,
            )
        )
    return snapshots


def default_triage_path(*, review_dir: Path, triage_run_id: str) -> Path:
    return review_dir / "triage" / f"proposition_triage_{triage_run_id}.jsonl"


def default_queue_path(*, review_dir: Path, format_name: str) -> Path:
    suffix = "csv" if format_name == "csv" else "jsonl"
    return review_dir / "queue" / f"review_queue_{now_compact_utc()}.{suffix}"
