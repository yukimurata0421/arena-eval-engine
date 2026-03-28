from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arena.synthesis.db import ensure_schema
from arena.synthesis.normalization import normalize_optional_text
from arena.synthesis.paths import ENRICHED_DIR, REVIEW_DIR
from arena.synthesis.proposition_review_shared import (
    CONSENSUS_CLASS_VOCAB,
    IMPACT_WEIGHT,
    SUGGESTED_REVIEW_STATUS_VOCAB,
    VALIDITY_ASSESSMENT_VOCAB,
    PropositionSnapshot,
    TriagePropositionsReport,
    build_evidence_signature,
    clamp01,
    collect_next_data_needed,
    connect_for_update,
    contains_assertive_text,
    default_triage_path,
    derive_stance,
    fetch_proposition_snapshots,
    load_integrated_claim_refs,
    normalize_claim_type,
    normalize_string_list,
    now_compact_utc,
    now_iso,
    resolved_family,
    safe_json_loads_list,
    same_model_stability_bonus,
)


def _build_triage_record(
    proposition: PropositionSnapshot,
    *,
    triage_run_id: str,
    created_at: str,
    cross_conflict_refs: list[str],
    source_files: Sequence[str],
) -> dict[str, Any]:
    claim_count = len(proposition.claim_refs)
    missing_payload_count = 0
    missing_evidence_files_count = 0
    missing_evidence_refs_count = 0
    supported_without_metrics_count = 0
    missing_limitations_count = 0
    assertive_unknown_future_count = 0
    confidence_assertiveness_mismatch_count = 0
    metadata_incomplete_count = 0
    support_claim_count = 0
    negative_claim_count = 0
    unknown_future_claim_count = 0
    direct_evidence_claim_count = 0
    total_evidence_files = 0
    total_evidence_refs = 0
    total_metrics = 0
    limitations_present_count = 0
    support_units: list[float] = []
    next_data_needed: list[str] = []
    source_file_keys: set[tuple[str, str]] = set()
    model_counts: dict[str, int] = defaultdict(int)
    evidence_signatures: set[str] = set()
    supporting_evidence_signatures: set[str] = set()
    support_models: set[str] = set()
    negative_models: set[str] = set()
    observation_rows: list[dict[str, Any]] = []

    for ref in proposition.claim_refs:
        model_counts[ref.source_ai] += 1
        source_file_keys.add((ref.source_ai, ref.source_file))
        payload = ref.payload
        if payload is None:
            missing_payload_count += 1
            continue

        claim_type = normalize_claim_type(payload.get("claim_type"))
        stance = derive_stance(claim_type)
        claim_text = normalize_optional_text(payload.get("claim")) or ""
        basis_text = normalize_optional_text(payload.get("basis_summary")) or normalize_optional_text(payload.get("basis")) or ""
        evidence_files = normalize_string_list(payload.get("evidence_files"))
        evidence_refs = normalize_string_list(payload.get("evidence_refs"))
        if isinstance(payload.get("metrics_used"), str):
            metrics_used = safe_json_loads_list(payload.get("metrics_used"))
        else:
            metrics_used = payload.get("metrics_used")
        if not isinstance(metrics_used, list):
            metrics_used = []
        limitation = normalize_optional_text(payload.get("limitation_or_counterpoint"))
        has_limitation = limitation is not None
        evidence_signature = build_evidence_signature(payload)
        metadata_ok = (
            normalize_optional_text(payload.get("claim")) is not None
            and normalize_optional_text(payload.get("claim_type")) is not None
        )

        metric_count = len(metrics_used)
        total_metrics += metric_count
        total_evidence_files += len(evidence_files)
        total_evidence_refs += len(evidence_refs)
        evidence_signatures.add(evidence_signature)

        if evidence_files:
            direct_evidence_claim_count += 1
        else:
            missing_evidence_files_count += 1
        if not evidence_refs:
            missing_evidence_refs_count += 1
        if limitation is None:
            missing_limitations_count += 1
        else:
            limitations_present_count += 1
        if not metadata_ok:
            metadata_incomplete_count += 1

        if stance == "supported":
            support_claim_count += 1
            support_models.add(ref.source_ai)
            supporting_evidence_signatures.add(evidence_signature)
            if metric_count == 0:
                supported_without_metrics_count += 1
        elif stance == "negative":
            negative_claim_count += 1
            negative_models.add(ref.source_ai)
        elif stance in {"unknown", "future"}:
            unknown_future_claim_count += 1
            if contains_assertive_text(f"{claim_text} {basis_text}"):
                assertive_unknown_future_count += 1
                if normalize_optional_text(payload.get("evidence_level")) != "direct":
                    confidence_assertiveness_mismatch_count += 1

        next_data_needed.extend(collect_next_data_needed(payload))

        unit = 0.0
        if evidence_files:
            unit += 0.35
        if evidence_refs:
            unit += 0.20
        if metric_count > 0:
            unit += 0.25
        if limitation is not None:
            unit += 0.10
        if claim_type == "supported":
            unit += 0.10
        if claim_type in {"unknown", "future"}:
            unit -= 0.10
        support_units.append(clamp01(unit))
        observation_rows.append(
            {
                "source_ai": ref.source_ai,
                "stance": stance,
                "has_limitation": has_limitation,
                "evidence_signature": evidence_signature,
            }
        )

    unique_model_count = len(model_counts)
    unique_source_file_count = len(source_file_keys)
    unique_observation_count = claim_count
    unique_evidence_signature_count = len(evidence_signatures)
    same_model_repeat_count_max = max(0, max(model_counts.values()) - 1) if model_counts else 0
    supporting_observation_count = support_claim_count

    supporting_signature_count = len(supporting_evidence_signatures)
    support_signature_diversity = (
        supporting_signature_count / max(1, supporting_observation_count)
        if supporting_observation_count > 0
        else 0.0
    )

    has_cross_model_support = len(support_models) >= 2
    cross_model_negative_count = sum(
        1 for row in observation_rows if row["stance"] == "negative" and row["source_ai"] not in support_models
    )
    cross_model_uncertain_count = sum(
        1
        for row in observation_rows
        if row["stance"] in {"unknown", "future"} and row["has_limitation"] and row["source_ai"] not in support_models
    )
    contradicting_observation_count = (
        cross_model_negative_count + cross_model_uncertain_count if supporting_observation_count > 0 else 0
    )
    has_cross_model_conflict = supporting_observation_count > 0 and cross_model_negative_count > 0
    has_soft_cross_model_conflict = supporting_observation_count > 0 and cross_model_uncertain_count > 0

    base_evidence_score = sum(support_units) / len(support_units) if support_units else 0.0
    if claim_count > 0 and missing_payload_count > 0:
        base_evidence_score = clamp01(base_evidence_score - (0.20 * (missing_payload_count / claim_count)))

    same_model_bonus = same_model_stability_bonus(model_counts)
    cross_model_bonus_base = 0.0
    if len(support_models) >= 3:
        cross_model_bonus_base = 0.18
    elif len(support_models) == 2:
        cross_model_bonus_base = 0.12
    cross_model_bonus = cross_model_bonus_base * (0.6 + (0.4 * support_signature_diversity))

    if unique_evidence_signature_count > 1:
        evidence_diversity_bonus = min(
            0.06,
            0.03 * ((unique_evidence_signature_count - 1) / max(1, unique_observation_count - 1)),
        )
    else:
        evidence_diversity_bonus = 0.0

    contradiction_penalty = 0.0
    if has_cross_model_conflict:
        contradiction_penalty += 0.22
    if contradicting_observation_count > 0:
        contradiction_penalty += min(0.16, 0.04 * contradicting_observation_count)
    if has_soft_cross_model_conflict and not has_cross_model_conflict:
        contradiction_penalty += 0.08
    contradiction_penalty = min(0.40, contradiction_penalty)

    support_strength = clamp01(
        base_evidence_score + cross_model_bonus + evidence_diversity_bonus + same_model_bonus - contradiction_penalty
    )

    coverage_ratio = direct_evidence_claim_count / claim_count if claim_count > 0 else 0.0
    unknown_ratio = unknown_future_claim_count / claim_count if claim_count > 0 else 0.0
    internal_conflict = support_claim_count > 0 and negative_claim_count > 0
    has_cross_conflict = bool(cross_conflict_refs)

    key_issues: list[str] = []
    if missing_payload_count > 0:
        key_issues.append(f"claim payload missing for {missing_payload_count} relation(s)")
    if missing_evidence_files_count > 0:
        key_issues.append(f"missing evidence_files in {missing_evidence_files_count} claim(s)")
    if missing_evidence_refs_count > 0:
        key_issues.append(f"missing evidence_refs in {missing_evidence_refs_count} claim(s)")
    if supported_without_metrics_count > 0:
        key_issues.append(f"supported claims without metrics_used: {supported_without_metrics_count}")
    if missing_limitations_count > 0:
        key_issues.append(f"limitations absent in {missing_limitations_count} claim(s)")
    if assertive_unknown_future_count > 0:
        key_issues.append(f"unknown/future claims phrased assertively: {assertive_unknown_future_count}")
    if confidence_assertiveness_mismatch_count > 0:
        key_issues.append(f"confidence/assertiveness mismatch: {confidence_assertiveness_mismatch_count}")
    if coverage_ratio < 0.4 and claim_count > 0:
        key_issues.append(f"suspiciously low direct evidence coverage ({coverage_ratio:.2f})")
    if unique_model_count == 1 and same_model_repeat_count_max > 0:
        key_issues.append("same-model repetition detected; treated as weak stability, not independent confirmation")
    if has_cross_model_support and unique_evidence_signature_count <= 1:
        key_issues.append("cross-model agreement exists but evidence signature diversity is low")
    if has_cross_model_conflict:
        key_issues.append("cross-model contradiction detected between supported and negative observations")
    elif has_soft_cross_model_conflict:
        key_issues.append("cross-model tension detected: supported vs unknown/future with explicit limitations")
    if metadata_incomplete_count > 0:
        key_issues.append(f"metadata completeness issues in {metadata_incomplete_count} claim(s)")
    if internal_conflict:
        key_issues.append("internal conflict: both supported and negative claims are present")
    if has_cross_conflict:
        key_issues.append(f"cross-proposition conflict with: {', '.join(cross_conflict_refs)}")
    if not key_issues:
        key_issues.append("no critical deterministic precheck issue detected")

    impact_weight = IMPACT_WEIGHT.get(proposition.priority_hint.casefold(), 0.5)
    support_penalty = 1.0 - support_strength
    conflict_bonus = 0.20 if (internal_conflict or has_cross_conflict) else 0.0
    coverage_penalty = 0.2 if coverage_ratio < 0.4 else (0.1 if coverage_ratio < 0.7 else 0.0)
    unknown_bonus = 0.1 if unknown_ratio > 0.5 else 0.0
    same_model_only_penalty = 0.08 if (unique_model_count == 1 and same_model_repeat_count_max > 0) else 0.0
    cross_model_conflict_boost = 0.30 if has_cross_model_conflict else (0.12 if has_soft_cross_model_conflict else 0.0)
    decision_risk = clamp01(
        (0.40 * impact_weight)
        + (0.40 * support_penalty)
        + conflict_bonus
        + coverage_penalty
        + unknown_bonus
        + same_model_only_penalty
        + cross_model_conflict_boost
    )
    review_priority_score = clamp01(
        (0.55 * decision_risk)
        + (0.25 * support_penalty)
        + (0.15 * impact_weight)
        + (0.20 if has_cross_model_conflict else 0.0)
        + (0.07 if (unique_model_count == 1 and same_model_repeat_count_max > 0) else 0.0)
        + (0.10 if len(key_issues) >= 3 else 0.0)
    )

    if has_cross_model_conflict or has_soft_cross_model_conflict or internal_conflict or has_cross_conflict:
        validity_assessment = "conflicting"
    elif support_strength >= 0.78 and coverage_ratio >= 0.7 and has_cross_model_support:
        validity_assessment = "strong"
    elif support_strength >= 0.5:
        validity_assessment = "moderate"
    else:
        validity_assessment = "weak"
    assert validity_assessment in VALIDITY_ASSESSMENT_VOCAB

    if has_cross_model_conflict or has_soft_cross_model_conflict:
        consensus_class = "cross_model_conflicting"
    elif has_cross_model_support and contradicting_observation_count == 0:
        consensus_class = "cross_model_supported"
    elif unique_model_count == 1 and same_model_repeat_count_max > 0:
        consensus_class = "same_model_repeated"
    else:
        consensus_class = "weak_sparse"
    assert consensus_class in CONSENSUS_CLASS_VOCAB

    if has_cross_model_conflict or has_soft_cross_model_conflict:
        suggested_status = "human_review_required"
    elif support_strength < 0.2 and coverage_ratio < 0.3 and impact_weight >= 0.65:
        suggested_status = "reject_candidate"
    elif missing_payload_count > 0 or (claim_count > 0 and missing_evidence_files_count == claim_count):
        suggested_status = "hold"
    elif (
        impact_weight >= 0.65
        or (consensus_class == "same_model_repeated" and support_strength < 0.65)
        or decision_risk >= 0.45
    ):
        suggested_status = "human_review_required"
    else:
        suggested_status = "auto_accept_candidate"
    assert suggested_status in SUGGESTED_REVIEW_STATUS_VOCAB

    rewrite_suggestion = (
        "Use tentative wording (e.g., 'may indicate') until direct evidence is added."
        if assertive_unknown_future_count > 0
        else None
    )

    deduped_next_data_needed: list[str] = []
    for item in next_data_needed:
        token = normalize_optional_text(item)
        if token is None or token in deduped_next_data_needed:
            continue
        deduped_next_data_needed.append(token)
    if missing_evidence_files_count > 0:
        deduped_next_data_needed.append("Attach direct evidence files per claim.")
    if supported_without_metrics_count > 0:
        deduped_next_data_needed.append("Add metrics_used entries for supported claims.")
    if missing_limitations_count > 0:
        deduped_next_data_needed.append("Document limitation_or_counterpoint for each affected claim.")
    if unique_model_count == 1:
        deduped_next_data_needed.append("Seek at least one independent observation from a different model.")

    evidence_coverage = {
        "has_direct_evidence": direct_evidence_claim_count > 0,
        "evidence_file_count": total_evidence_files,
        "evidence_ref_count": total_evidence_refs,
        "metric_count": total_metrics,
        "limitations_present": limitations_present_count > 0,
        "claim_count": claim_count,
        "coverage_ratio": round(coverage_ratio, 4),
    }
    observation_metrics = {
        "unique_model_count": unique_model_count,
        "unique_source_file_count": unique_source_file_count,
        "unique_observation_count": unique_observation_count,
        "unique_evidence_signature_count": unique_evidence_signature_count,
        "same_model_repeat_count_max": same_model_repeat_count_max,
        "supporting_observation_count": supporting_observation_count,
        "contradicting_observation_count": contradicting_observation_count,
        "has_cross_model_support": bool(has_cross_model_support),
        "has_cross_model_conflict": bool(has_cross_model_conflict),
    }
    consistency_check = {
        "internally_consistent": not internal_conflict,
        "cross_proposition_conflict": has_cross_conflict,
        "conflict_refs": cross_conflict_refs,
        "cross_model_conflict": has_cross_model_conflict,
    }
    precheck_flags = {
        "missing_evidence_files_count": missing_evidence_files_count,
        "missing_evidence_refs_count": missing_evidence_refs_count,
        "supported_without_metrics_count": supported_without_metrics_count,
        "missing_limitations_count": missing_limitations_count,
        "assertive_unknown_future_count": assertive_unknown_future_count,
        "confidence_assertiveness_mismatch_count": confidence_assertiveness_mismatch_count,
        "suspicious_low_support_coverage": int(coverage_ratio < 0.4 and claim_count > 0),
        "metadata_incomplete_count": metadata_incomplete_count,
        "missing_payload_count": missing_payload_count,
        "unique_model_count": unique_model_count,
        "same_model_repeat_count_max": same_model_repeat_count_max,
        "contradicting_observation_count": contradicting_observation_count,
    }
    source_provenance = {
        "triage_run_id": triage_run_id,
        "source": "proposition_triage_v1",
        "source_files": list(source_files),
        "claim_refs": [
            {"source_ai": ref.source_ai, "source_file": ref.source_file, "claim_id": ref.claim_id}
            for ref in proposition.claim_refs
        ],
    }
    action_suggestion = {
        "recommended_action": suggested_status,
        "reason": key_issues[0],
        "next_data_needed": deduped_next_data_needed,
    }

    return {
        "proposition_id": proposition.proposition_id,
        "proposition_question": proposition.question,
        "proposition_target": proposition.target,
        "review_status_suggested": suggested_status,
        "validity_assessment": validity_assessment,
        "support_strength": round(support_strength, 4),
        "decision_risk": round(decision_risk, 4),
        "review_priority_score": round(review_priority_score, 4),
        "consensus_class": consensus_class,
        "key_issues": key_issues,
        "evidence_coverage": evidence_coverage,
        "observation_metrics": observation_metrics,
        "consistency_check": consistency_check,
        "action_suggestion": action_suggestion,
        "rewrite_suggestion": rewrite_suggestion,
        "next_data_needed": deduped_next_data_needed,
        "precheck_flags": precheck_flags,
        "source_provenance": source_provenance,
        "created_at": created_at,
        "triage_run_id": triage_run_id,
        "scoring_components": {
            "base_evidence_score": round(base_evidence_score, 4),
            "cross_model_bonus": round(cross_model_bonus, 4),
            "evidence_diversity_bonus": round(evidence_diversity_bonus, 4),
            "same_model_stability_bonus": round(same_model_bonus, 4),
            "contradiction_penalty": round(contradiction_penalty, 4),
        },
    }


def _insert_triage_records(conn: sqlite3.Connection, *, records: Sequence[dict[str, Any]], created_at: str) -> int:
    written = 0
    for record in records:
        cur = conn.execute(
            """
            INSERT INTO proposition_triage_records (
              proposition_id,
              triage_run_id,
              review_status_suggested,
              validity_assessment,
              support_strength,
              decision_risk,
              review_priority_score,
              consensus_class,
              observation_metrics_json,
              key_issues_json,
              evidence_coverage_json,
              consistency_check_json,
              action_suggestion_json,
              rewrite_suggestion,
              next_data_needed_json,
              precheck_flags_json,
              source_provenance_json,
              created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposition_id, triage_run_id) DO UPDATE SET
              review_status_suggested = excluded.review_status_suggested,
              validity_assessment = excluded.validity_assessment,
              support_strength = excluded.support_strength,
              decision_risk = excluded.decision_risk,
              review_priority_score = excluded.review_priority_score,
              consensus_class = excluded.consensus_class,
              observation_metrics_json = excluded.observation_metrics_json,
              key_issues_json = excluded.key_issues_json,
              evidence_coverage_json = excluded.evidence_coverage_json,
              consistency_check_json = excluded.consistency_check_json,
              action_suggestion_json = excluded.action_suggestion_json,
              rewrite_suggestion = excluded.rewrite_suggestion,
              next_data_needed_json = excluded.next_data_needed_json,
              precheck_flags_json = excluded.precheck_flags_json,
              source_provenance_json = excluded.source_provenance_json,
              created_at = excluded.created_at
            """,
            (
                record["proposition_id"],
                record["triage_run_id"],
                record["review_status_suggested"],
                record["validity_assessment"],
                float(record["support_strength"]),
                float(record["decision_risk"]),
                float(record["review_priority_score"]),
                record["consensus_class"],
                json.dumps(record["observation_metrics"], ensure_ascii=False),
                json.dumps(record["key_issues"], ensure_ascii=False),
                json.dumps(record["evidence_coverage"], ensure_ascii=False),
                json.dumps(record["consistency_check"], ensure_ascii=False),
                json.dumps(record["action_suggestion"], ensure_ascii=False),
                record["rewrite_suggestion"],
                json.dumps(record["next_data_needed"], ensure_ascii=False),
                json.dumps(record["precheck_flags"], ensure_ascii=False),
                json.dumps(record["source_provenance"], ensure_ascii=False),
                created_at,
            ),
        )
        triage_record_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO proposition_review_state (
              proposition_id, current_status, triage_record_id, updated_at, note
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(proposition_id) DO UPDATE SET
              triage_record_id = excluded.triage_record_id
            """,
            (
                record["proposition_id"],
                "pending",
                triage_record_id,
                created_at,
                "initialized by triage",
            ),
        )
        written += 1
    return written


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def triage_propositions(
    *,
    db_path: Path | None = None,
    enriched_dir: Path = ENRICHED_DIR,
    review_dir: Path = REVIEW_DIR,
    triage_run_id: str | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> TriagePropositionsReport:
    created_at = now_iso()
    run_id = normalize_optional_text(triage_run_id) or now_compact_utc()
    target_output = output_path or default_triage_path(review_dir=review_dir, triage_run_id=run_id)

    proposition_claim_refs, source_files = load_integrated_claim_refs(enriched_dir)
    with connect_for_update(db_path) as conn:
        ensure_schema(conn)
        propositions = fetch_proposition_snapshots(conn, proposition_claim_refs=proposition_claim_refs)
        proposition_by_id = {item.proposition_id: item for item in propositions}

        claim_to_props: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for proposition in propositions:
            for ref in proposition.claim_refs:
                claim_to_props[(ref.source_ai, ref.source_file, ref.claim_id)].add(proposition.proposition_id)

        records: list[dict[str, Any]] = []
        for proposition in propositions:
            cross_conflict_refs: set[str] = set()
            this_family = resolved_family(proposition.resolved_type)
            for ref in proposition.claim_refs:
                neighbors = claim_to_props.get((ref.source_ai, ref.source_file, ref.claim_id), set())
                for neighbor_id in neighbors:
                    if neighbor_id == proposition.proposition_id:
                        continue
                    neighbor = proposition_by_id.get(neighbor_id)
                    if neighbor is None:
                        continue
                    neighbor_family = resolved_family(neighbor.resolved_type)
                    if this_family == "uncertain" or neighbor_family == "uncertain":
                        continue
                    if this_family != neighbor_family:
                        cross_conflict_refs.add(neighbor_id)

            records.append(
                _build_triage_record(
                    proposition,
                    triage_run_id=run_id,
                    created_at=created_at,
                    cross_conflict_refs=sorted(cross_conflict_refs),
                    source_files=source_files,
                )
            )

        records.sort(key=lambda item: (-float(item["review_priority_score"]), item["proposition_id"]))
        write_jsonl(target_output, rows=records)

        if dry_run:
            written = 0
        else:
            written = _insert_triage_records(conn, records=records, created_at=created_at)
            conn.commit()

    queue_candidates = sum(1 for row in records if row["review_status_suggested"] != "auto_accept_candidate")
    return TriagePropositionsReport(
        triage_run_id=run_id,
        scanned_propositions=len(records),
        triage_records_written=written,
        queue_candidates=queue_candidates,
        output_path=str(target_output),
        dry_run=bool(dry_run),
    )
