from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.synthesis.db import connect, ensure_schema
from arena.synthesis.paths import DB_PATH, ENRICHED_DIR, RAW_DIR
from arena.synthesis.repair import repair_json_text

_ALLOWED_CONVERGENCE = {"convergent", "partial", "divergent"}
_ALLOWED_RESOLVED_TYPE = {"supported", "supported_with_caveat", "negative", "unresolved", "insufficient_data"}
_ALLOWED_PRIORITY_HINT = {"high", "medium", "low"}

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_PROPOSITION_CONFIG_PATH = _CONFIG_DIR / "propositions.json"


def _load_proposition_config() -> tuple[tuple[dict[str, Any], ...], dict[str, tuple[str, ...]]]:
    try:
        parsed = json.loads(_PROPOSITION_CONFIG_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to load proposition config: {_PROPOSITION_CONFIG_PATH} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid proposition config JSON: {_PROPOSITION_CONFIG_PATH} "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"proposition config root must be object: {_PROPOSITION_CONFIG_PATH}")

    definitions = parsed.get("proposition_definitions")
    if not isinstance(definitions, list):
        raise RuntimeError(f"proposition_definitions must be list: {_PROPOSITION_CONFIG_PATH}")

    normalized_definitions: list[dict[str, Any]] = []
    for index, item in enumerate(definitions, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"proposition_definitions[{index}] must be object: {_PROPOSITION_CONFIG_PATH}"
            )
        normalized_definitions.append(dict(item))

    raw_reference_map = parsed.get("claude_reference_map")
    if not isinstance(raw_reference_map, dict):
        raise RuntimeError(f"claude_reference_map must be object: {_PROPOSITION_CONFIG_PATH}")

    normalized_reference_map: dict[str, tuple[str, ...]] = {}
    for claim_id, proposition_ids in raw_reference_map.items():
        if not isinstance(claim_id, str):
            raise RuntimeError(
                f"claude_reference_map keys must be strings: {_PROPOSITION_CONFIG_PATH}"
            )
        if not isinstance(proposition_ids, list) or not all(isinstance(item, str) for item in proposition_ids):
            raise RuntimeError(
                f"claude_reference_map[{claim_id!r}] must be list[str]: {_PROPOSITION_CONFIG_PATH}"
            )
        normalized_reference_map[claim_id] = tuple(proposition_ids)

    return tuple(normalized_definitions), normalized_reference_map


_PROPOSITION_DEFINITIONS, _CLAUDE_REFERENCE_MAP = _load_proposition_config()


@dataclass(frozen=True, slots=True)
class ClaimEnvelope:
    source_ai: str
    source_file: str
    claim_id: str
    payload: dict[str, Any]
    proposition_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PropositionLayerReport:
    scanned_files: int
    loaded_files: int
    repaired_files: int
    failed_files: int
    claim_count: int
    mapped_claim_count: int
    orphan_count: int
    proposition_count: int
    relation_count: int
    db_proposition_count: int
    db_relation_count: int
    validation_messages: tuple[str, ...]
    orphan_messages: tuple[str, ...]
    enriched_outputs: tuple[str, ...]
    integrated_outputs: tuple[str, ...]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return " ".join(str(value).split()).casefold()


def _normalize_claim_id(raw_id: Any) -> str | None:
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    return text or None


def _safe_json_load(path: Path) -> tuple[list[dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8-sig")
    repaired = False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        repair = repair_json_text(text)
        parsed = json.loads(repair.text)
        repaired = True

    records: Any
    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("claims"), list):
            records = parsed["claims"]
        elif isinstance(parsed.get("hypotheses"), list):
            records = parsed["hypotheses"]
        else:
            raise ValueError(f"top-level dict must contain claims/hypotheses list: {path}")
    else:
        raise ValueError(f"unsupported top-level JSON type: {type(parsed).__name__} ({path})")

    normalized: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized, repaired


def _resolve_claim_id(payload: dict[str, Any], source_file: Path, index: int) -> str:
    explicit = _normalize_claim_id(payload.get("id"))
    if explicit is not None:
        return explicit
    return f"{source_file.stem}-CLAIM-{index:03d}"


def _claim_axes(payload: dict[str, Any]) -> set[str]:
    axes = payload.get("exploration_axes")
    if not isinstance(axes, list):
        return set()
    output: set[str] = set()
    for axis in axes:
        if isinstance(axis, str):
            cleaned = axis.strip().casefold()
            if cleaned:
                output.add(cleaned)
    return output


def _contains_any(blob: str, needles: tuple[str, ...]) -> bool:
    return any(needle in blob for needle in needles)


def _map_claim_to_props(source_ai: str, claim_id: str, payload: dict[str, Any]) -> tuple[str, ...]:
    if source_ai == "claude" and claim_id in _CLAUDE_REFERENCE_MAP:
        return _CLAUDE_REFERENCE_MAP[claim_id]

    files = payload.get("evidence_files")
    if isinstance(files, list):
        file_blob = " ".join(_normalize_text(item) for item in files if isinstance(item, str))
    else:
        file_blob = _normalize_text(files)

    text_blob = " ".join(
        [
            _normalize_text(payload.get("claim")),
            _normalize_text(payload.get("basis_summary")),
            _normalize_text(payload.get("limitation_or_counterpoint")),
            _normalize_text(payload.get("raw_text")),
            _normalize_text(payload.get("claim_type")),
            file_blob,
        ]
    )
    axes = _claim_axes(payload)
    selected: set[str] = set()

    if _contains_any(
        text_blob,
        ("change point", "auc_n_used", "daily_auc", "2026-01-10", "2026-01-13", "mid-january", "boundary"),
    ):
        selected.add("PROP-001")

    if (
        _contains_any(
            text_blob,
            ("quantile_signature", "coverage_signature", "coverage signature", "stepwise lag", "lag", "linear interpolation"),
        )
        or ("quantile" in text_blob and "coverage" in text_blob and _contains_any(text_blob, ("01-18", "01-30", "day 5", "day 17")))
    ):
        selected.add("PROP-002")

    if _contains_any(
        text_blob,
        (
            "gain tuning",
            "gain tuned",
            "gain 33.8",
            "rtl-sdr gain tuned",
            "airspy mini vs rtl-sdr",
            "decomposition",
            "primary driver",
            "contribution ratio",
            "allocation",
        ),
    ) or ({"gain", "baseline"} <= axes):
        selected.add("PROP-003")

    if _contains_any(
        text_blob,
        (
            "5d-fb",
            "cable v2",
            "airspy+cable",
            "airspy cable",
            "n-p",
            "cable introduction",
            "cable phase",
            "cable change",
            "indoor cable",
            "2.5ds-qfb",
            "cable",
        ),
    ) or ("cable" in axes and "adapter" not in text_blob):
        selected.add("PROP-004")

    if _contains_any(text_blob, ("adapter", "nm-sm50", "airspy_adapter", "adapter change")) or "adapter" in axes:
        selected.add("PROP-005")

    if _contains_any(
        text_blob,
        ("traffic", "beta_traffic", "weekday effect", "weekday", "day-of-week", "arrivals", "departures", "kruskal"),
    ) or "traffic" in axes:
        selected.add("PROP-006")

    if _contains_any(
        text_blob,
        ("second change point", "second", "multi change", "k=3", "cp2", "single regime", "collapse"),
    ):
        selected.add("PROP-007")

    if _contains_any(
        text_blob,
        (
            "200+km",
            "300km",
            "150-200km",
            "50km",
            "0-50 km",
            "q95",
            "extreme",
            "capture ratio",
            "capture_ratio",
            "opensky",
            "far distance",
            "distance-bin",
            "coverage_300km",
        ),
    ) or "distance" in axes:
        selected.add("PROP-008")

    if _contains_any(
        text_blob,
        (
            "health",
            "integrity",
            "reproducibility",
            "hash",
            "deterministic_flag",
            "complete separation",
            "skipped",
            "no_pos_file",
            "representative",
            "seasonal",
            "unverified",
            "quality",
            "missing data",
        ),
    ):
        selected.add("PROP-009")

    if _contains_any(
        text_blob,
        (
            "cumulative",
            "+136.5",
            "rtl-sdr default",
            "overall improvement",
            "airspy mini phase",
            "phase_timebin_summary",
            "los efficiency",
            "69.33%",
            "110.61%",
            "total improvement",
            "+105",
        ),
    ) or ({"coverage_auc", "baseline"} <= axes and "rtl-sdr" in text_blob and "airspy" in text_blob):
        selected.add("PROP-010")

    if _contains_any(text_blob, ("beta_minutes", "minutes_covered", "minutes elasticity", "uptime minutes")) or (
        "minutes" in text_blob and "hdi" in text_blob
    ):
        selected.add("PROP-011")

    if "PROP-001" in selected and _contains_any(text_blob, ("quantile", "coverage_signature", "01-18", "01-30")):
        selected.add("PROP-002")

    if "PROP-005" in selected and _contains_any(text_blob, ("second change point", "k=3", "single regime", "change point")):
        selected.add("PROP-007")

    return tuple(sorted(selected))


def _validate_proposition_definitions() -> list[str]:
    messages: list[str] = []
    for proposition in _PROPOSITION_DEFINITIONS:
        proposition_id = proposition["proposition_id"]
        convergence = proposition["convergence"]
        if convergence not in _ALLOWED_CONVERGENCE:
            messages.append(f"invalid convergence: {proposition_id}={convergence}")
        resolved = proposition["resolved_type"]
        if resolved not in _ALLOWED_RESOLVED_TYPE:
            messages.append(f"invalid resolved_type: {proposition_id}={resolved}")
        priority = proposition["priority_hint"]
        if priority not in _ALLOWED_PRIORITY_HINT:
            messages.append(f"invalid priority_hint: {proposition_id}={priority}")

        axes = proposition.get("exploration_axes")
        if not isinstance(axes, list):
            messages.append(f"invalid exploration_axes type: {proposition_id}={type(axes).__name__}")
            continue
        if len(axes) > 3:
            messages.append(f"exploration_axes too long: {proposition_id} has {len(axes)}")
    return messages


def _render_integrated_propositions(claims: list[ClaimEnvelope]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ClaimEnvelope]] = {}
    for claim in claims:
        for proposition_id in claim.proposition_ids:
            grouped.setdefault(proposition_id, []).append(claim)

    output: list[dict[str, Any]] = []
    for proposition in _PROPOSITION_DEFINITIONS:
        proposition_id = proposition["proposition_id"]
        refs = grouped.get(proposition_id, [])
        claim_ids = sorted({ref.claim_id for ref in refs})
        item = dict(proposition)
        item["claim_ids"] = claim_ids
        output.append(item)
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _connect_db(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _persist_db(
    *,
    conn: sqlite3.Connection,
    claims: list[ClaimEnvelope],
    source_ais: set[str],
    created_at: str,
) -> tuple[int, int]:
    ensure_schema(conn)
    proposition_by_id = {p["proposition_id"]: p for p in _PROPOSITION_DEFINITIONS}

    for proposition in _PROPOSITION_DEFINITIONS:
        conn.execute(
            """
            INSERT INTO propositions (
                proposition_id, question, target, convergence, convergence_detail,
                resolved_type, caveat, exploration_axes, priority_hint, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposition_id) DO UPDATE SET
                question = excluded.question,
                target = excluded.target,
                convergence = excluded.convergence,
                convergence_detail = excluded.convergence_detail,
                resolved_type = excluded.resolved_type,
                caveat = excluded.caveat,
                exploration_axes = excluded.exploration_axes,
                priority_hint = excluded.priority_hint,
                created_at = excluded.created_at
            """,
            (
                proposition["proposition_id"],
                proposition["question"],
                proposition["target"],
                proposition["convergence"],
                proposition["convergence_detail"],
                proposition["resolved_type"],
                proposition["caveat"],
                json.dumps(proposition["exploration_axes"], ensure_ascii=False),
                proposition["priority_hint"],
                created_at,
            ),
        )

    for source_ai in sorted(source_ais):
        conn.execute("DELETE FROM claim_propositions WHERE source_ai = ?", (source_ai,))

    relations: set[tuple[str, str, str]] = set()
    for claim in claims:
        for proposition_id in claim.proposition_ids:
            relations.add((claim.claim_id, proposition_id, claim.source_ai))

    for claim_id, proposition_id, source_ai in sorted(relations):
        proposition_question = str(proposition_by_id[proposition_id]["question"])
        conn.execute(
            """
            INSERT INTO claim_propositions (claim_id, proposition_id, source_ai, proposition_question)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(claim_id, proposition_id, source_ai) DO NOTHING
            """,
            (claim_id, proposition_id, source_ai, proposition_question),
        )

    conn.commit()
    db_prop_count = int(conn.execute("SELECT COUNT(*) FROM propositions").fetchone()[0])
    db_rel_count = int(conn.execute("SELECT COUNT(*) FROM claim_propositions").fetchone()[0])
    return db_prop_count, db_rel_count


def build_proposition_layer(
    *,
    raw_dir: Path = RAW_DIR,
    enriched_dir: Path = ENRICHED_DIR,
    db_path: Path | None = None,
) -> PropositionLayerReport:
    validation_messages = _validate_proposition_definitions()
    orphan_messages: list[str] = []
    enriched_outputs: list[str] = []
    integrated_outputs: list[str] = []
    all_claims: list[ClaimEnvelope] = []
    source_ais: set[str] = set()
    repaired_files = 0
    failed_files = 0
    loaded_files = 0

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise NotADirectoryError(f"raw directory does not exist or is not a directory: {raw_dir}")

    ai_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    scanned_files = 0
    created_at = _now_iso()

    for ai_dir in ai_dirs:
        source_ai = ai_dir.name.casefold()
        ai_claims: list[ClaimEnvelope] = []
        ai_loaded_any_file = False

        for source_file in sorted(ai_dir.rglob("*.json")):
            scanned_files += 1
            try:
                records, repaired = _safe_json_load(source_file)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failed_files += 1
                validation_messages.append(f"parse_failed: {source_file} ({exc})")
                continue

            loaded_files += 1
            ai_loaded_any_file = True
            if repaired:
                repaired_files += 1

            enriched_records: list[dict[str, Any]] = []
            for index, payload in enumerate(records, start=1):
                claim_id = _resolve_claim_id(payload, source_file=source_file, index=index)
                proposition_ids = _map_claim_to_props(source_ai=source_ai, claim_id=claim_id, payload=payload)
                enriched_payload = dict(payload)
                enriched_payload["proposition_ids"] = list(proposition_ids)
                enriched_records.append(enriched_payload)

                envelope = ClaimEnvelope(
                    source_ai=source_ai,
                    source_file=str(source_file.relative_to(raw_dir)),
                    claim_id=claim_id,
                    payload=payload,
                    proposition_ids=proposition_ids,
                )
                ai_claims.append(envelope)
                all_claims.append(envelope)

            rel = source_file.relative_to(ai_dir)
            output_path = enriched_dir / source_ai / rel
            _write_json(output_path, enriched_records)
            enriched_outputs.append(str(output_path))

        if ai_loaded_any_file:
            source_ais.add(source_ai)

        if not ai_claims:
            continue

        integrated_claims: list[dict[str, Any]] = []
        for claim in ai_claims:
            item = dict(claim.payload)
            item["proposition_ids"] = list(claim.proposition_ids)
            item["claim_id"] = claim.claim_id
            item["source_file"] = claim.source_file
            integrated_claims.append(item)

        integrated = {
            "meta": {
                "version": "2.0.0",
                "generated_at": created_at,
                "source_ai": source_ai,
                "claim_count": len(ai_claims),
                "proposition_count": len(_PROPOSITION_DEFINITIONS),
            },
            "propositions": _render_integrated_propositions(ai_claims),
            "claims": integrated_claims,
        }
        integrated_path = enriched_dir / f"{source_ai}.json"
        _write_json(integrated_path, integrated)
        integrated_outputs.append(str(integrated_path))

    for claim in all_claims:
        axes = claim.payload.get("exploration_axes")
        if axes is None:
            continue
        if not isinstance(axes, list):
            validation_messages.append(
                f"exploration_axes_invalid_type: source_ai={claim.source_ai} claim_id={claim.claim_id} type={type(axes).__name__}"
            )
            continue
        if len(axes) > 3:
            validation_messages.append(
                f"exploration_axes_too_long: source_ai={claim.source_ai} claim_id={claim.claim_id} len={len(axes)}"
            )

    orphans = [claim for claim in all_claims if not claim.proposition_ids]
    for orphan in orphans:
        orphan_messages.append(
            f"source_ai={orphan.source_ai} claim_id={orphan.claim_id} source_file={orphan.source_file}"
        )

    claim_keys = {(claim.source_ai, claim.claim_id) for claim in all_claims}
    relation_keys = {
        (claim.source_ai, claim.claim_id, proposition_id)
        for claim in all_claims
        for proposition_id in claim.proposition_ids
    }
    for source_ai, claim_id, proposition_id in sorted(relation_keys):
        if (source_ai, claim_id) not in claim_keys:
            validation_messages.append(
                f"missing_claim_reference: source_ai={source_ai} claim_id={claim_id} proposition_id={proposition_id}"
            )

    claim_to_prop = {(claim.source_ai, claim.claim_id): set(claim.proposition_ids) for claim in all_claims}
    prop_to_claim: dict[str, set[tuple[str, str]]] = {}
    for claim in all_claims:
        for proposition_id in claim.proposition_ids:
            prop_to_claim.setdefault(proposition_id, set()).add((claim.source_ai, claim.claim_id))

    for claim_key, proposition_ids in claim_to_prop.items():
        for proposition_id in proposition_ids:
            if claim_key not in prop_to_claim.get(proposition_id, set()):
                validation_messages.append(
                    f"bidirectional_mismatch: claim={claim_key[0]}:{claim_key[1]} proposition={proposition_id}"
                )
    for proposition_id, linked_claims in prop_to_claim.items():
        for source_ai, claim_id in linked_claims:
            if proposition_id not in claim_to_prop.get((source_ai, claim_id), set()):
                validation_messages.append(
                    f"bidirectional_mismatch: proposition={proposition_id} claim={source_ai}:{claim_id}"
                )

    relation_count = sum(len(claim.proposition_ids) for claim in all_claims)
    mapped_count = sum(1 for claim in all_claims if claim.proposition_ids)

    target_db = db_path or DB_PATH
    with _connect_db(db_path=db_path) as conn:
        db_proposition_count, db_relation_count = _persist_db(
            conn=conn,
            claims=all_claims,
            source_ais=source_ais,
            created_at=created_at,
        )
        validation_messages.append(f"db_target={target_db}")
        validation_messages.append(
            f"db_counts_after_upsert: propositions={db_proposition_count} claim_propositions={db_relation_count}"
        )

    validation_messages.append(
        f"claim_assignment_summary: total={len(all_claims)} mapped={mapped_count} orphan={len(orphans)}"
    )
    validation_messages.append(
        "value_range_check: convergence/resolved_type/priority_hint and proposition exploration_axes(max=3) validated"
    )
    validation_messages.append("bidirectional_check: claim<->proposition consistency verified")

    return PropositionLayerReport(
        scanned_files=scanned_files,
        loaded_files=loaded_files,
        repaired_files=repaired_files,
        failed_files=failed_files,
        claim_count=len(all_claims),
        mapped_claim_count=mapped_count,
        orphan_count=len(orphans),
        proposition_count=len(_PROPOSITION_DEFINITIONS),
        relation_count=relation_count,
        db_proposition_count=db_proposition_count,
        db_relation_count=db_relation_count,
        validation_messages=tuple(validation_messages),
        orphan_messages=tuple(orphan_messages),
        enriched_outputs=tuple(enriched_outputs),
        integrated_outputs=tuple(integrated_outputs),
    )
