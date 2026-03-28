from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from arena.synthesis.ingest_store import now_iso


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_artifact_path(base_dir: Path, *, source_path: Path, raw_root: Path, source_hash: str) -> Path:
    rel = source_path.relative_to(raw_root)
    filename = f"{source_path.stem}__{source_hash[:12]}{source_path.suffix}"
    return base_dir / rel.parent / filename


def write_bytes_once(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return 0
    path.write_bytes(payload)
    return 1


def write_text_once(path: Path, payload: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return 0
    path.write_text(payload, encoding="utf-8", newline="\n")
    return 1


def append_repair_log(log_dir: Path, payload: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ingest_repair.jsonl"
    with log_path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False))
        fp.write("\n")
    return log_path


def _parse_action_count(action: str) -> tuple[str, int]:
    name, sep, count_text = action.rpartition(":")
    if sep and count_text.isdigit():
        return name, int(count_text)
    return action, 1


def build_layered_repair_actions(
    *,
    syntax_repairs_applied: list[str] | tuple[str, ...],
    shape_repairs_applied: list[str] | tuple[str, ...],
    semantic_repairs_applied: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    repair_actions: list[dict[str, Any]] = []
    for layer, actions in (
        ("syntax", syntax_repairs_applied),
        ("shape", shape_repairs_applied),
        ("semantic", semantic_repairs_applied),
    ):
        for action in actions:
            action_name, count = _parse_action_count(action)
            repair_actions.append(
                {
                    "action": action_name,
                    "count": count,
                    "repair_layer": layer,
                }
            )
    return repair_actions


def log_repair_event(
    repair_log_dir: Path,
    *,
    source_path: Path,
    source_hash: str,
    repaired_sha256: str | None,
    original_path: Path | None,
    repaired_path: Path | None,
    repair_requested: bool,
    repair_semantic_requested: bool,
    syntax_repairs_applied: list[str] | tuple[str, ...],
    shape_repairs_applied: list[str] | tuple[str, ...],
    semantic_repairs_applied: list[str] | tuple[str, ...],
    warnings: list[str] | tuple[str, ...],
    validation_result: str,
    db_ingest_result: str,
    inserted_records: int,
    failed_records: int,
    record_count_before: int | None,
    record_count_after: int | None,
    skipped_reason: str | None = None,
) -> Path:
    repair_actions = build_layered_repair_actions(
        syntax_repairs_applied=syntax_repairs_applied,
        shape_repairs_applied=shape_repairs_applied,
        semantic_repairs_applied=semantic_repairs_applied,
    )
    repair_layers = sorted({str(item["repair_layer"]) for item in repair_actions})
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "source_file": str(source_path),
        "source_sha256": source_hash,
        "repaired_sha256": repaired_sha256,
        "original_file": str(original_path) if original_path else None,
        "repaired_file": str(repaired_path) if repaired_path else None,
        "repair_requested": bool(repair_requested),
        "repair_semantic_requested": bool(repair_semantic_requested),
        "repair_layer": repair_layers,
        "repair_actions": repair_actions,
        "syntax_repairs_applied": list(syntax_repairs_applied),
        "shape_repairs_applied": list(shape_repairs_applied),
        "semantic_repairs_applied": list(semantic_repairs_applied),
        "warnings": list(warnings),
        "validation_result": validation_result,
        "db_ingest_result": db_ingest_result,
        "inserted_records": int(inserted_records),
        "failed_records": int(failed_records),
        "record_count_before": record_count_before,
        "record_count_after": record_count_after,
    }
    if skipped_reason:
        payload["skipped_reason"] = skipped_reason
    return append_repair_log(repair_log_dir, payload)
