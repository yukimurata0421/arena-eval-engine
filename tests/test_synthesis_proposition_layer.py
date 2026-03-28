from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena.synthesis.proposition_layer import _PROPOSITION_CONFIG_PATH, _PROPOSITION_DEFINITIONS, build_proposition_layer


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_proposition_config_is_externalized() -> None:
    assert _PROPOSITION_CONFIG_PATH.exists()
    assert len(_PROPOSITION_DEFINITIONS) >= 1


def test_build_proposition_layer_writes_enriched_outputs_and_db(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    db_path = tmp_path / "db" / "synthesis.sqlite3"

    claude_records = [
        {
            "id": "CLAIM-001",
            "claim_type": "supported",
            "claim": "The change point in auc_n_used is around 2026-01-10.",
            "exploration_axes": ["gain", "baseline", "coverage_auc"],
            "evidence_files": ["change_point_report.txt"],
        },
        {
            "id": "CLAIM-010",
            "claim_type": "supported",
            "claim": "beta_minutes is positive and significant.",
            "exploration_axes": ["stats", "dropout"],
            "evidence_files": ["phase_evaluator_report.txt"],
        },
    ]
    gpt_records = [
        {
            "claim_type": "supported",
            "claim": "Adapter change phase improves daily AUC by +17.4% versus Airspy baseline.",
            "exploration_axes": ["coverage_auc", "baseline", "hardware"],
            "evidence_files": ["phase_evaluator_results.csv"],
        },
        {
            "claim_type": "future",
            "claim": "This statement should not map to any proposition.",
            "exploration_axes": ["misc"],
            "evidence_files": ["unknown.txt"],
        },
    ]

    source_claude = raw_dir / "claude" / "20260324.json"
    source_gpt = raw_dir / "gpt" / "20260324.json"
    _write_json(source_claude, claude_records)
    _write_json(source_gpt, gpt_records)

    report = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)

    assert report.failed_files == 0
    assert report.claim_count == 4
    assert report.mapped_claim_count == 3
    assert report.orphan_count == 1
    assert report.db_proposition_count == 11

    enriched_claude = json.loads((enriched_dir / "claude" / "20260324.json").read_text(encoding="utf-8"))
    assert enriched_claude[0]["proposition_ids"] == ["PROP-001", "PROP-002"]
    assert enriched_claude[1]["proposition_ids"] == ["PROP-011"]

    raw_claude_after = json.loads(source_claude.read_text(encoding="utf-8"))
    assert "proposition_ids" not in raw_claude_after[0]

    integrated_claude = json.loads((enriched_dir / "claude.json").read_text(encoding="utf-8"))
    assert integrated_claude["meta"]["source_ai"] == "claude"
    assert integrated_claude["meta"]["claim_count"] == 2
    assert integrated_claude["meta"]["proposition_count"] == 11

    with sqlite3.connect(db_path) as conn:
        proposition_count = conn.execute("SELECT COUNT(*) FROM propositions").fetchone()[0]
        relation_count = conn.execute("SELECT COUNT(*) FROM claim_propositions").fetchone()[0]
        claude_map = conn.execute(
            "SELECT proposition_id FROM claim_propositions WHERE source_ai = 'claude' AND claim_id = 'CLAIM-001' ORDER BY proposition_id"
        ).fetchall()

    assert proposition_count == 11
    assert relation_count == report.relation_count
    assert [row[0] for row in claude_map] == ["PROP-001", "PROP-002"]
    assert any("source_ai=gpt" in message for message in report.orphan_messages)


def test_build_proposition_layer_handles_top_level_dict_without_claims_list(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    db_path = tmp_path / "db" / "synthesis.sqlite3"

    _write_json(raw_dir / "claude" / "20260324.json", {"unexpected": "shape"})

    report = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)

    assert report.failed_files == 1
    assert report.loaded_files == 0
    assert report.claim_count == 0
    assert report.relation_count == 0
    assert report.db_proposition_count == 11
    assert report.db_relation_count == 0
    assert any("parse_failed:" in message for message in report.validation_messages)
    assert not (enriched_dir / "claude" / "20260324.json").exists()
    assert not (enriched_dir / "claude.json").exists()


def test_build_proposition_layer_repairs_code_fence_wrapped_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    db_path = tmp_path / "db" / "synthesis.sqlite3"

    wrapped = """prefix
```json
[
  {
    "claim": "Adapter change phase improves daily AUC by +17.4% versus Airspy baseline.",
    "claim_type": "supported",
    "exploration_axes": ["coverage_auc", "baseline", "hardware"],
    "evidence_files": ["phase_evaluator_results.csv"]
  }
]
```
suffix
"""
    _write_text(raw_dir / "gpt" / "20260324.json", wrapped)

    report = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)

    assert report.failed_files == 0
    assert report.loaded_files == 1
    assert report.repaired_files == 1
    assert report.claim_count == 1
    assert report.mapped_claim_count == 1

    enriched = json.loads((enriched_dir / "gpt" / "20260324.json").read_text(encoding="utf-8"))
    assert enriched[0]["proposition_ids"] == ["PROP-005"]


def test_build_proposition_layer_logs_invalid_exploration_axes_type(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    db_path = tmp_path / "db" / "synthesis.sqlite3"

    _write_json(
        raw_dir / "gemini" / "20260324.json",
        [
            {
                "id": "CLAIM-999",
                "claim": "dummy",
                "claim_type": "unknown",
                "exploration_axes": "gain",
                "evidence_files": ["x.csv"],
            }
        ],
    )

    report = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    assert any("exploration_axes_invalid_type:" in message for message in report.validation_messages)


def test_build_proposition_layer_keeps_existing_relations_when_ai_input_fully_broken(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    db_path = tmp_path / "db" / "synthesis.sqlite3"
    source_file = raw_dir / "claude" / "20260324.json"

    _write_json(
        source_file,
        [
            {
                "id": "CLAIM-001",
                "claim_type": "supported",
                "claim": "The change point in auc_n_used is around 2026-01-10.",
                "exploration_axes": ["gain", "baseline", "coverage_auc"],
                "evidence_files": ["change_point_report.txt"],
            }
        ],
    )
    first = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    assert first.failed_files == 0
    assert first.mapped_claim_count == 1

    with sqlite3.connect(db_path) as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM claim_propositions WHERE source_ai = 'claude'"
        ).fetchone()[0]
    assert before > 0

    _write_text(source_file, '{"broken": ')  # unrecoverable parse error
    second = build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    assert second.failed_files == 1
    assert second.loaded_files == 0

    with sqlite3.connect(db_path) as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM claim_propositions WHERE source_ai = 'claude'"
        ).fetchone()[0]
    assert after == before
