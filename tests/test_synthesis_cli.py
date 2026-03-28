from __future__ import annotations

import pytest

from arena.synthesis import cli as synthesis_cli
from arena.synthesis.cluster_baselines import BaselineClusterReport
from arena.synthesis.enrich import EnrichReport
from arena.synthesis.ingest import IngestRawReport
from arena.synthesis.paths import RAW_DIR, RAW_ORIGINAL_DIR, RAW_REPAIRED_DIR, REPAIR_LOG_DIR, REVIEW_DIR
from arena.synthesis.proposition_layer import PropositionLayerReport


def test_ingest_alias_defaults_to_raw_dir() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["ingest"])
    assert args.path == str(RAW_DIR)
    assert args.repair is True
    assert args.repair_semantic is False
    assert args.raw_original_dir == str(RAW_ORIGINAL_DIR)
    assert args.raw_repaired_dir == str(RAW_REPAIRED_DIR)
    assert args.repair_log_dir == str(REPAIR_LOG_DIR)
    assert args.func is synthesis_cli.cmd_ingest_raw


def test_ingest_raw_defaults_to_raw_dir() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["ingest-raw"])
    assert args.path == str(RAW_DIR)
    assert args.repair is False
    assert args.repair_semantic is False
    assert args.raw_original_dir == str(RAW_ORIGINAL_DIR)
    assert args.raw_repaired_dir == str(RAW_REPAIRED_DIR)
    assert args.repair_log_dir == str(REPAIR_LOG_DIR)
    assert args.func is synthesis_cli.cmd_ingest_raw


def test_ingest_accepts_repair_flag() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["ingest", "--repair"])
    assert args.repair is True


def test_ingest_accepts_repair_semantic_flag() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["ingest", "--repair-semantic"])
    assert args.repair is True
    assert args.repair_semantic is True


def test_cmd_ingest_raw_prints_new_summary_fields(monkeypatch, capsys) -> None:
    def fake_ingest_raw_tree(
        raw_dir,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        assert repair is True
        assert repair_semantic is False
        return IngestRawReport(
            scanned_files=4,
            scanned_candidates=3,
            ingested_files=2,
            skipped_duplicates=1,
            skipped_layout=1,
            ingested_records=7,
            failed_records=0,
            warnings=("skip (layout): x",),
            original_saved=1,
            repaired_saved=1,
            repairs_applied=4,
            warnings_count=1,
        )

    monkeypatch.setattr(synthesis_cli, "ingest_raw_tree", fake_ingest_raw_tree)

    rc = synthesis_cli.main(["ingest"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "scanned_candidates=3" in captured.out
    assert "original_saved=1" in captured.out
    assert "repairs_applied=4" in captured.out
    assert "failed_records=0" in captured.out
    assert "status=SUCCESS" in captured.out


def test_cmd_ingest_raw_passes_repair_flag(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_ingest_raw_tree(
        raw_dir,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        seen["repair"] = str(repair)
        seen["repair_semantic"] = str(repair_semantic)
        seen["db_path"] = str(db_path)
        seen["raw_original_dir"] = str(raw_original_dir)
        seen["raw_repaired_dir"] = str(raw_repaired_dir)
        seen["repair_log_dir"] = str(repair_log_dir)
        return IngestRawReport(
            scanned_files=1,
            scanned_candidates=1,
            ingested_files=1,
            skipped_duplicates=0,
            skipped_layout=0,
            ingested_records=1,
            failed_records=0,
            warnings=(),
        )

    monkeypatch.setattr(synthesis_cli, "ingest_raw_tree", fake_ingest_raw_tree)

    rc = synthesis_cli.main(["ingest", "--repair"])
    _ = capsys.readouterr()

    assert rc == 0
    assert seen["repair"] == "True"
    assert seen["repair_semantic"] == "False"
    assert seen["db_path"] == "None"
    assert seen["raw_original_dir"] == str(RAW_ORIGINAL_DIR)
    assert seen["raw_repaired_dir"] == str(RAW_REPAIRED_DIR)
    assert seen["repair_log_dir"] == str(REPAIR_LOG_DIR)


def test_cmd_ingest_raw_passes_repair_semantic_flag(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_ingest_raw_tree(
        raw_dir,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        seen["repair"] = str(repair)
        seen["repair_semantic"] = str(repair_semantic)
        seen["db_path"] = str(db_path)
        return IngestRawReport(
            scanned_files=1,
            scanned_candidates=1,
            ingested_files=1,
            skipped_duplicates=0,
            skipped_layout=0,
            ingested_records=1,
            failed_records=0,
            warnings=(),
        )

    monkeypatch.setattr(synthesis_cli, "ingest_raw_tree", fake_ingest_raw_tree)

    rc = synthesis_cli.main(["ingest", "--repair", "--repair-semantic"])
    _ = capsys.readouterr()

    assert rc == 0
    assert seen["repair"] == "True"
    assert seen["repair_semantic"] == "True"
    assert seen["db_path"] == "None"


def test_cmd_ingest_raw_passes_db_override(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_ingest_raw_tree(
        raw_dir,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        seen["db_path"] = str(db_path)
        return IngestRawReport(
            scanned_files=1,
            scanned_candidates=1,
            ingested_files=1,
            skipped_duplicates=0,
            skipped_layout=0,
            ingested_records=1,
            failed_records=0,
            warnings=(),
        )

    monkeypatch.setattr(synthesis_cli, "ingest_raw_tree", fake_ingest_raw_tree)

    rc = synthesis_cli.main(["ingest", "--db", "tmp/synthesis.sqlite3"])
    _ = capsys.readouterr()

    assert rc == 0
    assert seen["db_path"].replace("\\", "/").endswith("tmp/synthesis.sqlite3")


def test_ingest_accepts_no_repair_flag() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["ingest", "--no-repair"])
    assert args.repair is False


def test_ingest_repair_semantic_requires_repair(capsys) -> None:
    rc = synthesis_cli.main(["ingest-raw", "--repair-semantic"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--repair-semantic requires --repair" in captured.err


def test_ingest_file_repair_semantic_requires_repair(capsys) -> None:
    rc = synthesis_cli.main(["ingest-file", "--path", "dummy.json", "--repair-semantic"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--repair-semantic requires --repair" in captured.err


def test_ingest_dir_repair_semantic_requires_repair(capsys) -> None:
    rc = synthesis_cli.main(["ingest-dir", "--path", "dummy-dir", "--repair-semantic"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--repair-semantic requires --repair" in captured.err


def test_cmd_ingest_file_passes_repair_semantic_flag(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_ingest_file(
        path,
        model=None,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        seen["repair"] = str(repair)
        seen["repair_semantic"] = str(repair_semantic)
        seen["path"] = str(path)
        return 1

    monkeypatch.setattr(synthesis_cli, "ingest_file", fake_ingest_file)

    rc = synthesis_cli.main(["ingest-file", "--path", "dummy.json", "--repair", "--repair-semantic"])
    captured = capsys.readouterr()

    assert rc == 0
    assert seen["repair"] == "True"
    assert seen["repair_semantic"] == "True"
    assert "ingested=1" in captured.out


def test_cmd_ingest_dir_passes_repair_semantic_flag(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_ingest_dir(
        dir_path,
        model=None,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        seen["repair"] = str(repair)
        seen["repair_semantic"] = str(repair_semantic)
        seen["path"] = str(dir_path)
        return 2

    monkeypatch.setattr(synthesis_cli, "ingest_dir", fake_ingest_dir)

    rc = synthesis_cli.main(["ingest-dir", "--path", "dummy-dir", "--repair", "--repair-semantic"])
    captured = capsys.readouterr()

    assert rc == 0
    assert seen["repair"] == "True"
    assert seen["repair_semantic"] == "True"
    assert "ingested=2" in captured.out


def test_cmd_ingest_raw_prints_failed_status(monkeypatch, capsys) -> None:
    def fake_ingest_raw_tree(
        raw_dir,
        db_path=None,
        repair=False,
        repair_semantic=False,
        raw_original_dir=None,
        raw_repaired_dir=None,
        repair_log_dir=None,
    ):
        return IngestRawReport(
            scanned_files=1,
            scanned_candidates=1,
            ingested_files=0,
            skipped_duplicates=0,
            skipped_layout=0,
            ingested_records=0,
            failed_records=1,
            warnings=("repair_failed: gpt/20260326.json (...)",),
        )

    monkeypatch.setattr(synthesis_cli, "ingest_raw_tree", fake_ingest_raw_tree)
    rc = synthesis_cli.main(["ingest", "--repair"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "status=FAILED" in captured.out


def test_synthesis_ingest_repair_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["ingest", "--repair", "--help"])
    assert exc.value.code == 0


def test_synthesis_enrich_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["enrich", "--help"])
    assert exc.value.code == 0


def test_enrich_parser_accepts_new_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "enrich",
            "--db",
            "workspace/synthesis/db/synthesis.sqlite3",
            "--topic-only",
            "--limit",
            "10",
            "--where-review-status",
            "auto_unreviewed",
            "--dry-run",
            "--ids",
            "1",
            "2",
            "3",
            "--only-unreviewed",
        ]
    )
    assert args.db.endswith("synthesis.sqlite3")
    assert args.topic_only is True
    assert args.baseline_only is False
    assert args.limit == 10
    assert args.where_review_status == "auto_unreviewed"
    assert args.dry_run is True
    assert args.ids == [1, 2, 3]
    assert args.only_unreviewed is True


def test_enrich_defaults_only_unreviewed_and_all_records_disables_it() -> None:
    parser = synthesis_cli.build_parser()
    default_args = parser.parse_args(["enrich"])
    all_args = parser.parse_args(["enrich", "--all-records"])
    assert default_args.only_unreviewed is True
    assert all_args.only_unreviewed is False


def test_cmd_enrich_prints_summary(monkeypatch, capsys) -> None:
    def fake_enrich_claims(
        *,
        db_path=None,
        topic_only=False,
        baseline_only=False,
        limit=None,
        where_review_status=None,
        dry_run=False,
        ids=None,
        only_unreviewed=False,
    ):
        assert topic_only is True
        assert dry_run is True
        return EnrichReport(
            scanned_claims=12,
            updated_claims=5,
            skipped_human_locked=2,
            topic_assigned=4,
            baseline_assigned=2,
            needs_review_count=3,
            high_confidence_count=1,
            dry_run=True,
        )

    monkeypatch.setattr(synthesis_cli, "enrich_claims", fake_enrich_claims)
    rc = synthesis_cli.main(["enrich", "--topic-only", "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "scanned_claims=12" in captured.out
    assert "updated_claims=5" in captured.out
    assert "skipped_human_locked=2" in captured.out
    assert "dry_run=1" in captured.out


def test_cluster_baselines_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["cluster", "--db", "x.sqlite3", "--rebuild", "--dry-run"])
    assert args.db == "x.sqlite3"
    assert args.rebuild is True
    assert args.dry_run is True


def test_cluster_defaults_rebuild_true() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["cluster"])
    assert args.rebuild is True


def test_cmd_cluster_baselines_prints_summary(monkeypatch, capsys) -> None:
    def fake_cluster_baselines(*, db_path=None, rebuild=False, dry_run=False):
        assert rebuild is True
        assert dry_run is True
        return BaselineClusterReport(
            scanned_claims=10,
            cluster_count=4,
            claim_link_count=9,
            high_count=1,
            medium_count=2,
            low_count=1,
            dry_run=True,
        )

    monkeypatch.setattr(synthesis_cli, "cluster_baselines", fake_cluster_baselines)
    rc = synthesis_cli.main(["cluster", "--rebuild", "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "clusters=4" in captured.out
    assert "claim_links=9" in captured.out
    assert "high=1" in captured.out
    assert "dry_run=1" in captured.out


def test_synthesis_cluster_baselines_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["cluster", "--help"])
    assert exc.value.code == 0


def test_synthesis_report_baselines_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["report", "--help"])
    assert exc.value.code == 0


def test_report_baselines_parser_accepts_required_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["report", "--db", "x.sqlite3", "--json", "--limit", "7"])
    assert args.db == "x.sqlite3"
    assert args.json_output is True
    assert args.limit == 7


def test_cmd_report_baselines_json_output(monkeypatch, capsys) -> None:
    def fake_build_baseline_report(**kwargs):
        assert kwargs["limit"] == 5
        return {
            "summary": {"total_clusters": 3},
            "topic_distribution": [],
            "mixed_clusters": [],
            "high_priority_without_baseline": [],
            "cluster_details": [],
            "filters": {"limit": 5},
        }

    monkeypatch.setattr(synthesis_cli, "build_baseline_report", fake_build_baseline_report)
    rc = synthesis_cli.main(["report", "--json", "--limit", "5"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"summary"' in captured.out
    assert '"total_clusters": 3' in captured.out


def test_synthesis_suggest_actions_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["suggest", "--help"])
    assert exc.value.code == 0


def test_suggest_actions_parser_accepts_required_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "suggest",
            "--db",
            "x.sqlite3",
            "--json",
            "--limit",
            "9",
            "--min-severity",
            "medium",
            "--sort-by",
            "severity",
        ]
    )
    assert args.db == "x.sqlite3"
    assert args.json_output is True
    assert args.limit == 9
    assert args.min_severity == "medium"
    assert args.sort_by == "severity"


def test_cmd_suggest_actions_json_output(monkeypatch, capsys) -> None:
    def fake_build_action_suggestions(**kwargs):
        assert kwargs["limit"] == 4
        assert kwargs["min_severity"] == "high"
        assert kwargs["sort_by"] == "score"
        return {
            "bottlenecks": [{"claim_id": 1}],
            "conflicts": [],
            "fragmentation": [],
            "weak_baselines": [],
            "topic_health_summary": [{"topic": "gain_tuning", "health_score": 60, "health_status": "needs_attention"}],
        }

    monkeypatch.setattr(synthesis_cli, "build_action_suggestions", fake_build_action_suggestions)
    rc = synthesis_cli.main(["suggest", "--json", "--limit", "4", "--min-severity", "high", "--sort-by", "score"])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"bottlenecks"' in captured.out
    assert '"claim_id": 1' in captured.out
    assert '"topic_health_summary"' in captured.out


def test_run_parser_defaults() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(["run"])
    assert args.path == str(RAW_DIR)
    assert args.enriched_dir.replace("\\", "/").endswith("workspace/synthesis/enriched")
    assert args.review_dir == str(REVIEW_DIR)
    assert args.repair is True
    assert args.repair_semantic is False
    assert args.report_limit == 20
    assert args.suggest_limit == 20
    assert args.review_queue_format == "jsonl"
    assert args.review_queue_limit is None
    assert args.min_severity == "high"
    assert args.sort_by == "score"
    assert args.full_output is False
    assert args.skip_proposition is False
    assert args.skip_triage is False
    assert args.skip_review_queue is False


def test_cmd_run_executes_all_steps_in_order(monkeypatch, capsys) -> None:
    seen: list[str] = []

    def fake_cmd_ingest_raw(args):
        seen.append(f"ingest:{args.repair}:{args.repair_semantic}")
        return 0

    def fake_cmd_enrich(args):
        seen.append(f"enrich:{args.only_unreviewed}")
        return 0

    def fake_cmd_cluster_baselines(args):
        seen.append(f"cluster:{args.rebuild}")
        return 0

    def fake_cmd_report_baselines(args):
        seen.append(f"report:{args.limit}")
        return 0

    def fake_cmd_suggest_actions(args):
        seen.append(f"suggest:{args.min_severity}:{args.sort_by}")
        return 0

    def fake_cmd_proposition_layer(args):
        seen.append(f"proposition:{args.raw_dir}:{args.enriched_dir}")
        return 0

    def fake_cmd_triage_propositions(args):
        seen.append(f"triage:{args.enriched_dir}:{args.review_dir}:{args.dry_run}")
        return 0

    def fake_cmd_export_review_queue(args):
        seen.append(f"queue:{args.review_dir}:{args.format}:{args.limit}")
        return 0

    monkeypatch.setattr(synthesis_cli, "cmd_ingest_raw", fake_cmd_ingest_raw)
    monkeypatch.setattr(synthesis_cli, "cmd_enrich", fake_cmd_enrich)
    monkeypatch.setattr(synthesis_cli, "cmd_cluster_baselines", fake_cmd_cluster_baselines)
    monkeypatch.setattr(synthesis_cli, "cmd_report_baselines", fake_cmd_report_baselines)
    monkeypatch.setattr(synthesis_cli, "cmd_suggest_actions", fake_cmd_suggest_actions)
    monkeypatch.setattr(synthesis_cli, "cmd_proposition_layer", fake_cmd_proposition_layer)
    monkeypatch.setattr(synthesis_cli, "cmd_triage_propositions", fake_cmd_triage_propositions)
    monkeypatch.setattr(synthesis_cli, "cmd_export_review_queue", fake_cmd_export_review_queue)

    rc = synthesis_cli.main(["run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert seen == [
        "ingest:True:False",
        "enrich:True",
        "cluster:True",
        "report:20",
        "suggest:high:score",
        f"proposition:{RAW_DIR}:{synthesis_cli.ENRICHED_DIR}",
        f"triage:{synthesis_cli.ENRICHED_DIR}:{REVIEW_DIR}:False",
        f"queue:{REVIEW_DIR}:jsonl:None",
    ]
    assert "status=SUCCESS outcome=pipeline_completed" in captured.out


def test_cmd_run_stops_on_first_failed_step(monkeypatch, capsys) -> None:
    seen: list[str] = []

    def fake_cmd_ingest_raw(args):
        seen.append("ingest")
        return 1

    def fake_cmd_enrich(args):
        seen.append("enrich")
        return 0

    monkeypatch.setattr(synthesis_cli, "cmd_ingest_raw", fake_cmd_ingest_raw)
    monkeypatch.setattr(synthesis_cli, "cmd_enrich", fake_cmd_enrich)

    rc = synthesis_cli.main(["run"])
    _ = capsys.readouterr()

    assert rc == 1
    assert seen == ["ingest"]


def test_run_repair_semantic_requires_repair(capsys) -> None:
    rc = synthesis_cli.main(["run", "--no-repair", "--repair-semantic"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--repair-semantic requires --repair" in captured.err


def test_cmd_run_skips_proposition_when_flagged(monkeypatch, capsys) -> None:
    seen: list[str] = []

    monkeypatch.setattr(synthesis_cli, "cmd_ingest_raw", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_enrich", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_cluster_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_report_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_suggest_actions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_proposition_layer", lambda args: seen.append("proposition") or 0)
    monkeypatch.setattr(synthesis_cli, "cmd_triage_propositions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_export_review_queue", lambda args: 0)

    rc = synthesis_cli.main(["run", "--skip-proposition"])
    _ = capsys.readouterr()

    assert rc == 0
    assert seen == []


def test_cmd_run_compact_ingest_output_summarizes_warnings(monkeypatch, capsys) -> None:
    def fake_cmd_ingest_raw(args):
        print("warning: WARNING [record 1] required_files: no overlap with evidence_files; verify references are intentional.")
        print("warning: WARNING [record 2] evidence_files: list is empty.")
        print("warning: repair applied: 20260327.json")
        print("status=SUCCESS outcome=ingested_new_records")
        print(
            "scanned_files=1 scanned_candidates=1 original_saved=1 repaired_saved=1 repairs_applied=1 "
            "warnings_count=3 ingested_files=1 ingested_records=1 skipped_duplicates=0 skipped_layout=0 "
            "failed_records=0 repair_failed_files=0 validation_failed_files=0"
        )
        return 0

    monkeypatch.setattr(synthesis_cli, "cmd_ingest_raw", fake_cmd_ingest_raw)
    monkeypatch.setattr(synthesis_cli, "cmd_enrich", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_cluster_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_report_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_suggest_actions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_proposition_layer", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_triage_propositions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_export_review_queue", lambda args: 0)

    rc = synthesis_cli.main(
        [
            "run",
            "--skip-enrich",
            "--skip-cluster",
            "--skip-report",
            "--skip-suggest",
            "--skip-proposition",
            "--skip-triage",
            "--skip-review-queue",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "warning_summary=total:3 repair_applied:1 no_overlap:1 evidence_files_empty:1" in captured.out
    assert "warning: WARNING [record 1]" not in captured.out


def test_cmd_run_skips_review_queue_when_dry_run_triage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(synthesis_cli, "cmd_ingest_raw", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_enrich", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_cluster_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_report_baselines", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_suggest_actions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_proposition_layer", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_triage_propositions", lambda args: 0)
    monkeypatch.setattr(synthesis_cli, "cmd_export_review_queue", lambda args: pytest.fail("queue export should be skipped in dry-run"))

    rc = synthesis_cli.main(["run", "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "status=SKIPPED outcome=review_queue_export_skipped_in_dry_run" in captured.out


def test_synthesis_set_review_status_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["set-review-status", "--help"])
    assert exc.value.code == 0


def test_synthesis_proposition_layer_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        synthesis_cli.main(["proposition-layer", "--help"])
    assert exc.value.code == 0


def test_proposition_layer_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "proposition-layer",
            "--raw-dir",
            "workspace/synthesis/raw",
            "--enriched-dir",
            "workspace/synthesis/enriched",
            "--db",
            "workspace/synthesis/db/synthesis.sqlite3",
        ]
    )
    assert args.raw_dir.endswith("workspace/synthesis/raw")
    assert args.enriched_dir.endswith("workspace/synthesis/enriched")
    assert args.db.endswith("synthesis.sqlite3")


def test_cmd_proposition_layer_prints_summary(monkeypatch, capsys) -> None:
    def fake_build_proposition_layer(*, raw_dir=None, enriched_dir=None, db_path=None):
        assert str(raw_dir).replace("\\", "/").endswith("workspace/synthesis/raw")
        assert str(enriched_dir).replace("\\", "/").endswith("workspace/synthesis/enriched")
        assert str(db_path).replace("\\", "/").endswith("synthesis.sqlite3")
        return PropositionLayerReport(
            scanned_files=4,
            loaded_files=4,
            repaired_files=1,
            failed_files=0,
            claim_count=12,
            mapped_claim_count=11,
            orphan_count=1,
            proposition_count=11,
            relation_count=17,
            db_proposition_count=11,
            db_relation_count=17,
            validation_messages=("v1", "v2"),
            orphan_messages=("o1",),
            enriched_outputs=("x",),
            integrated_outputs=("y",),
        )

    monkeypatch.setattr(synthesis_cli, "build_proposition_layer", fake_build_proposition_layer)
    rc = synthesis_cli.main(
        [
            "proposition-layer",
            "--raw-dir",
            "workspace/synthesis/raw",
            "--enriched-dir",
            "workspace/synthesis/enriched",
            "--db",
            "workspace/synthesis/db/synthesis.sqlite3",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "validation: v1" in captured.out
    assert "orphan: o1" in captured.out
    assert "claims=12" in captured.out


def test_triage_propositions_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "triage-propositions",
            "--db",
            "workspace/synthesis/db/synthesis.sqlite3",
            "--enriched-dir",
            "workspace/synthesis/enriched",
            "--review-dir",
            "workspace/synthesis/review",
            "--triage-run-id",
            "RUN-001",
            "--output",
            "workspace/synthesis/review/triage/run-001.jsonl",
            "--dry-run",
        ]
    )
    assert args.db.endswith("synthesis.sqlite3")
    assert args.enriched_dir.endswith("workspace/synthesis/enriched")
    assert args.review_dir.endswith("workspace/synthesis/review")
    assert args.triage_run_id == "RUN-001"
    assert args.output.endswith("run-001.jsonl")
    assert args.dry_run is True


def test_cmd_triage_propositions_prints_summary(monkeypatch, capsys) -> None:
    class DummyReport:
        triage_run_id = "RUN-001"
        scanned_propositions = 11
        triage_records_written = 11
        queue_candidates = 7
        output_path = "workspace/synthesis/review/triage/proposition_triage_RUN-001.jsonl"
        dry_run = False

    def fake_triage_propositions(**kwargs):
        assert kwargs["triage_run_id"] == "RUN-001"
        return DummyReport()

    monkeypatch.setattr(synthesis_cli, "triage_propositions", fake_triage_propositions)
    rc = synthesis_cli.main(["triage-propositions", "--triage-run-id", "RUN-001"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "triage_run_id=RUN-001" in captured.out
    assert "triage_records_written=11" in captured.out


def test_export_review_queue_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "export-review-queue",
            "--db",
            "x.sqlite3",
            "--review-dir",
            "workspace/synthesis/review",
            "--output",
            "workspace/synthesis/review/queue/latest.csv",
            "--format",
            "csv",
            "--limit",
            "10",
            "--include-final",
        ]
    )
    assert args.db == "x.sqlite3"
    assert args.review_dir.endswith("workspace/synthesis/review")
    assert args.output.endswith("latest.csv")
    assert args.format == "csv"
    assert args.limit == 10
    assert args.include_final is True


def test_cmd_export_review_queue_prints_summary(monkeypatch, capsys) -> None:
    class DummyReport:
        exported_rows = 5
        output_path = "workspace/synthesis/review/queue/review_queue.csv"
        format = "csv"
        include_final = True

    def fake_export_review_queue(**kwargs):
        assert kwargs["format_name"] == "csv"
        return DummyReport()

    monkeypatch.setattr(synthesis_cli, "export_review_queue", fake_export_review_queue)
    rc = synthesis_cli.main(["export-review-queue", "--format", "csv", "--include-final"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "exported_rows=5" in captured.out
    assert "format=csv" in captured.out


def test_set_proposition_review_status_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "set-proposition-review-status",
            "--db",
            "x.sqlite3",
            "--proposition-ids",
            "PROP-001",
            "PROP-002",
            "--triage-record-ids",
            "12",
            "15",
            "--review-status",
            "human_confirmed",
            "--dry-run",
            "--limit",
            "5",
            "--where-current-status",
            "triaged",
            "--reason",
            "manual review",
            "--allow-human-overwrite",
            "--force",
        ]
    )
    assert args.db == "x.sqlite3"
    assert args.proposition_ids == ["PROP-001", "PROP-002"]
    assert args.triage_record_ids == [12, 15]
    assert args.review_status == "human_confirmed"
    assert args.dry_run is True
    assert args.limit == 5
    assert args.where_current_status == "triaged"
    assert args.reason == "manual review"
    assert args.allow_human_overwrite is True
    assert args.force is True


def test_cmd_set_proposition_review_status_prints_summary(monkeypatch, capsys) -> None:
    class DummyReport:
        requested_status = "human_confirmed"
        matched_records = 2
        updated_records = 1
        unchanged_records = 0
        skipped_locked_records = 1
        skipped_missing_records = 0
        skipped_transition_records = 0
        history_records_written = 1
        reason_used = "manual"
        dry_run = True
        updated_ids = ("PROP-001",)
        skipped_locked_ids = ("PROP-002",)
        skipped_missing_ids = ()
        skipped_transition_ids = ()

    def fake_set_proposition_review_status(**kwargs):
        assert kwargs["review_status"] == "human_confirmed"
        assert kwargs["dry_run"] is True
        return DummyReport()

    monkeypatch.setattr(synthesis_cli, "set_proposition_review_status", fake_set_proposition_review_status)
    rc = synthesis_cli.main(
        [
            "set-proposition-review-status",
            "--proposition-ids",
            "PROP-001",
            "--review-status",
            "human_confirmed",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "requested_status=human_confirmed" in captured.out
    assert "updated_records=1" in captured.out
    assert "skipped_locked_records=1" in captured.out


def test_set_review_status_parser_accepts_flags() -> None:
    parser = synthesis_cli.build_parser()
    args = parser.parse_args(
        [
            "set-review-status",
            "--db",
            "x.sqlite3",
            "--ids",
            "1",
            "2",
            "--review-status",
            "human_confirmed",
            "--dry-run",
            "--topic",
            "gain_tuning",
            "--baseline-cluster-id",
            "9",
            "--limit",
            "10",
            "--where-current-status",
            "needs_review",
            "--reason",
            "manual review done",
            "--allow-human-overwrite",
        ]
    )
    assert args.db == "x.sqlite3"
    assert args.ids == [1, 2]
    assert args.review_status == "human_confirmed"
    assert args.dry_run is True
    assert args.topic == "gain_tuning"
    assert args.baseline_cluster_id == 9
    assert args.limit == 10
    assert args.where_current_status == "needs_review"
    assert args.reason == "manual review done"
    assert args.allow_human_overwrite is True


def test_cmd_set_review_status_prints_summary(monkeypatch, capsys) -> None:
    class DummyReport:
        requested_status = "human_confirmed"
        matched_records = 3
        updated_records = 2
        unchanged_records = 0
        skipped_locked_records = 1
        skipped_missing_records = 1
        history_records_written = 2
        reason_used = "manual"
        dry_run = True
        updated_ids = (1, 2)
        skipped_locked_ids = (3,)
        skipped_missing_ids = (999,)

    def fake_set_review_status(**kwargs):
        assert kwargs["review_status"] == "human_confirmed"
        assert kwargs["dry_run"] is True
        assert kwargs["reason"] == "manual"
        return DummyReport()

    monkeypatch.setattr(synthesis_cli, "set_review_status", fake_set_review_status)
    rc = synthesis_cli.main(["set-review-status", "--ids", "1", "--review-status", "human_confirmed", "--reason", "manual", "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "requested_status=human_confirmed" in captured.out
    assert "updated_records=2" in captured.out
    assert "history_records_written=2" in captured.out
