from __future__ import annotations

import argparse
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

from arena.synthesis.cluster_baselines import cluster_baselines
from arena.synthesis.db import init_db
from arena.synthesis.enrich import enrich_claims
from arena.synthesis.ingest import ingest_dir, ingest_file, ingest_raw_tree
from arena.synthesis.paths import RAW_ORIGINAL_DIR, RAW_REPAIRED_DIR, REPAIR_LOG_DIR, as_dict
from arena.synthesis.proposition_layer import build_proposition_layer
from arena.synthesis.proposition_review import (
    export_review_queue,
    render_proposition_review_update_result,
    render_queue_export_result,
    render_triage_result,
    set_proposition_review_status,
    triage_propositions,
)
from arena.synthesis.report_baselines import (
    build_baseline_report,
    render_baseline_report_json,
    render_baseline_report_text,
)
from arena.synthesis.review_status import render_review_update_result, set_review_status
from arena.synthesis.suggest_actions import (
    build_action_suggestions,
    render_action_suggestions_json,
    render_action_suggestions_text,
)

MODEL_CHOICES = ("claude", "gemini", "gpt", "grok")


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _print_captured_output(text: str) -> None:
    if not text:
        return
    print(text, end="" if text.endswith("\n") else "\n")


def _run_step_and_render(
    fn,
    args: argparse.Namespace,
    *,
    full_output: bool,
    compact_renderer,
) -> int:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = int(fn(args))
    captured = buffer.getvalue()

    if full_output or rc != 0:
        _print_captured_output(captured)
        return rc

    compact_renderer(captured)
    return rc


def _render_compact_ingest(captured: str) -> None:
    lines = _nonempty_lines(captured)
    warning_lines = [line for line in lines if line.startswith("warning: ")]
    status_line = next((line for line in lines if line.startswith("status=")), None)
    summary_line = next((line for line in lines if line.startswith("scanned_files=")), None)

    if status_line is not None:
        print(status_line)

    if warning_lines:
        repair_applied = sum(1 for line in warning_lines if "repair applied:" in line)
        no_overlap = sum(1 for line in warning_lines if "no overlap with evidence_files" in line)
        evidence_files_empty = sum(1 for line in warning_lines if "evidence_files: list is empty" in line)
        required_without_evidence = sum(
            1 for line in warning_lines if "required_files is present but evidence_files is empty" in line
        )
        refs_without_evidence = sum(
            1 for line in warning_lines if "evidence_refs is present but evidence_files is empty" in line
        )
        known = (
            repair_applied + no_overlap + evidence_files_empty + required_without_evidence + refs_without_evidence
        )
        other = max(0, len(warning_lines) - known)
        print(
            "warning_summary="
            f"total:{len(warning_lines)} "
            f"repair_applied:{repair_applied} "
            f"no_overlap:{no_overlap} "
            f"evidence_files_empty:{evidence_files_empty} "
            f"required_without_evidence:{required_without_evidence} "
            f"refs_without_evidence:{refs_without_evidence} "
            f"other:{other}"
        )

    if summary_line is not None:
        print(summary_line)


def _render_compact_single_line(captured: str) -> None:
    lines = _nonempty_lines(captured)
    if lines:
        print(lines[-1])


def _render_compact_report(captured: str) -> None:
    lines = _nonempty_lines(captured)
    summary_line = next((line for line in lines if line.startswith("summary:")), None)
    top_topic_line = next((line for line in lines if line.startswith("- topic=")), None)
    if summary_line is not None:
        print(summary_line)
    if top_topic_line is not None:
        print(f"top_topic={top_topic_line[2:]}")


def _render_compact_suggest(captured: str) -> None:
    lines = _nonempty_lines(captured)
    section = None
    counts = {
        "bottlenecks": 0,
        "conflicts": 0,
        "fragmentation": 0,
        "topic_health": 0,
    }
    for line in lines:
        if line.startswith("[HIGH] Missing baseline"):
            section = "bottlenecks"
            continue
        if line.startswith("[MEDIUM] Mixed cluster detected"):
            section = "conflicts"
            continue
        if line.startswith("[LOW] Fragmented baseline"):
            section = "fragmentation"
            continue
        if line.startswith("Topic Health Summary:"):
            section = "topic_health"
            continue
        if line.startswith("- ") and section is not None:
            counts[section] += 1

    print(
        "suggest_summary="
        f"bottlenecks:{counts['bottlenecks']} "
        f"conflicts:{counts['conflicts']} "
        f"fragmentation:{counts['fragmentation']} "
        f"topic_health:{counts['topic_health']}"
    )


def _render_compact_proposition(captured: str) -> None:
    lines = _nonempty_lines(captured)
    mismatch_count = sum(1 for line in lines if line.startswith("validation: bidirectional_mismatch:"))
    orphan_count = sum(1 for line in lines if line.startswith("orphan: "))
    assignment_line = next((line for line in lines if "claim_assignment_summary:" in line), None)
    assignment_suffix = ""
    if assignment_line is not None:
        match = re.search(r"total=(\d+)\s+mapped=(\d+)\s+orphan=(\d+)", assignment_line)
        if match is not None:
            assignment_suffix = (
                f" claim_assignment_total={match.group(1)}"
                f" mapped={match.group(2)}"
                f" orphan={match.group(3)}"
            )
    print(
        f"validation_summary=bidirectional_mismatch:{mismatch_count} orphan_messages:{orphan_count}"
        f"{assignment_suffix}"
    )
    summary_line = next((line for line in lines if line.startswith("scanned_files=")), None)
    if summary_line is not None:
        print(summary_line)


def cmd_init_db(args: argparse.Namespace) -> int:
    path = init_db()
    print(f"db_path={path}")
    return 0


def cmd_show_paths(args: argparse.Namespace) -> int:
    for k, v in as_dict().items():
        print(f"{k}={v}")
    return 0


def cmd_ingest_file(args: argparse.Namespace) -> int:
    if bool(getattr(args, "repair_semantic", False)) and not bool(args.repair):
        raise ValueError("--repair-semantic requires --repair")
    path = Path(args.path)
    count = ingest_file(
        path,
        model=args.model,
        db_path=Path(args.db) if args.db else None,
        repair=bool(args.repair),
        repair_semantic=bool(args.repair_semantic),
        raw_original_dir=Path(args.raw_original_dir),
        raw_repaired_dir=Path(args.raw_repaired_dir),
        repair_log_dir=Path(args.repair_log_dir),
    )
    print(f"ingested={count}")
    return 0


def cmd_ingest_dir(args: argparse.Namespace) -> int:
    if bool(getattr(args, "repair_semantic", False)) and not bool(args.repair):
        raise ValueError("--repair-semantic requires --repair")
    dir_path = Path(args.path)
    count = ingest_dir(
        dir_path,
        model=args.model,
        db_path=Path(args.db) if args.db else None,
        repair=bool(args.repair),
        repair_semantic=bool(args.repair_semantic),
        raw_original_dir=Path(args.raw_original_dir),
        raw_repaired_dir=Path(args.raw_repaired_dir),
        repair_log_dir=Path(args.repair_log_dir),
    )
    print(f"ingested={count}")
    return 0


def cmd_ingest_raw(args: argparse.Namespace) -> int:
    if bool(getattr(args, "repair_semantic", False)) and not bool(args.repair):
        raise ValueError("--repair-semantic requires --repair")
    report = ingest_raw_tree(
        raw_dir=Path(args.path),
        db_path=Path(args.db) if args.db else None,
        repair=bool(args.repair),
        repair_semantic=bool(args.repair_semantic),
        raw_original_dir=Path(args.raw_original_dir),
        raw_repaired_dir=Path(args.raw_repaired_dir),
        repair_log_dir=Path(args.repair_log_dir),
    )
    for warning in report.warnings:
        print(f"warning: {warning}")
    if report.failed_records > 0:
        print("status=FAILED outcome=partial_ingest_failure")
    else:
        print("status=SUCCESS outcome=ingested_new_records")
    print(
        "scanned_files="
        f"{report.scanned_files} "
        f"scanned_candidates={report.scanned_candidates} "
        f"original_saved={report.original_saved} "
        f"repaired_saved={report.repaired_saved} "
        f"repairs_applied={report.repairs_applied} "
        f"warnings_count={report.warnings_count} "
        f"ingested_files={report.ingested_files} "
        f"ingested_records={report.ingested_records} "
        f"skipped_duplicates={report.skipped_duplicates} "
        f"skipped_layout={report.skipped_layout} "
        f"failed_records={report.failed_records} "
        f"repair_failed_files={report.repair_failed_files} "
        f"validation_failed_files={report.validation_failed_files}"
    )
    return 0 if report.failed_records == 0 else 1


def cmd_enrich(args: argparse.Namespace) -> int:
    report = enrich_claims(
        db_path=Path(args.db) if args.db else None,
        topic_only=bool(args.topic_only),
        baseline_only=bool(args.baseline_only),
        limit=args.limit,
        where_review_status=args.where_review_status,
        dry_run=bool(args.dry_run),
        ids=args.ids,
        only_unreviewed=bool(args.only_unreviewed),
    )
    print(
        f"scanned_claims={report.scanned_claims} updated_claims={report.updated_claims} "
        f"skipped_human_locked={report.skipped_human_locked} "
        f"topic_assigned={report.topic_assigned} baseline_assigned={report.baseline_assigned} "
        f"needs_review_count={report.needs_review_count} high_confidence_count={report.high_confidence_count} "
        f"dry_run={int(report.dry_run)}"
    )
    return 0


def cmd_cluster_baselines(args: argparse.Namespace) -> int:
    report = cluster_baselines(
        db_path=Path(args.db) if args.db else None,
        rebuild=bool(args.rebuild),
        dry_run=bool(args.dry_run),
    )
    print(
        f"scanned_claims={report.scanned_claims} clusters={report.cluster_count} "
        f"claim_links={report.claim_link_count} high={report.high_count} "
        f"medium={report.medium_count} low={report.low_count} dry_run={int(report.dry_run)}"
    )
    return 0


def cmd_report_baselines(args: argparse.Namespace) -> int:
    report = build_baseline_report(
        db_path=Path(args.db) if args.db else None,
        limit=args.limit,
        topic=args.topic,
        only_mixed=bool(args.only_mixed),
        only_high_priority_missing=bool(args.only_high_priority_missing),
        include_cluster_details=bool(args.include_cluster_details),
    )
    if bool(args.json_output):
        print(render_baseline_report_json(report))
    else:
        print(render_baseline_report_text(report))
    return 0


def cmd_suggest_actions(args: argparse.Namespace) -> int:
    report = build_action_suggestions(
        db_path=Path(args.db) if args.db else None,
        limit=args.limit,
        topic=args.topic,
        only_high=bool(args.only_high),
        only_conflicts=bool(args.only_conflicts),
        min_severity=args.min_severity,
        sort_by=args.sort_by,
    )
    if bool(args.json_output):
        print(render_action_suggestions_json(report))
    else:
        print(render_action_suggestions_text(report))
    return 0


def cmd_set_review_status(args: argparse.Namespace) -> int:
    report = set_review_status(
        db_path=Path(args.db) if args.db else None,
        ids=args.ids,
        topic=args.topic,
        baseline_cluster_id=args.baseline_cluster_id,
        review_status=args.review_status,
        dry_run=bool(args.dry_run),
        limit=args.limit,
        where_current_status=args.where_current_status,
        reason=args.reason,
        allow_human_overwrite=bool(args.allow_human_overwrite),
    )
    print(render_review_update_result(report))
    return 0


def cmd_proposition_layer(args: argparse.Namespace) -> int:
    report = build_proposition_layer(
        raw_dir=Path(args.raw_dir),
        enriched_dir=Path(args.enriched_dir),
        db_path=Path(args.db) if args.db else None,
    )
    for message in report.validation_messages:
        print(f"validation: {message}")
    for message in report.orphan_messages:
        print(f"orphan: {message}")
    print(
        f"scanned_files={report.scanned_files} loaded_files={report.loaded_files} repaired_files={report.repaired_files} "
        f"failed_files={report.failed_files} claims={report.claim_count} mapped_claims={report.mapped_claim_count} "
        f"orphans={report.orphan_count} propositions={report.proposition_count} relations={report.relation_count} "
        f"db_propositions={report.db_proposition_count} db_relations={report.db_relation_count} "
        f"enriched_outputs={len(report.enriched_outputs)} integrated_outputs={len(report.integrated_outputs)}"
    )
    return 0


def cmd_triage_propositions(args: argparse.Namespace) -> int:
    report = triage_propositions(
        db_path=Path(args.db) if args.db else None,
        enriched_dir=Path(args.enriched_dir),
        review_dir=Path(args.review_dir),
        triage_run_id=args.triage_run_id,
        output_path=Path(args.output) if args.output else None,
        dry_run=bool(args.dry_run),
    )
    print(render_triage_result(report))
    return 0


def cmd_export_review_queue(args: argparse.Namespace) -> int:
    report = export_review_queue(
        db_path=Path(args.db) if args.db else None,
        review_dir=Path(args.review_dir),
        output_path=Path(args.output) if args.output else None,
        format_name=args.format,
        limit=args.limit,
        include_final=bool(args.include_final),
    )
    print(render_queue_export_result(report))
    return 0


def cmd_set_proposition_review_status(args: argparse.Namespace) -> int:
    report = set_proposition_review_status(
        db_path=Path(args.db) if args.db else None,
        proposition_ids=args.proposition_ids,
        triage_record_ids=args.triage_record_ids,
        review_status=args.review_status,
        dry_run=bool(args.dry_run),
        limit=args.limit,
        where_current_status=args.where_current_status,
        allow_human_overwrite=bool(args.allow_human_overwrite),
        force=bool(args.force),
        reason=args.reason,
    )
    print(render_proposition_review_update_result(report))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    full_output = bool(args.full_output)

    if bool(getattr(args, "repair_semantic", False)) and not bool(args.repair):
        raise ValueError("--repair-semantic requires --repair")

    print("pipeline=synthesis_daily_run")

    if not bool(args.skip_ingest):
        print("step=ingest")
        rc = _run_step_and_render(
            cmd_ingest_raw,
            argparse.Namespace(
                db=args.db,
                path=args.path,
                repair=args.repair,
                repair_semantic=args.repair_semantic,
                raw_original_dir=args.raw_original_dir,
                raw_repaired_dir=args.raw_repaired_dir,
                repair_log_dir=args.repair_log_dir,
            ),
            full_output=full_output,
            compact_renderer=_render_compact_ingest,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_enrich):
        print("step=enrich")
        rc = _run_step_and_render(
            cmd_enrich,
            argparse.Namespace(
                db=args.db,
                topic_only=False,
                baseline_only=False,
                limit=None,
                where_review_status=None,
                dry_run=bool(args.dry_run),
                ids=None,
                only_unreviewed=True,
            ),
            full_output=full_output,
            compact_renderer=_render_compact_single_line,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_cluster):
        print("step=cluster")
        rc = _run_step_and_render(
            cmd_cluster_baselines,
            argparse.Namespace(
                db=args.db,
                rebuild=True,
                dry_run=bool(args.dry_run),
            ),
            full_output=full_output,
            compact_renderer=_render_compact_single_line,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_report):
        print("step=report")
        rc = _run_step_and_render(
            cmd_report_baselines,
            argparse.Namespace(
                db=args.db,
                json_output=bool(args.json_output),
                limit=args.report_limit,
                topic=None,
                only_mixed=False,
                only_high_priority_missing=False,
                include_cluster_details=False,
            ),
            full_output=full_output,
            compact_renderer=_render_compact_report,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_suggest):
        print("step=suggest")
        rc = _run_step_and_render(
            cmd_suggest_actions,
            argparse.Namespace(
                db=args.db,
                json_output=bool(args.json_output),
                limit=args.suggest_limit,
                topic=None,
                only_high=False,
                only_conflicts=False,
                min_severity=args.min_severity,
                sort_by=args.sort_by,
            ),
            full_output=full_output,
            compact_renderer=_render_compact_suggest,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_proposition):
        print("step=proposition")
        rc = _run_step_and_render(
            cmd_proposition_layer,
            argparse.Namespace(
                raw_dir=args.path,
                enriched_dir=args.enriched_dir,
                db=args.db,
            ),
            full_output=full_output,
            compact_renderer=_render_compact_proposition,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_triage):
        print("step=triage")
        rc = _run_step_and_render(
            cmd_triage_propositions,
            argparse.Namespace(
                db=args.db,
                enriched_dir=args.enriched_dir,
                review_dir=args.review_dir,
                triage_run_id=None,
                output=None,
                dry_run=bool(args.dry_run),
            ),
            full_output=full_output,
            compact_renderer=_render_compact_single_line,
        )
        if rc != 0:
            return rc

    if not bool(args.skip_review_queue):
        print("step=review_queue_export")
        if bool(args.dry_run) and not bool(args.skip_triage):
            print("status=SKIPPED outcome=review_queue_export_skipped_in_dry_run")
        else:
            rc = _run_step_and_render(
                cmd_export_review_queue,
                argparse.Namespace(
                    db=args.db,
                    review_dir=args.review_dir,
                    output=None,
                    format=args.review_queue_format,
                    limit=args.review_queue_limit,
                    include_final=False,
                ),
                full_output=full_output,
                compact_renderer=_render_compact_single_line,
            )
            if rc != 0:
                return rc

    print("status=SUCCESS outcome=pipeline_completed")
    return 0


def add_repair_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repair", action="store_true", help="Attempt safe JSON repair when parsing fails.")
    parser.add_argument(
        "--repair-semantic",
        action="store_true",
        help="Enable semantic repair rules (requires --repair).",
    )
    parser.add_argument("--raw-original-dir", default=str(RAW_ORIGINAL_DIR), help="Directory to persist original raw snapshots.")
    parser.add_argument("--raw-repaired-dir", default=str(RAW_REPAIRED_DIR), help="Directory to persist repaired ingest candidates.")
    parser.add_argument("--repair-log-dir", default=str(REPAIR_LOG_DIR), help="Directory to append repair ingest logs.")
