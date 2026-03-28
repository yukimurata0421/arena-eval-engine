from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from arena.synthesis import cli_handlers as _handlers
from arena.synthesis.cluster_baselines import cluster_baselines
from arena.synthesis.enrich import enrich_claims
from arena.synthesis.ingest import ingest_dir, ingest_file, ingest_raw_tree
from arena.synthesis.paths import ENRICHED_DIR, RAW_DIR, RAW_ORIGINAL_DIR, RAW_REPAIRED_DIR, REPAIR_LOG_DIR, REVIEW_DIR
from arena.synthesis.proposition_layer import build_proposition_layer
from arena.synthesis.proposition_review import export_review_queue, set_proposition_review_status, triage_propositions
from arena.synthesis.report_baselines import build_baseline_report
from arena.synthesis.review_status import set_review_status
from arena.synthesis.suggest_actions import build_action_suggestions

MODEL_CHOICES = _handlers.MODEL_CHOICES
add_repair_args = _handlers.add_repair_args

_CMD_INIT_DB_IMPL = _handlers.cmd_init_db
_CMD_SHOW_PATHS_IMPL = _handlers.cmd_show_paths
_CMD_INGEST_FILE_IMPL = _handlers.cmd_ingest_file
_CMD_INGEST_DIR_IMPL = _handlers.cmd_ingest_dir
_CMD_INGEST_RAW_IMPL = _handlers.cmd_ingest_raw
_CMD_ENRICH_IMPL = _handlers.cmd_enrich
_CMD_CLUSTER_BASELINES_IMPL = _handlers.cmd_cluster_baselines
_CMD_REPORT_BASELINES_IMPL = _handlers.cmd_report_baselines
_CMD_SUGGEST_ACTIONS_IMPL = _handlers.cmd_suggest_actions
_CMD_SET_REVIEW_STATUS_IMPL = _handlers.cmd_set_review_status
_CMD_PROPOSITION_LAYER_IMPL = _handlers.cmd_proposition_layer
_CMD_TRIAGE_PROPOSITIONS_IMPL = _handlers.cmd_triage_propositions
_CMD_EXPORT_REVIEW_QUEUE_IMPL = _handlers.cmd_export_review_queue
_CMD_SET_PROPOSITION_REVIEW_STATUS_IMPL = _handlers.cmd_set_proposition_review_status
_CMD_RUN_IMPL = _handlers.cmd_run


def _sync_handler_dependencies() -> None:
    _handlers.ingest_raw_tree = ingest_raw_tree
    _handlers.ingest_file = ingest_file
    _handlers.ingest_dir = ingest_dir
    _handlers.enrich_claims = enrich_claims
    _handlers.cluster_baselines = cluster_baselines
    _handlers.build_baseline_report = build_baseline_report
    _handlers.build_action_suggestions = build_action_suggestions
    _handlers.build_proposition_layer = build_proposition_layer
    _handlers.triage_propositions = triage_propositions
    _handlers.export_review_queue = export_review_queue
    _handlers.set_proposition_review_status = set_proposition_review_status
    _handlers.set_review_status = set_review_status


def cmd_init_db(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_INIT_DB_IMPL(args)


def cmd_show_paths(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_SHOW_PATHS_IMPL(args)


def cmd_ingest_file(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_INGEST_FILE_IMPL(args)


def cmd_ingest_dir(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_INGEST_DIR_IMPL(args)


def cmd_ingest_raw(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_INGEST_RAW_IMPL(args)


def cmd_enrich(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_ENRICH_IMPL(args)


def cmd_cluster_baselines(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_CLUSTER_BASELINES_IMPL(args)


def cmd_report_baselines(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_REPORT_BASELINES_IMPL(args)


def cmd_suggest_actions(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_SUGGEST_ACTIONS_IMPL(args)


def cmd_set_review_status(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_SET_REVIEW_STATUS_IMPL(args)


def cmd_proposition_layer(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_PROPOSITION_LAYER_IMPL(args)


def cmd_triage_propositions(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_TRIAGE_PROPOSITIONS_IMPL(args)


def cmd_export_review_queue(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_EXPORT_REVIEW_QUEUE_IMPL(args)


def cmd_set_proposition_review_status(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    return _CMD_SET_PROPOSITION_REVIEW_STATUS_IMPL(args)


def cmd_run(args: argparse.Namespace) -> int:
    _sync_handler_dependencies()
    # Preserve monkeypatch behavior from synthesis.cli tests.
    _handlers.cmd_ingest_raw = cmd_ingest_raw
    _handlers.cmd_enrich = cmd_enrich
    _handlers.cmd_cluster_baselines = cmd_cluster_baselines
    _handlers.cmd_report_baselines = cmd_report_baselines
    _handlers.cmd_suggest_actions = cmd_suggest_actions
    _handlers.cmd_proposition_layer = cmd_proposition_layer
    _handlers.cmd_triage_propositions = cmd_triage_propositions
    _handlers.cmd_export_review_queue = cmd_export_review_queue
    return _CMD_RUN_IMPL(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena.synthesis.cli")
    sub = p.add_subparsers(required=True)

    p_init = sub.add_parser("init-db")
    p_init.set_defaults(func=cmd_init_db)

    p_paths = sub.add_parser("show-paths")
    p_paths.set_defaults(func=cmd_show_paths)

    p_run = sub.add_parser(
        "run",
        help="Run daily synthesis pipeline (ingest -> enrich -> cluster -> report -> suggest -> proposition -> triage -> review_queue_export).",
    )
    p_run.add_argument("--path", required=False, default=str(RAW_DIR), help="Raw input directory root.")
    p_run.add_argument("--enriched-dir", required=False, default=str(ENRICHED_DIR), help="Output directory for enriched JSON.")
    p_run.add_argument("--review-dir", required=False, default=str(REVIEW_DIR), help="Directory for review artifacts (triage/queue exports).")
    p_run.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_run.add_argument("--repair", dest="repair", action="store_true", help="Enable safe JSON repair in ingest step.")
    p_run.add_argument("--no-repair", dest="repair", action="store_false", help="Disable JSON repair in ingest step.")
    p_run.add_argument(
        "--repair-semantic",
        action="store_true",
        help="Enable semantic repair rules in ingest step (requires --repair).",
    )
    p_run.set_defaults(repair=True)
    p_run.add_argument("--raw-original-dir", default=str(RAW_ORIGINAL_DIR), help="Directory to persist original raw snapshots.")
    p_run.add_argument("--raw-repaired-dir", default=str(RAW_REPAIRED_DIR), help="Directory to persist repaired ingest candidates.")
    p_run.add_argument("--repair-log-dir", default=str(REPAIR_LOG_DIR), help="Directory to append repair ingest logs.")
    p_run.add_argument("--dry-run", action="store_true", help="Dry-run for enrich/cluster steps (ingest is still executed).")
    p_run.add_argument("--json", action="store_true", dest="json_output", help="Use JSON output for report/suggest.")
    p_run.add_argument("--report-limit", required=False, type=int, default=20, help="Max rows for report step.")
    p_run.add_argument("--suggest-limit", required=False, type=int, default=20, help="Max rows for suggest step.")
    p_run.add_argument("--review-queue-format", required=False, choices=("jsonl", "csv"), default="jsonl", help="Queue export format for run step.")
    p_run.add_argument("--review-queue-limit", required=False, type=int, default=None, help="Optional max rows for review queue export step.")
    p_run.add_argument("--min-severity", required=False, choices=("high", "medium", "low"), default="high", help="Minimum severity for suggest step.")
    p_run.add_argument("--sort-by", required=False, choices=("score", "severity"), default="score", help="Sort key for suggest step.")
    p_run.add_argument("--full-output", action="store_true", help="Print full step outputs instead of compact run summary.")
    p_run.add_argument("--skip-ingest", action="store_true", help="Skip ingest step.")
    p_run.add_argument("--skip-enrich", action="store_true", help="Skip enrich step.")
    p_run.add_argument("--skip-cluster", action="store_true", help="Skip cluster step.")
    p_run.add_argument("--skip-report", action="store_true", help="Skip report step.")
    p_run.add_argument("--skip-suggest", action="store_true", help="Skip suggest step.")
    p_run.add_argument("--skip-proposition", action="store_true", help="Skip proposition step.")
    p_run.add_argument("--skip-triage", action="store_true", help="Skip proposition triage step.")
    p_run.add_argument("--skip-review-queue", action="store_true", help="Skip review queue export step.")
    p_run.set_defaults(func=cmd_run)

    p_file = sub.add_parser("ingest-file")
    p_file.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_file.add_argument("--model", required=False, choices=MODEL_CHOICES)
    p_file.add_argument("--path", required=True)
    add_repair_args(p_file)
    p_file.set_defaults(func=cmd_ingest_file)

    p_dir = sub.add_parser("ingest-dir")
    p_dir.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_dir.add_argument("--model", required=False, choices=MODEL_CHOICES)
    p_dir.add_argument("--path", required=True)
    add_repair_args(p_dir)
    p_dir.set_defaults(func=cmd_ingest_dir)

    p_raw = sub.add_parser("ingest-raw")
    p_raw.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_raw.add_argument("--path", required=False, default=str(RAW_DIR))
    add_repair_args(p_raw)
    p_raw.set_defaults(func=cmd_ingest_raw)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_ingest.add_argument("--path", required=False, default=str(RAW_DIR))
    add_repair_args(p_ingest)
    p_ingest.add_argument("--no-repair", dest="repair", action="store_false", help="Disable JSON repair for ingest.")
    p_ingest.set_defaults(repair=True)
    p_ingest.set_defaults(func=cmd_ingest_raw)

    p_enrich = sub.add_parser("enrich")
    p_enrich.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_enrich.add_argument("--topic-only", action="store_true", help="Only run topic/axes enrich fields.")
    p_enrich.add_argument("--baseline-only", action="store_true", help="Only run baseline enrich fields.")
    p_enrich.add_argument("--limit", required=False, type=int, default=None, help="Process at most N claims.")
    p_enrich.add_argument("--where-review-status", required=False, default=None, help="Filter claims by exact review_status.")
    p_enrich.add_argument("--dry-run", action="store_true", help="Compute enrich updates but do not write DB.")
    p_enrich.add_argument("--ids", nargs="+", type=int, default=None, help="Only enrich specific claim IDs.")
    p_enrich.add_argument(
        "--only-unreviewed",
        action="store_true",
        help="Only process unreviewed/auto review_status claims (human-reviewed records stay skipped).",
    )
    p_enrich.add_argument(
        "--all-records",
        action="store_false",
        dest="only_unreviewed",
        help="Process all records (disable default only-unreviewed behavior).",
    )
    p_enrich.set_defaults(only_unreviewed=True)
    p_enrich.set_defaults(func=cmd_enrich)

    p_cluster = sub.add_parser("cluster", aliases=["cluster-baselines"])
    p_cluster.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_cluster.add_argument("--rebuild", action="store_true", help="Rebuild clusters and links from current claims baseline fields.")
    p_cluster.add_argument("--no-rebuild", action="store_false", dest="rebuild", help="Disable rebuild mode.")
    p_cluster.add_argument("--dry-run", action="store_true", help="Compute cluster plan without writing DB.")
    p_cluster.set_defaults(rebuild=True)
    p_cluster.set_defaults(func=cmd_cluster_baselines)

    p_report = sub.add_parser("report", aliases=["report-baselines"])
    p_report.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_report.add_argument("--json", action="store_true", dest="json_output", help="Output machine-readable JSON report.")
    p_report.add_argument("--limit", required=False, type=int, default=20, help="Max rows per section (default: 20).")
    p_report.add_argument("--topic", required=False, default=None, help="Filter report by topic.")
    p_report.add_argument("--only-mixed", action="store_true", help="Show mixed cluster section only.")
    p_report.add_argument("--only-high-priority-missing", action="store_true", help="Show high-priority missing-baseline section only.")
    p_report.add_argument("--include-cluster-details", action="store_true", help="Include detailed per-cluster breakdown section.")
    p_report.set_defaults(func=cmd_report_baselines)

    p_suggest = sub.add_parser("suggest", aliases=["suggest-actions"])
    p_suggest.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_suggest.add_argument("--json", action="store_true", dest="json_output", help="Output machine-readable JSON suggestions.")
    p_suggest.add_argument("--limit", required=False, type=int, default=20, help="Max rows per category (default: 20).")
    p_suggest.add_argument("--topic", required=False, default=None, help="Filter suggestions by topic.")
    p_suggest.add_argument("--only-high", action="store_true", help="Only output HIGH-priority bottlenecks.")
    p_suggest.add_argument("--only-conflicts", action="store_true", help="Only output baseline conflicts.")
    p_suggest.add_argument("--min-severity", required=False, choices=("high", "medium", "low"), default="high", help="Only keep suggestions at or above this severity.")
    p_suggest.add_argument("--sort-by", required=False, choices=("score", "severity"), default="score", help="Sort suggestion items by score or severity.")
    p_suggest.set_defaults(func=cmd_suggest_actions)

    p_set_review = sub.add_parser("set-review-status")
    p_set_review.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_set_review.add_argument("--ids", nargs="+", type=int, default=None, help="Target hypothesis IDs to update.")
    p_set_review.add_argument("--topic", required=False, default=None, help="Target claims by topic.")
    p_set_review.add_argument("--baseline-cluster-id", required=False, type=int, default=None, help="Target claims linked to baseline cluster ID.")
    p_set_review.add_argument(
        "--review-status",
        required=True,
        help="Destination review status (null, auto_unreviewed, auto_high_confidence, needs_review, human_confirmed, human_corrected).",
    )
    p_set_review.add_argument("--dry-run", action="store_true", help="Preview changes without updating DB.")
    p_set_review.add_argument("--limit", required=False, type=int, default=None, help="Limit number of matched records.")
    p_set_review.add_argument("--where-current-status", required=False, default=None, help="Only update records with this current review_status.")
    p_set_review.add_argument("--reason", required=False, default=None, help="Optional human review reason stored in review status history.")
    p_set_review.add_argument(
        "--allow-human-overwrite",
        action="store_true",
        help="Allow overwriting human_confirmed/human_corrected records.",
    )
    p_set_review.set_defaults(func=cmd_set_review_status)

    p_prop = sub.add_parser("proposition-layer")
    p_prop.add_argument("--raw-dir", required=False, default=str(RAW_DIR), help="Raw input directory root.")
    p_prop.add_argument("--enriched-dir", required=False, default=str(ENRICHED_DIR), help="Output directory for enriched JSON.")
    p_prop.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_prop.set_defaults(func=cmd_proposition_layer)

    p_prop_alias = sub.add_parser("build-propositions")
    p_prop_alias.add_argument("--raw-dir", required=False, default=str(RAW_DIR), help="Raw input directory root.")
    p_prop_alias.add_argument("--enriched-dir", required=False, default=str(ENRICHED_DIR), help="Output directory for enriched JSON.")
    p_prop_alias.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_prop_alias.set_defaults(func=cmd_proposition_layer)

    p_triage = sub.add_parser("triage-propositions")
    p_triage.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_triage.add_argument("--enriched-dir", required=False, default=str(ENRICHED_DIR), help="Directory containing proposition-layer integrated outputs.")
    p_triage.add_argument("--review-dir", required=False, default=str(REVIEW_DIR), help="Directory for triage/queue exports.")
    p_triage.add_argument("--triage-run-id", required=False, default=None, help="Optional deterministic run identifier.")
    p_triage.add_argument("--output", required=False, default=None, help="Optional output JSONL path for triage records.")
    p_triage.add_argument("--dry-run", action="store_true", help="Generate triage records without writing DB triage tables.")
    p_triage.set_defaults(func=cmd_triage_propositions)

    p_queue = sub.add_parser("export-review-queue")
    p_queue.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_queue.add_argument("--review-dir", required=False, default=str(REVIEW_DIR), help="Directory for queue export output when --output is omitted.")
    p_queue.add_argument("--output", required=False, default=None, help="Explicit output path (.jsonl/.csv).")
    p_queue.add_argument("--format", required=False, choices=("jsonl", "csv"), default="jsonl", help="Queue export format.")
    p_queue.add_argument("--limit", required=False, type=int, default=None, help="Export at most N queue items.")
    p_queue.add_argument("--include-final", action="store_true", help="Include human_confirmed/human_corrected/human_rejected rows in queue export.")
    p_queue.set_defaults(func=cmd_export_review_queue)

    p_set_prop_review = sub.add_parser("set-proposition-review-status")
    p_set_prop_review.add_argument("--db", required=False, default=None, help="Override synthesis DB path.")
    p_set_prop_review.add_argument("--proposition-ids", nargs="+", default=None, help="Target proposition IDs.")
    p_set_prop_review.add_argument("--triage-record-ids", nargs="+", type=int, default=None, help="Target by triage record IDs.")
    p_set_prop_review.add_argument(
        "--review-status",
        required=True,
        help="Destination review status (pending, triaged, human_review_required, human_confirmed, human_corrected, human_rejected, on_hold).",
    )
    p_set_prop_review.add_argument("--dry-run", action="store_true", help="Preview changes without updating DB.")
    p_set_prop_review.add_argument("--limit", required=False, type=int, default=None, help="Limit number of matched records.")
    p_set_prop_review.add_argument("--where-current-status", required=False, default=None, help="Only update records with this current status.")
    p_set_prop_review.add_argument("--reason", required=False, default=None, help="Optional review reason saved in proposition review history.")
    p_set_prop_review.add_argument("--allow-human-overwrite", action="store_true", help="Allow overwriting locked human statuses.")
    p_set_prop_review.add_argument("--force", action="store_true", help="Bypass transition checks for explicit override.")
    p_set_prop_review.set_defaults(func=cmd_set_proposition_review_status)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (RuntimeError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
