from __future__ import annotations

from arena.synthesis.proposition_review_shared import (
    PropositionReviewStatusUpdateReport,
    ReviewQueueExportReport,
    TriagePropositionsReport,
)


def render_triage_result(report: TriagePropositionsReport) -> str:
    parts = [
        f"triage_run_id={report.triage_run_id}",
        f"scanned_propositions={report.scanned_propositions}",
        f"triage_records_written={report.triage_records_written}",
        f"queue_candidates={report.queue_candidates}",
        f"output_path={report.output_path}",
        f"dry_run={int(report.dry_run)}",
    ]
    return " ".join(parts)


def render_queue_export_result(report: ReviewQueueExportReport) -> str:
    parts = [
        f"exported_rows={report.exported_rows}",
        f"format={report.format}",
        f"include_final={int(report.include_final)}",
        f"output_path={report.output_path}",
    ]
    return " ".join(parts)


def render_proposition_review_update_result(report: PropositionReviewStatusUpdateReport) -> str:
    reason_text = report.reason_used if report.reason_used is not None else "-"
    parts = [
        f"requested_status={report.requested_status}",
        f"matched_records={report.matched_records}",
        f"updated_records={report.updated_records}",
        f"unchanged_records={report.unchanged_records}",
        f"skipped_locked_records={report.skipped_locked_records}",
        f"skipped_missing_records={report.skipped_missing_records}",
        f"skipped_transition_records={report.skipped_transition_records}",
        f"history_records_written={report.history_records_written}",
        f"reason_used={reason_text}",
        f"dry_run={int(report.dry_run)}",
        f"updated_ids={','.join(report.updated_ids) if report.updated_ids else '-'}",
        f"skipped_locked_ids={','.join(report.skipped_locked_ids) if report.skipped_locked_ids else '-'}",
        f"skipped_missing_ids={','.join(report.skipped_missing_ids) if report.skipped_missing_ids else '-'}",
        f"skipped_transition_ids={','.join(report.skipped_transition_ids) if report.skipped_transition_ids else '-'}",
    ]
    return " ".join(parts)
