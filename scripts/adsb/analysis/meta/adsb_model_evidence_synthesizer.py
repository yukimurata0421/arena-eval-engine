from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from arena.evidence import build_claim_routes, load_evidence_rows, render_disagreement_report
from arena.evidence.schema import EVIDENCE_CSV_FIELDS
from arena.lib.paths import OUTPUT_DIR as DEFAULT_OUTPUT_ROOT
from arena.log import get_script_logger

log = get_script_logger(__name__)


def _write_matrix(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_evidence_synthesis(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, str]:
    rows = load_evidence_rows(output_root)
    routes = build_claim_routes(rows)

    performance_dir = output_root / "performance"
    matrix_path = performance_dir / "model_evidence_matrix.csv"
    summary_path = performance_dir / "model_evidence_summary.json"
    report_path = performance_dir / "model_disagreement_report.md"

    _write_matrix(matrix_path, rows)
    _write_json(summary_path, routes)
    report_path.write_text(render_disagreement_report(routes), encoding="utf-8")

    log.info("Evidence rows: %d", len(rows))
    log.info("Claim candidates: %d", len(routes.get("claims") or []))
    log.info("Validation targets: %d", len(routes.get("validation_targets") or []))
    log.info("Evidence matrix: %s", matrix_path)
    log.info("Evidence summary: %s", summary_path)
    log.info("Disagreement report: %s", report_path)

    return {
        "matrix": str(matrix_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ARENA model evidence matrix and claim-routing artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="ARENA output root. Defaults to resolved ARENA_OUTPUT_DIR.",
    )
    args = parser.parse_args(argv)
    run_evidence_synthesis(output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
