from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DailySignatureSources:
    auc_file: Path | None
    auc_reason: str
    auc_candidates: list[dict[str, Any]] = field(default_factory=list)
    dist_quantile_files: list[Path] = field(default_factory=list)
    dist_quantile_reason: str = ""
    dist_coverage_files: list[Path] = field(default_factory=list)
    dist_coverage_reason: str = ""
    traffic_file: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _rank_auc_candidate(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if name == "adsb_daily_summary_v2.csv":
        score += 300
    elif name == "adsb_daily_summary.csv":
        score += 250
    elif name == "adsb_daily_summary_raw.csv":
        score += 200
    elif "adsb_daily_summary" in name:
        score += 150
    if str(path).lower().endswith(r"\output\adsb_daily_summary_v2.csv"):
        score += 50
    lowered = str(path).lower()
    if "merged_for_ai" in lowered or "_tmp_" in lowered:
        score -= 40
    return score


def _inspect_auc_candidate(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "has_date": False,
        "has_auc": False,
        "rows": 0,
        "score": _rank_auc_candidate(path),
        "reason": "",
    }
    if not path.exists():
        row["reason"] = "file_not_found"
        return row
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        row["reason"] = f"read_failed:{exc}"
        return row

    row["rows"] = int(len(df))
    row["has_date"] = "date" in df.columns
    row["has_auc"] = "auc_n_used" in df.columns
    if row["has_date"] and row["has_auc"]:
        row["score"] += 100
        non_null_auc = int(pd.to_numeric(df["auc_n_used"], errors="coerce").notna().sum())
        row["score"] += min(non_null_auc, 100)
        if "minutes_covered" in df.columns:
            row["score"] += 25
        row["reason"] = "usable"
    else:
        row["reason"] = "missing_required_columns"
    return row


def _discover_auc_candidates(repo_root: Path) -> list[Path]:
    output_root = repo_root / "output"
    direct = [
        output_root / "adsb_daily_summary_v2.csv",
        output_root / "adsb_daily_summary.csv",
        output_root / "adsb_daily_summary_raw.csv",
    ]
    found: dict[str, Path] = {str(p): p for p in direct}
    if output_root.exists():
        for p in output_root.rglob("adsb_daily_summary*.csv"):
            found[str(p)] = p
    return list(found.values())


def discover_daily_signature_sources(
    data_root: Path,
    repo_root: Path,
    traffic_csv: Path | None = None,
) -> DailySignatureSources:
    warnings: list[str] = []

    auc_candidates = _discover_auc_candidates(repo_root=repo_root)
    inspected = [_inspect_auc_candidate(p) for p in auc_candidates]
    inspected.sort(key=lambda x: x.get("score", 0), reverse=True)

    auc_file: Path | None = None
    auc_reason = "no_usable_auc_file"
    for row in inspected:
        if row.get("reason") == "usable":
            auc_file = Path(row["path"])
            auc_reason = (
                f"selected_highest_score={row.get('score')} rows={row.get('rows')} "
                f"has_minutes_covered={'minutes_covered' in pd.read_csv(auc_file, nrows=1).columns}"
            )
            break
    if auc_file is None:
        warnings.append("auc_source_missing_or_unusable")

    quantile_files: list[Path] = []
    dist_current = data_root / "dist_1m.jsonl"
    if dist_current.exists():
        quantile_files.append(dist_current)
    archive_root = data_root / "raw" / "past_log"
    if archive_root.exists():
        quantile_files.extend(sorted(archive_root.glob("*dist*.jsonl.till-*")))

    quantile_reason = (
        "dist_1m current + raw/past_log/*dist*.jsonl.till-*."
        "Construct daily quantile by adopting only src=dist_1m in record."
    )
    if not quantile_files:
        warnings.append("quantile_raw_distance_files_not_found")

    coverage_files: list[Path] = []
    coverage_current = data_root / "dist_signal_stats_1m.jsonl"
    if coverage_current.exists():
        coverage_files.append(coverage_current)
    if archive_root.exists():
        coverage_files.extend(sorted(archive_root.glob("dist_signal_stats_1m*.jsonl.till-*")))
    coverage_reason = (
        "dist_signal_stats_1m target current + archive."
        "Daily aggregation of coverage grid from distance bucket n_samples."
    )
    if not coverage_files:
        warnings.append("coverage_raw_distance_files_not_found")

    chosen_traffic = traffic_csv if traffic_csv is not None else data_root / "flight_data" / "airport_movements.csv"
    if not chosen_traffic.exists():
        warnings.append(f"traffic_csv_not_found:{chosen_traffic}")

    return DailySignatureSources(
        auc_file=auc_file,
        auc_reason=auc_reason,
        auc_candidates=inspected,
        dist_quantile_files=quantile_files,
        dist_quantile_reason=quantile_reason,
        dist_coverage_files=coverage_files,
        dist_coverage_reason=coverage_reason,
        traffic_file=chosen_traffic if chosen_traffic.exists() else None,
        warnings=warnings,
    )


def build_source_manifest_rows(sources: DailySignatureSources) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "category": "source_selection",
            "item": "auc_selected",
            "status": "ok" if sources.auc_file else "missing",
            "path": str(sources.auc_file) if sources.auc_file else "",
            "reason": sources.auc_reason,
        }
    )
    for candidate in sources.auc_candidates:
        rows.append(
            {
                "category": "source_candidate",
                "item": "auc_candidate",
                "status": candidate.get("reason", ""),
                "path": candidate.get("path", ""),
                "reason": (
                    f"score={candidate.get('score')} rows={candidate.get('rows')} "
                    f"has_date={candidate.get('has_date')} has_auc={candidate.get('has_auc')}"
                ),
            }
        )

    rows.append(
        {
            "category": "source_selection",
            "item": "quantile_raw_files",
            "status": "ok" if sources.dist_quantile_files else "missing",
            "path": "|".join(str(p) for p in sources.dist_quantile_files),
            "reason": sources.dist_quantile_reason,
        }
    )
    rows.append(
        {
            "category": "source_selection",
            "item": "coverage_raw_files",
            "status": "ok" if sources.dist_coverage_files else "missing",
            "path": "|".join(str(p) for p in sources.dist_coverage_files),
            "reason": sources.dist_coverage_reason,
        }
    )
    rows.append(
        {
            "category": "source_selection",
            "item": "traffic_csv",
            "status": "ok" if sources.traffic_file else "missing",
            "path": str(sources.traffic_file) if sources.traffic_file else "",
            "reason": "airport movements daily traffic",
        }
    )
    for warning in sources.warnings:
        rows.append(
            {
                "category": "warning",
                "item": "source",
                "status": "warn",
                "path": "",
                "reason": warning,
            }
        )
    return rows
