from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import pandas as pd

# Sibling modules live in the same directory; make importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from daily_signature_builders import (
    build_daily_coverage_signature,
    build_daily_quantile_signature,
    load_daily_auc_series,
)
from daily_signature_sources import build_source_manifest_rows, discover_daily_signature_sources
from signature_change_point_models import detect_change_point_for_columns
from signature_config import (
    DEFAULT_COVERAGE_GRID_KM,
    DEFAULT_QUANTILES,
    DailySignatureConfig,
    TrafficAdjustmentConfig,
    make_timestamped_run_dir,
    parse_float_list,
    parse_int_list,
)
from signature_effect_estimation import build_series_comparison_rows, compare_before_after, split_series_by_change_date
from signature_reports import build_summary_report, generate_figures, write_dataframe, write_json, write_text
from traffic_adjustment import apply_traffic_adjustment, load_daily_traffic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADSB daily signature change point analysis")
    parser.add_argument("--data-root", default="", help="default: ARENA_DATA_DIR or <repo>/data")
    parser.add_argument(
        "--output-root",
        default="",
        help="default: ARENA_OUTPUT_DIR/performance/change_points or <repo>/output/performance/change_points",
    )
    parser.add_argument("--traffic-csv", default="", help="default: <data-root>/flight_data/airport_movements.csv")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--coverage-grid-km", default=",".join(str(v) for v in DEFAULT_COVERAGE_GRID_KM))
    parser.add_argument("--quantiles", default=",".join(str(v) for v in DEFAULT_QUANTILES))
    parser.add_argument("--cp-model", default="rank_scan")
    parser.add_argument("--effect-model", default="hl_mwu")
    parser.add_argument("--min-segment-days", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--phase-config", default="", help="override phase config path (default: scripts/config/phases.txt)")
    parser.add_argument("--settings", default="", help="override settings path (default: scripts/config/settings.toml)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daily_signature_cp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _append_manifest(
    manifest_rows: list[dict[str, Any]],
    *,
    category: str,
    item: str,
    status: str,
    path: str = "",
    reason: str = "",
) -> None:
    manifest_rows.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "item": item,
            "status": status,
            "path": path,
            "reason": reason,
        }
    )


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [df.copy() for df in frames if df is not None and not df.empty and "date" in df.columns]
    if not usable:
        return pd.DataFrame(columns=["date"])
    out = usable[0]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date")
    for frame in usable[1:]:
        next_frame = frame.copy()
        next_frame["date"] = pd.to_datetime(next_frame["date"], errors="coerce").dt.normalize()
        next_frame = next_frame.dropna(subset=["date"]).sort_values("date")
        out = out.merge(next_frame, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _standardize_comparison_series_name(series_name: str) -> str:
    name = series_name.replace("traffic_adjusted_", "")
    if name == "daily_auc.daily_auc":
        return "daily_auc"
    if name.startswith("daily_auc_adjusted."):
        return "daily_auc_adjusted"
    if name.startswith("quantile_signature_adjusted."):
        return "quantile_signature." + name.split(".", 1)[1]
    if name.startswith("coverage_signature_adjusted."):
        return "coverage_signature." + name.split(".", 1)[1]
    return name


def _resolve_config_metadata(
    repo_root: Path,
    *,
    phase_override: str,
    settings_override: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    active_env = env if env is not None else os.environ
    cfg_root = (repo_root / "scripts" / "config").resolve()
    default_phase = (cfg_root / "phases.txt").resolve()
    default_settings = (cfg_root / "settings.toml").resolve()
    phase_raw = str(phase_override or "").strip()
    settings_raw = str(settings_override or "").strip()
    if not phase_raw:
        phase_raw = (
            str(active_env.get("ARENA_PHASE_CONFIG", "")).strip()
            or str(active_env.get("ADSB_PHASE_CONFIG", "")).strip()
        )
    if not settings_raw:
        settings_raw = (
            str(active_env.get("ARENA_SETTINGS", "")).strip()
            or str(active_env.get("ADSB_SETTINGS", "")).strip()
        )
    resolved_phase = Path(phase_raw).expanduser().resolve() if phase_raw else default_phase
    resolved_settings = Path(settings_raw).expanduser().resolve() if settings_raw else default_settings
    used_default_phase = resolved_phase == default_phase
    used_default_settings = resolved_settings == default_settings
    experimental_mode = (
        resolved_phase.name.lower() == "phases_v3_airspy_baseline.txt"
        or resolved_settings.name.lower() == "settings_experimental_distance_bins.toml"
    )
    return {
        "resolved_phase_config_path": str(resolved_phase),
        "resolved_settings_path": str(resolved_settings),
        "used_default_phase_config": bool(used_default_phase),
        "used_default_settings": bool(used_default_settings),
        "experimental_mode": bool(experimental_mode),
        "default_phase_config_path": str(default_phase),
        "default_settings_path": str(default_settings),
    }


def _validate_config_metadata(config_metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    phase_path = Path(str(config_metadata.get("resolved_phase_config_path", "")))
    settings_path = Path(str(config_metadata.get("resolved_settings_path", "")))
    if not phase_path.exists():
        errors.append(
            "phase config missing: {path} (default={default})".format(
                path=phase_path,
                default=config_metadata.get("default_phase_config_path", ""),
            )
        )
    if not settings_path.exists():
        errors.append(
            "settings missing: {path} (default={default})".format(
                path=settings_path,
                default=config_metadata.get("default_settings_path", ""),
            )
        )
    return errors


def main() -> int:
    args = _build_parser().parse_args()
    quantiles = parse_float_list(args.quantiles)
    coverage_grid_km = parse_int_list(args.coverage_grid_km)
    repo_root = Path(__file__).resolve().parents[4]
    default_data_root = Path(os.getenv("ARENA_DATA_DIR", str(repo_root / "data")))
    default_output_root = (
        Path(os.getenv("ARENA_OUTPUT_DIR", str(repo_root / "output"))) / "performance" / "change_points"
    )
    data_root = Path(args.data_root).expanduser().resolve() if str(args.data_root).strip() else default_data_root
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root).strip()
        else default_output_root
    )
    traffic_csv = (
        Path(args.traffic_csv).expanduser().resolve()
        if str(args.traffic_csv).strip()
        else (data_root / "flight_data" / "airport_movements.csv")
    )
    config_metadata = _resolve_config_metadata(
        repo_root=repo_root,
        phase_override=str(args.phase_config or "").strip(),
        settings_override=str(args.settings or "").strip(),
        env=os.environ,
    )
    os.environ["ARENA_PHASE_CONFIG"] = str(config_metadata["resolved_phase_config_path"])
    os.environ["ARENA_SETTINGS"] = str(config_metadata["resolved_settings_path"])
    os.environ["ADSB_PHASE_CONFIG"] = str(config_metadata["resolved_phase_config_path"])
    os.environ["ADSB_SETTINGS"] = str(config_metadata["resolved_settings_path"])

    config = DailySignatureConfig(
        data_root=data_root,
        output_root=output_root,
        traffic_csv=traffic_csv,
        min_days=int(args.min_days),
        quantiles=quantiles,
        coverage_grid_km=coverage_grid_km,
        cp_model=str(args.cp_model),
        effect_model=str(args.effect_model),
        dry_run=bool(args.dry_run),
        min_segment_days=int(args.min_segment_days),
        bootstrap_iterations=int(args.bootstrap_iterations),
        traffic_adjustment=TrafficAdjustmentConfig(),
    )

    run_dir = make_timestamped_run_dir(config.output_root)
    figure_dir = run_dir / "figures"
    log_dir = run_dir / "logs"
    log_path = log_dir / "run.log"
    logger = _setup_logger(log_path)

    run_started_at = datetime.now().isoformat(timespec="seconds")
    manifest_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    assumptions: list[str] = []

    paths = {
        "summary_report": run_dir / "summary_report.md",
        "manifest": run_dir / "run_manifest.csv",
        "run_config": run_dir / "run_config.json",
        "daily_metric": run_dir / "daily_metric_table.csv",
        "cp_csv": run_dir / "change_point_results.csv",
        "cp_json": run_dir / "change_point_results.json",
        "traffic_test": run_dir / "traffic_test_results.csv",
        "unadjusted": run_dir / "traffic_unadjusted_results.csv",
        "adjusted": run_dir / "traffic_adjusted_results.csv",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_config_payload = config.to_dict()
    run_config_payload.update(
        {
            "resolved_phase_config_path": config_metadata["resolved_phase_config_path"],
            "resolved_settings_path": config_metadata["resolved_settings_path"],
            "used_default_phase_config": config_metadata["used_default_phase_config"],
            "used_default_settings": config_metadata["used_default_settings"],
            "experimental_mode": config_metadata["experimental_mode"],
            "default_phase_config_path": config_metadata["default_phase_config_path"],
            "default_settings_path": config_metadata["default_settings_path"],
        }
    )
    write_json(paths["run_config"], run_config_payload)
    _append_manifest(
        manifest_rows,
        category="config",
        item="resolved_phase_config_path",
        status="ok",
        path=str(config_metadata["resolved_phase_config_path"]),
        reason=f"used_default={int(bool(config_metadata['used_default_phase_config']))}",
    )
    _append_manifest(
        manifest_rows,
        category="config",
        item="resolved_settings_path",
        status="ok",
        path=str(config_metadata["resolved_settings_path"]),
        reason=f"used_default={int(bool(config_metadata['used_default_settings']))}",
    )
    _append_manifest(
        manifest_rows,
        category="config",
        item="experimental_mode",
        status="ok",
        path="",
        reason=str(config_metadata["experimental_mode"]),
    )

    selected_input_rows: list[dict[str, Any]] = []
    metric_df = pd.DataFrame(columns=["date"])
    had_fatal_error = False
    cp_df = pd.DataFrame(
        columns=[
            "series_name",
            "series_type",
            "adjustment",
            "detected_change_date",
            "detection_method",
            "score",
            "notes",
            "split_index",
            "n_obs",
        ]
    )
    traffic_test_df = pd.DataFrame(
        columns=[
            "series_name",
            "detected_change_date",
            "n_before",
            "n_after",
            "comparison_method",
            "p_value",
            "effect_size",
            "hl_or_pim_shift",
            "ci_low",
            "ci_high",
            "notes",
        ]
    )
    unadjusted_df = traffic_test_df.copy()
    adjusted_df = traffic_test_df.copy()

    try:
        logger.info("repo_root=%s", repo_root)
        logger.info("run_dir=%s", run_dir)
        logger.info("resolved_phase_config_path=%s", config_metadata["resolved_phase_config_path"])
        logger.info("resolved_settings_path=%s", config_metadata["resolved_settings_path"])
        logger.info(
            "used_default_phase_config=%s used_default_settings=%s experimental_mode=%s",
            config_metadata["used_default_phase_config"],
            config_metadata["used_default_settings"],
            config_metadata["experimental_mode"],
        )

        config_errors = _validate_config_metadata(config_metadata)
        if config_errors:
            for err in config_errors:
                warnings.append(err)
                _append_manifest(manifest_rows, category="error", item="config", status="fatal", reason=err)
            raise FileNotFoundError("; ".join(config_errors))

        sources = discover_daily_signature_sources(
            data_root=config.data_root,
            repo_root=repo_root,
            traffic_csv=config.traffic_csv,
        )
        selected_input_rows = build_source_manifest_rows(sources)
        for row in selected_input_rows:
            _append_manifest(
                manifest_rows,
                category=row.get("category", ""),
                item=row.get("item", ""),
                status=row.get("status", ""),
                path=row.get("path", ""),
                reason=row.get("reason", ""),
            )
        warnings.extend(sources.warnings)

        if config.dry_run:
            _append_manifest(manifest_rows, category="run", item="dry_run", status="ok", reason="analysis_skipped")
        else:
            auc_result = load_daily_auc_series(sources.auc_file) if sources.auc_file else load_daily_auc_series(Path(""))
            quantile_result = build_daily_quantile_signature(
                dist_files=sources.dist_quantile_files,
                quantiles=config.quantiles,
                timezone_name=config.timezone,
            )
            coverage_result = build_daily_coverage_signature(
                signal_files=sources.dist_coverage_files,
                coverage_grid_km=config.coverage_grid_km,
                timezone_name=config.timezone,
            )
            warnings.extend(auc_result.warnings + quantile_result.warnings + coverage_result.warnings)
            assumptions.extend(quantile_result.assumptions + coverage_result.assumptions)

            metric_df = _merge_frames([auc_result.frame, quantile_result.frame, coverage_result.frame])
            if "date" in metric_df.columns:
                metric_df["date"] = pd.to_datetime(metric_df["date"], errors="coerce").dt.normalize()
                metric_df = metric_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            traffic_result = load_daily_traffic(sources.traffic_file) if sources.traffic_file else load_daily_traffic(Path(""))
            warnings.extend(traffic_result.warnings)
            if not traffic_result.frame.empty:
                metric_df = _merge_frames([metric_df, traffic_result.frame])
                metric_df["traffic_source"] = (
                    sources.traffic_file.name if sources.traffic_file is not None else ""
                )

            quantile_cols = [f"q{int(round(q * 100))}" for q in config.quantiles if f"q{int(round(q * 100))}" in metric_df.columns]
            coverage_cols = [
                f"coverage_{int(km)}km"
                for km in config.coverage_grid_km
                if f"coverage_{int(km)}km" in metric_df.columns
            ]
            target_cols = [col for col in (["daily_auc"] + quantile_cols + coverage_cols) if col in metric_df.columns]
            metric_df, adjustment_meta, adjustment_warnings = apply_traffic_adjustment(
                metric_df,
                target_columns=target_cols,
                min_points=config.traffic_adjustment.min_points,
                clip_min_traffic=config.traffic_adjustment.clip_min_traffic,
            )
            warnings.extend(adjustment_warnings)
            for meta in adjustment_meta:
                _append_manifest(
                    manifest_rows,
                    category="traffic_adjustment",
                    item=meta.get("column", ""),
                    status=meta.get("status", ""),
                    reason=str(meta),
                )

            if "traffic_adjusted_daily_auc" in metric_df.columns:
                metric_df["traffic_adjusted_auc"] = metric_df["traffic_adjusted_daily_auc"]

            cp_records: list[dict[str, Any]] = []
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=["daily_auc"],
                    series_name="daily_auc",
                    series_type="scalar_auc",
                    adjustment="unadjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=["traffic_adjusted_auc"],
                    series_name="daily_auc",
                    series_type="scalar_auc",
                    adjustment="traffic_adjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=quantile_cols,
                    series_name="quantile_signature",
                    series_type="quantile_signature",
                    adjustment="unadjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=[f"traffic_adjusted_{c}" for c in quantile_cols],
                    series_name="quantile_signature",
                    series_type="quantile_signature",
                    adjustment="traffic_adjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=coverage_cols,
                    series_name="coverage_signature",
                    series_type="coverage_signature",
                    adjustment="unadjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )
            cp_records.append(
                detect_change_point_for_columns(
                    metric_df=metric_df,
                    columns=[f"traffic_adjusted_{c}" for c in coverage_cols],
                    series_name="coverage_signature",
                    series_type="coverage_signature",
                    adjustment="traffic_adjusted",
                    cp_model=config.cp_model,
                    min_segment_days=config.min_segment_days,
                    min_days=config.min_days,
                ).to_record()
            )

            traffic_cp = detect_change_point_for_columns(
                metric_df=metric_df,
                columns=["traffic_count"],
                series_name="traffic_count",
                series_type="traffic",
                adjustment="unadjusted",
                cp_model=config.cp_model,
                min_segment_days=config.min_segment_days,
                min_days=config.min_days,
            ).to_record()
            cp_records.append(traffic_cp)
            cp_df = pd.DataFrame(cp_records)

            def _cp_date(series_name: str, adjustment: str) -> str:
                hit = cp_df[
                    (cp_df["series_name"] == series_name) & (cp_df["adjustment"] == adjustment)
                ]
                if hit.empty:
                    return ""
                return str(hit.iloc[0].get("detected_change_date", ""))

            # traffic tests
            traffic_change_date = _cp_date("traffic_count", "unadjusted")
            before_t, after_t, split_status_t = split_series_by_change_date(
                metric_df, value_column="traffic_count", change_date=traffic_change_date
            )
            traffic_stats = compare_before_after(
                before=before_t,
                after=after_t,
                n_bootstrap=config.bootstrap_iterations,
                random_seed=config.random_seed,
            )
            traffic_test_df = pd.DataFrame(
                [
                    {
                        "series_name": "traffic_count",
                        "detected_change_date": traffic_change_date,
                        "n_before": traffic_stats["n_before"],
                        "n_after": traffic_stats["n_after"],
                        "comparison_method": traffic_stats["comparison_method"],
                        "p_value": traffic_stats["p_value"],
                        "effect_size": traffic_stats["effect_size"],
                        "hl_or_pim_shift": traffic_stats["hl_or_pim_shift"],
                        "ci_low": traffic_stats["ci_low"],
                        "ci_high": traffic_stats["ci_high"],
                        "notes": f"{traffic_stats['notes']};split={split_status_t}",
                    }
                ]
            )

            # unadjusted comparisons
            unadjusted_rows: list[dict[str, Any]] = []
            unadjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("daily_auc", "unadjusted"),
                    columns=["daily_auc"],
                    series_prefix="daily_auc",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            unadjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("quantile_signature", "unadjusted"),
                    columns=quantile_cols,
                    series_prefix="quantile_signature",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            unadjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("coverage_signature", "unadjusted"),
                    columns=coverage_cols,
                    series_prefix="coverage_signature",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            unadjusted_df = pd.DataFrame(unadjusted_rows)
            if not unadjusted_df.empty:
                unadjusted_df["series_name"] = unadjusted_df["series_name"].apply(
                    _standardize_comparison_series_name
                )

            # adjusted comparisons
            adjusted_rows: list[dict[str, Any]] = []
            adjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("daily_auc", "traffic_adjusted"),
                    columns=["traffic_adjusted_auc"],
                    series_prefix="daily_auc_adjusted",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            adjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("quantile_signature", "traffic_adjusted"),
                    columns=[f"traffic_adjusted_{c}" for c in quantile_cols],
                    series_prefix="quantile_signature_adjusted",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            adjusted_rows.extend(
                build_series_comparison_rows(
                    metric_df=metric_df,
                    change_date=_cp_date("coverage_signature", "traffic_adjusted"),
                    columns=[f"traffic_adjusted_{c}" for c in coverage_cols],
                    series_prefix="coverage_signature_adjusted",
                    n_bootstrap=config.bootstrap_iterations,
                    random_seed=config.random_seed,
                )
            )
            adjusted_df = pd.DataFrame(adjusted_rows)
            if not adjusted_df.empty:
                adjusted_df["series_name"] = adjusted_df["series_name"].apply(_standardize_comparison_series_name)

            figure_paths = generate_figures(
                figure_dir=figure_dir,
                metric_df=metric_df,
                change_point_df=cp_df,
                quantile_cols=quantile_cols,
                coverage_cols=coverage_cols,
            )
            for fp in figure_paths:
                _append_manifest(manifest_rows, category="figure", item="plot", status="ok", path=fp, reason="")

    except Exception as exc:
        had_fatal_error = True
        error_text = traceback.format_exc()
        warnings.append(f"fatal_error:{exc}")
        _append_manifest(
            manifest_rows,
            category="error",
            item="exception",
            status="fatal",
            reason=str(exc),
        )
        logger.error("fatal_error=%s", exc)
        logger.error(error_text)

    # Persist outputs even on failure.
    write_dataframe(paths["daily_metric"], metric_df)
    write_dataframe(paths["cp_csv"], cp_df)
    write_json(paths["cp_json"], cp_df.to_dict(orient="records"))
    write_dataframe(paths["traffic_test"], traffic_test_df)
    write_dataframe(paths["unadjusted"], unadjusted_df)
    write_dataframe(paths["adjusted"], adjusted_df)

    run_completed_at = datetime.now().isoformat(timespec="seconds")
    summary_text = build_summary_report(
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        selected_input_rows=selected_input_rows,
        change_point_df=cp_df,
        traffic_test_df=traffic_test_df,
        unadjusted_df=unadjusted_df,
        adjusted_df=adjusted_df,
        assumptions=assumptions,
        warnings=warnings,
        cp_model=config.cp_model,
        effect_model=config.effect_model,
        traffic_adjustment_method=config.traffic_adjustment.method,
        config_metadata=config_metadata,
    )
    write_text(paths["summary_report"], summary_text)

    if warnings:
        for warning in warnings:
            _append_manifest(manifest_rows, category="warning", item="run", status="warn", reason=warning)
    _append_manifest(manifest_rows, category="run", item="completed_at", status="ok", reason=run_completed_at)
    write_dataframe(paths["manifest"], pd.DataFrame(manifest_rows))

    logger.info("output_dir=%s", run_dir)
    logger.info("summary_report=%s", paths["summary_report"])
    return 1 if had_fatal_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
