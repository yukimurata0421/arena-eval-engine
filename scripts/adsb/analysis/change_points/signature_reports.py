from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_change_points(change_point_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if change_point_df.empty:
        return ["- No change points detected."]
    for _, row in change_point_df.iterrows():
        lines.append(
            "- {series_name} ({adjustment}) -> {date} [{method}] score={score}".format(
                series_name=row.get("series_name", ""),
                adjustment=row.get("adjustment", ""),
                date=row.get("detected_change_date", ""),
                method=row.get("detection_method", ""),
                score=row.get("score", ""),
            )
        )
    return lines


def summarize_comparison_rows(df: pd.DataFrame, top_n: int = 6) -> list[str]:
    if df.empty:
        return ["- Statistical comparison result is empty."]
    ordered = df.sort_values("p_value", na_position="last").head(top_n)
    lines = []
    for _, row in ordered.iterrows():
        lines.append(
            "- {series}: p={p:.4g}, effect={eff:.4g}, shift={shift:.4g}, CI=[{lo:.4g}, {hi:.4g}]".format(
                series=row.get("series_name", ""),
                p=float(row.get("p_value")) if pd.notna(row.get("p_value")) else float("nan"),
                eff=float(row.get("effect_size")) if pd.notna(row.get("effect_size")) else float("nan"),
                shift=float(row.get("hl_or_pim_shift")) if pd.notna(row.get("hl_or_pim_shift")) else float("nan"),
                lo=float(row.get("ci_low")) if pd.notna(row.get("ci_low")) else float("nan"),
                hi=float(row.get("ci_high")) if pd.notna(row.get("ci_high")) else float("nan"),
            )
        )
    return lines


def build_summary_report(
    *,
    run_started_at: str,
    run_completed_at: str,
    selected_input_rows: list[dict[str, Any]],
    change_point_df: pd.DataFrame,
    traffic_test_df: pd.DataFrame,
    unadjusted_df: pd.DataFrame,
    adjusted_df: pd.DataFrame,
    assumptions: list[str],
    warnings: list[str],
    cp_model: str,
    effect_model: str,
    traffic_adjustment_method: str,
    config_metadata: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Daily Signature Change Point Summary")
    lines.append("")
    lines.append("## Execution date and time")
    lines.append(f"- Started: {run_started_at}")
    lines.append(f"- Completed: {run_completed_at}")
    lines.append("")
    lines.append("## Input adopted file list")
    if selected_input_rows:
        for row in selected_input_rows:
            lines.append(
                f"- {row.get('item','')}: {row.get('path','')} ({row.get('status','')}) / {row.get('reason','')}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Config Resolution")
    lines.append(f"- resolved_phase_config_path: {config_metadata.get('resolved_phase_config_path', '')}")
    lines.append(f"- resolved_settings_path: {config_metadata.get('resolved_settings_path', '')}")
    lines.append(f"- used_default_phase_config: {config_metadata.get('used_default_phase_config', '')}")
    lines.append(f"- used_default_settings: {config_metadata.get('used_default_settings', '')}")
    lines.append(f"- experimental_mode: {config_metadata.get('experimental_mode', '')}")
    lines.append(
        "- production default policy: If override is not specified, scripts/config/phases.txt and "
        "Use scripts/config/settings.toml and do not auto-replace default values."
    )
    lines.append("")
    lines.append("## Reason for adding under existing change_points")
    lines.append(
        "- Maintaining the existing change point analysis responsibility and allowing stage integration by import unit when linking to stage 9."
        "Extended under scripts/adsb/analysis/change_points with append-only."
    )
    lines.append("")
    lines.append("## Change point method adopted and reason")
    lines.append(f"- method: {cp_model}")
    lines.append(
        "- Adopted based on rank_scan. Relatively robust to non-normal/outlier values, light dependence, and easy to reuse in future multi-change point expansion."
    )
    lines.extend(summarize_change_points(change_point_df))
    lines.append("")
    lines.append("## Reason for choosing HL or PIM")
    lines.append(f"- effect_model: {effect_model}")
    lines.append(
        "- The initial implementation uses MWU + Hodges-Lehmann. Prioritizes ease of implementation, reproducibility, and interpretability."
        "We focused on making it easy to port to stage 9 with a configuration that does not require additional dependencies."
    )
    lines.append("")
    lines.append("## traffic correction method and reason")
    lines.append(f"- Correction method: {traffic_adjustment_method}")
    lines.append(
        "- Linear residualization of log1p of traffic_count as a covariate. Explainable and easy to recalculate."
        "We adopted this method because it is clear how it breaks (correction is skipped if there is insufficient data)."
    )
    lines.append("")
    lines.append("## Comparison summary of without correction / with correction")
    lines.append("- No correction:")
    lines.extend(summarize_comparison_rows(unadjusted_df))
    lines.append("- With correction:")
    lines.extend(summarize_comparison_rows(adjusted_df))
    lines.append("")
    lines.append("## traffic series verification")
    if traffic_test_df.empty:
        lines.append("- traffic_test_results is empty.")
    else:
        lines.extend(summarize_comparison_rows(traffic_test_df, top_n=3))
    lines.append("")
    lines.append("## What can you say")
    lines.append(
        "- Daily AUC, quantile signature, and coverage signature from the same run"
        "Changing point candidate dates and before/after comparison statistics can be output in a format that can be recalculated."
    )
    lines.append("")
    lines.append("## What can't be said yet")
    lines.append(
        "- The causal interpretation (whether equipment changes or operational changes were the main cause) cannot be determined from this analysis alone."
    )
    lines.append("")
    lines.append("## limit")
    if assumptions:
        for assumption in assumptions:
            lines.append(f"- {assumption}")
    lines.append("- coverage includes an approximation assuming uniform distribution within the distance buckets.")
    lines.append("- q99 uses interpolated values ​​in some sections.")
    if warnings:
        for warning in warnings:
            lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("## Connection points when integrating into stage 9")
    lines.append("- Data exploration: daily_signature_sources.discover_daily_signature_sources")
    lines.append("- Line generation: daily_signature_builders.*")
    lines.append("- traffic correction: traffic_adjustment.apply_traffic_adjustment")
    lines.append("- Change point: signature_change_point_models.detect_change_point_for_columns")
    lines.append("- Effect estimation: signature_effect_estimation.build_series_comparison_rows")
    lines.append("- Output: signature_reports.*")
    lines.append("")
    return "\n".join(lines) + "\n"


def generate_figures(
    *,
    figure_dir: Path,
    metric_df: pd.DataFrame,
    change_point_df: pd.DataFrame,
    quantile_cols: list[str],
    coverage_cols: list[str],
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if metric_df.empty:
        return written

    # 1) AUC
    auc_cols = [c for c in ["daily_auc", "traffic_adjusted_auc"] if c in metric_df.columns]
    if auc_cols:
        plt.figure(figsize=(12, 4))
        for col in auc_cols:
            plt.plot(metric_df["date"], metric_df[col], label=col, linewidth=1.5)
        for _, row in change_point_df[change_point_df["series_name"] == "daily_auc"].iterrows():
            if row.get("detected_change_date"):
                plt.axvline(pd.to_datetime(row["detected_change_date"]), color="red", linestyle="--", alpha=0.6)
        plt.title("Daily AUC with Change Points")
        plt.legend()
        plt.tight_layout()
        out = figure_dir / "daily_auc_change_points.png"
        plt.savefig(out, dpi=130)
        plt.close()
        written.append(str(out))

    # 2) Quantiles
    q_cols = [c for c in quantile_cols if c in metric_df.columns]
    if q_cols:
        plt.figure(figsize=(12, 5))
        for col in q_cols:
            plt.plot(metric_df["date"], metric_df[col], label=col, linewidth=1.1)
        plt.title("Daily Quantile Signature")
        plt.legend(ncol=3)
        plt.tight_layout()
        out = figure_dir / "daily_quantile_signature.png"
        plt.savefig(out, dpi=130)
        plt.close()
        written.append(str(out))

    # 3) Coverage
    c_cols = [c for c in coverage_cols if c in metric_df.columns]
    if c_cols:
        plt.figure(figsize=(12, 5))
        for col in c_cols:
            plt.plot(metric_df["date"], metric_df[col], label=col, linewidth=1.0)
        plt.title("Daily Coverage Signature")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        out = figure_dir / "daily_coverage_signature.png"
        plt.savefig(out, dpi=130)
        plt.close()
        written.append(str(out))

    # 4) Traffic
    if "traffic_count" in metric_df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(metric_df["date"], metric_df["traffic_count"], label="traffic_count", color="tab:green")
        traffic_rows = change_point_df[change_point_df["series_name"] == "traffic_count"]
        for _, row in traffic_rows.iterrows():
            if row.get("detected_change_date"):
                plt.axvline(pd.to_datetime(row["detected_change_date"]), color="red", linestyle="--", alpha=0.6)
        plt.title("Traffic Series with Change Point")
        plt.legend()
        plt.tight_layout()
        out = figure_dir / "traffic_change_point.png"
        plt.savefig(out, dpi=130)
        plt.close()
        written.append(str(out))

    return written
