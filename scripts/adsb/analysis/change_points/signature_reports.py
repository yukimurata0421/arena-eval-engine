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
        return ["- 変化点は検出されませんでした。"]
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
        return ["- 統計比較結果は空です。"]
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
    lines.append("## 実行日時")
    lines.append(f"- 開始: {run_started_at}")
    lines.append(f"- 完了: {run_completed_at}")
    lines.append("")
    lines.append("## 入力採用ファイル一覧")
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
        "- production default policy: override 未指定時は scripts/config/phases.txt と "
        "scripts/config/settings.toml を使用し、既定値を自動置換しない。"
    )
    lines.append("")
    lines.append("## 既存 change_points 配下に追加した理由")
    lines.append(
        "- 既存の変化点分析責務を維持し、stage 9 連携時に import 単位で段階統合できるように、"
        "scripts/adsb/analysis/change_points 配下へ append-only で拡張した。"
    )
    lines.append("")
    lines.append("## 採用した change point 手法と理由")
    lines.append(f"- 手法: {cp_model}")
    lines.append(
        "- rank_scan を基準に採用。非正規/外れ値に比較的頑健で、依存が軽く、将来の複数変化点拡張でも再利用しやすい。"
    )
    lines.extend(summarize_change_points(change_point_df))
    lines.append("")
    lines.append("## HL か PIM の選定理由")
    lines.append(f"- effect_model: {effect_model}")
    lines.append(
        "- 初期実装は MWU + Hodges-Lehmann を採用。実装容易性・再現性・解釈性を優先し、"
        "追加依存のない構成で stage 9 へ移植しやすいことを重視した。"
    )
    lines.append("")
    lines.append("## traffic 補正方法と理由")
    lines.append(f"- 補正方法: {traffic_adjustment_method}")
    lines.append(
        "- traffic_count の log1p を共変量として線形残差化。説明可能で再計算が容易、"
        "壊れ方（不足データ時は補正スキップ）が明確なため採用した。"
    )
    lines.append("")
    lines.append("## 補正なし / 補正あり の比較要約")
    lines.append("- 補正なし:")
    lines.extend(summarize_comparison_rows(unadjusted_df))
    lines.append("- 補正あり:")
    lines.extend(summarize_comparison_rows(adjusted_df))
    lines.append("")
    lines.append("## traffic 系列の検定")
    if traffic_test_df.empty:
        lines.append("- traffic_test_results は空。")
    else:
        lines.extend(summarize_comparison_rows(traffic_test_df, top_n=3))
    lines.append("")
    lines.append("## 何が言えるか")
    lines.append(
        "- 日次AUC・quantile signature・coverage signature の3系列で、同一実行から"
        "変化点候補日と前後比較統計を再計算可能な形で出力できる。"
    )
    lines.append("")
    lines.append("## 何がまだ言えないか")
    lines.append(
        "- 因果解釈（機材変更や運用変更が主因かどうか）はこの分析だけでは断定できない。"
    )
    lines.append("")
    lines.append("## 限界")
    if assumptions:
        for assumption in assumptions:
            lines.append(f"- 仮定: {assumption}")
    lines.append("- coverage は距離バケット内一様分布を仮定した近似を含む。")
    lines.append("- q99 は一部区間で補間値を使用する。")
    if warnings:
        for warning in warnings:
            lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("## stage 9 に統合する際の接続ポイント")
    lines.append("- データ探索: daily_signature_sources.discover_daily_signature_sources")
    lines.append("- 系列生成: daily_signature_builders.*")
    lines.append("- traffic補正: traffic_adjustment.apply_traffic_adjustment")
    lines.append("- 変化点: signature_change_point_models.detect_change_point_for_columns")
    lines.append("- 効果推定: signature_effect_estimation.build_series_comparison_rows")
    lines.append("- 出力: signature_reports.*")
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
