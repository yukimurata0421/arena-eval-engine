#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
OpenSky vs Local ADS-B Comparison Evaluator v2.

Sub-modules (same directory):
  _common.py        - shared constants and utility functions
  _data_loading.py  - OpenSky / local data loading and bin estimation
  _statistics.py    - statistical tests
  _plotting.py      - matplotlib plot generation
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure sibling modules are importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from _common import (
    DEFAULT_CR_CAP,
    DEFAULT_LOCAL_MIN_MINUTES_MATCH,
    DEFAULT_LOCAL_UTC_OFFSET_HOURS,
    DEFAULT_OS_MAX_KM_MAX,
    DEFAULT_OS_MIN_MINUTES_PER_DAY,
    DEFAULT_OS_MIN_N_USED,
    DISTANCE_BIN_LABELS,
    SITE_LAT,
    SITE_LON,
    date_str_to_iso,
    normalize_path,
    resolve_phase_detailed,
    resolve_phase_for_date,
)
from _data_loading import (
    OpenSkyMinute,
    estimate_opensky_bin_counts,
    get_existing_local_dates,
    load_local_data,
    load_opensky_data,
)
from _plotting import (
    plot_capture_by_distance_bin,
    plot_capture_ratio_by_phase,
    plot_daily_bin_trend_compact,
    plot_daily_trend_compact,
    plot_distance_heatmap,
)
from _stats import run_statistics

from arena.lib.paths import DATA_DIR, OUTPUT_DIR, ensure_dir
from arena.lib.phase_config import get_config

from arena.log import get_script_logger



log = get_script_logger(__name__)
# ════════════════════════════════════════════════
# Merge
# ════════════════════════════════════════════════

@dataclass
class MergedMinute:
    minute_key: int
    date_str: str
    date_iso: str
    phase: str
    phase_detail: str
    os_n_total: int
    os_n_used: int
    os_km_max: float
    os_km_avg: float
    os_bin_counts: Dict[str, float]
    local_n_unique: int
    local_km_max: float
    local_bin_counts: Dict[str, int]
    capture_ratio: float
    capture_by_bin: Dict[str, float]


def merge_data(
    opensky_data: List[OpenSkyMinute],
    local_data,
    cfg,
    cr_cap: float = DEFAULT_CR_CAP,
) -> List[MergedMinute]:
    results: List[MergedMinute] = []
    for osm in opensky_data:
        mk = osm.minute_key
        lm = local_data.get(mk)

        date_iso = date_str_to_iso(osm.date_str)
        phase = resolve_phase_for_date(date_iso, cfg)
        phase_detail = resolve_phase_detailed(date_iso, cfg)
        os_bins = estimate_opensky_bin_counts(osm)

        if lm is None:
            local_n = 0
            local_km_max = 0.0
            local_bins: Dict[str, int] = {lab: 0 for lab in DISTANCE_BIN_LABELS}
        else:
            local_n = lm.n_unique
            local_km_max = lm.km_max
            local_bins = lm.bin_counts

        os_count = osm.n_used if osm.n_used > 0 else osm.n_total
        cr = local_n / os_count if os_count > 0 else np.nan
        if not np.isnan(cr) and cr > cr_cap:
            cr = cr_cap

        capture_by_bin: Dict[str, float] = {}
        for lab in DISTANCE_BIN_LABELS:
            os_b = os_bins.get(lab, 0.0)
            lc_b = float(local_bins.get(lab, 0))
            cb = lc_b / os_b if os_b > 0.5 else np.nan
            if not np.isnan(cb) and cb > cr_cap:
                cb = cr_cap
            capture_by_bin[lab] = cb

        results.append(MergedMinute(
            minute_key=mk, date_str=osm.date_str, date_iso=date_iso,
            phase=phase, phase_detail=phase_detail,
            os_n_total=osm.n_total, os_n_used=osm.n_used,
            os_km_max=osm.km_max, os_km_avg=osm.km_avg,
            os_bin_counts=os_bins,
            local_n_unique=local_n, local_km_max=local_km_max,
            local_bin_counts=local_bins,
            capture_ratio=cr, capture_by_bin=capture_by_bin,
        ))
    return results


def merged_to_dataframe(merged: List[MergedMinute]) -> pd.DataFrame:
    rows = []
    for mm in merged:
        row = {
            "minute_key": mm.minute_key, "date": mm.date_str,
            "date_iso": mm.date_iso, "phase": mm.phase,
            "phase_detail": mm.phase_detail,
            "os_n_total": mm.os_n_total, "os_n_used": mm.os_n_used,
            "os_km_max": mm.os_km_max, "os_km_avg": mm.os_km_avg,
            "local_n_unique": mm.local_n_unique,
            "local_km_max": mm.local_km_max,
            "capture_ratio": mm.capture_ratio,
        }
        for lab in DISTANCE_BIN_LABELS:
            row[f"os_bin_{lab}"] = mm.os_bin_counts.get(lab, 0.0)
            row[f"local_bin_{lab}"] = mm.local_bin_counts.get(lab, 0)
            row[f"capture_bin_{lab}"] = mm.capture_by_bin.get(lab, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def make_daily_summary(
    df: pd.DataFrame,
    local_dates_exist: set,
    os_min_minutes: int = DEFAULT_OS_MIN_MINUTES_PER_DAY,
    local_min_minutes: int = DEFAULT_LOCAL_MIN_MINUTES_MATCH,
) -> pd.DataFrame:
    agg_dict: dict = {
        "minute_key": "count", "os_n_used": "mean",
        "local_n_unique": "mean", "capture_ratio": "median",
        "os_km_max": "max", "local_km_max": "max",
    }
    rename_dict: dict = {
        "minute_key": "n_minutes", "os_n_used": "os_mean_n_used",
        "local_n_unique": "local_mean_n_unique",
        "capture_ratio": "median_capture_ratio",
        "os_km_max": "os_max_km", "local_km_max": "local_max_km",
    }
    for lab in DISTANCE_BIN_LABELS:
        agg_dict[f"capture_bin_{lab}"] = "median"
        rename_dict[f"capture_bin_{lab}"] = f"median_capture_{lab}"

    daily = df.groupby(["date", "date_iso", "phase", "phase_detail"]).agg(agg_dict)
    daily = daily.rename(columns=rename_dict).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    local_nonzero = df[df["local_n_unique"] > 0].groupby("date").size().reset_index(name="local_nonzero_minutes")
    daily = daily.merge(local_nonzero, on="date", how="left")
    daily["local_nonzero_minutes"] = daily["local_nonzero_minutes"].fillna(0).astype(int)

    has_pos_file = daily["date"].isin(local_dates_exist)
    has_enough_os = daily["n_minutes"] >= os_min_minutes
    has_enough_local = daily["local_nonzero_minutes"] >= local_min_minutes

    daily["pos_file_exists"] = has_pos_file
    daily["use_for_stats"] = has_pos_file & has_enough_os & has_enough_local

    reasons = []
    for _, r in daily.iterrows():
        if r["use_for_stats"]:
            reasons.append("")
        elif not r["pos_file_exists"]:
            reasons.append("no_pos_file")
        elif r["n_minutes"] < os_min_minutes:
            reasons.append(f"os_minutes<{os_min_minutes}")
        elif r["local_nonzero_minutes"] < local_min_minutes:
            reasons.append(f"local_nonzero<{local_min_minutes}")
        else:
            reasons.append("unknown")
    daily["skip_reason"] = reasons
    return daily


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="OpenSky vs Local ADS-B Comparison Evaluator v2")
    ap.add_argument("--opensky-dir", default=str(DATA_DIR))
    ap.add_argument("--opensky-pattern", default="dist_1m.jsonl")
    ap.add_argument("--local-dir", default=str(DATA_DIR / "plao_pos"))
    ap.add_argument("--local-pattern", default="pos_*.jsonl")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--out-subdir", default="opensky_comparison")
    ap.add_argument("--site-lat", type=float, default=SITE_LAT)
    ap.add_argument("--site-lon", type=float, default=SITE_LON)
    ap.add_argument("--phase-config", default=None)
    ap.add_argument("--plots", dest="plots", action="store_true", default=True)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--os-min-n-used", type=int, default=DEFAULT_OS_MIN_N_USED)
    ap.add_argument("--os-max-km-max", type=float, default=DEFAULT_OS_MAX_KM_MAX)
    ap.add_argument("--os-min-minutes", type=int, default=DEFAULT_OS_MIN_MINUTES_PER_DAY)
    ap.add_argument("--local-min-minutes", type=int, default=DEFAULT_LOCAL_MIN_MINUTES_MATCH)
    ap.add_argument("--date-utc-offset-hours", type=int, default=DEFAULT_LOCAL_UTC_OFFSET_HOURS)
    ap.add_argument("--cr-cap", type=float, default=DEFAULT_CR_CAP)
    args = ap.parse_args()

    args.opensky_dir = normalize_path(args.opensky_dir)
    args.local_dir = normalize_path(args.local_dir)
    output_root = Path(normalize_path(args.output_root)) if args.output_root else OUTPUT_DIR
    out_dir = str(output_root / args.out_subdir)
    ensure_dir(out_dir)

    cfg = get_config(args.phase_config)

    log.info("=" * 70)
    log.info("OpenSky vs ローカル ADS-B 比較評価 v2")
    log.info("=" * 70)
    log.info(f"  OpenSky ディレクトリ: {args.opensky_dir}")
    log.info(f"  ローカル dir:          {args.local_dir}")
    log.info(f"  出力:                  {out_dir}")
    log.info(f"  サイト:                ({args.site_lat:.6f}, {args.site_lon:.6f})")
    log.info(f"  距離帯:                {DISTANCE_BIN_LABELS}")
    log.info(f"  フェーズ設定:          {cfg.config_path}")
    log.info(f"  品質フィルタ: os_min_n_used={args.os_min_n_used}  os_max_km_max={args.os_max_km_max}")
    log.info(f"    os_min_minutes/day={args.os_min_minutes}  local_min_minutes={args.local_min_minutes}")
    log.info(f"    date_utc_offset_h={args.date_utc_offset_hours}  cr_cap={args.cr_cap}")
    log.info("")

    log.info("[1/6] OpenSky データ読み込み（品質フィルタ適用）...")
    opensky_data, os_quality = load_opensky_data(
        args.opensky_dir, args.opensky_pattern,
        min_n_used=args.os_min_n_used, max_km_max=args.os_max_km_max,
        date_utc_offset_hours=args.date_utc_offset_hours,
    )
    if not opensky_data:
        log.info("  エラー: OpenSky データが見つかりません！")
        sys.exit(1)
    log.info(f"  生レコード数: {os_quality['raw_records']}  受理: {os_quality['accepted']}")
    log.info(f"  除外 (低n): {os_quality['rejected_low_n_used']}  除外 (高km): {os_quality['rejected_high_km_max']}")
    os_dates = set(osm.date_str for osm in opensky_data)
    log.info(f"  日付数: {len(os_dates)} 日 ({min(os_dates)} ~ {max(os_dates)})")

    log.info("\n[2/6] ローカル pos ファイルを走査中 ...")
    local_dates_exist = get_existing_local_dates(args.local_dir, args.local_pattern)
    overlap = os_dates & local_dates_exist
    log.info(f"  ローカル pos ファイル数: {len(local_dates_exist)}  重複日数: {len(overlap)}")
    if not overlap:
        log.info("  警告: 重複日がありません。統計から全日除外されます。")

    log.info("\n[3/6] ローカル ADS-B データ読み込み ...")
    local_data = load_local_data(
        args.local_dir, args.local_pattern,
        target_dates=overlap, site_latlon=(args.site_lat, args.site_lon),
    )
    log.info(f"  分単位サマリ読み込み: {len(local_data)} 件")

    log.info("\n[4/6] OpenSky + ローカルを結合中 ...")
    merged = merge_data(opensky_data, local_data, cfg, cr_cap=args.cr_cap)
    df = merged_to_dataframe(merged)
    log.info(f"  結合レコード数: {len(df)}  フェーズ: {sorted(df['phase'].unique())}")
    matched = df["local_n_unique"] > 0
    log.info(f"  ローカルデータあり: {matched.sum()} / {len(df)} ({matched.mean()*100:.1f}%)")

    minutely_csv = os.path.join(out_dir, "opensky_local_minutely_merged.csv")
    df.to_csv(minutely_csv, index=False, encoding="utf-8-sig")
    log.info(f"  出力: {minutely_csv}")

    log.info("\n[5/6] 日次サマリ作成（品質フラグ付）...")
    daily = make_daily_summary(
        df, local_dates_exist,
        os_min_minutes=args.os_min_minutes, local_min_minutes=args.local_min_minutes,
    )
    daily_csv = os.path.join(out_dir, "opensky_comparison_daily_summary.csv")
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    n_used = int(daily["use_for_stats"].sum())
    n_skip = int((~daily["use_for_stats"]).sum())
    log.info(f"  出力: {daily_csv}  総日数: {len(daily)} | 使用: {n_used} | スキップ: {n_skip}")

    df_skip = daily[~daily["use_for_stats"]].copy()
    if len(df_skip) > 0:
        skip_csv = os.path.join(out_dir, "opensky_skipped_days.csv")
        df_skip.to_csv(skip_csv, index=False, encoding="utf-8-sig")

    log.info("\n[6/6] 統計解析を実行中 ...")
    report_lines = [
        "OpenSky vs Local ADS-B Comparison Report v2",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"OpenSky dir: {args.opensky_dir}", f"Local dir: {args.local_dir}",
        f"Site: ({args.site_lat}, {args.site_lon})", "",
        "Quality thresholds:",
        f"  os_min_n_used: {args.os_min_n_used}  os_max_km_max: {args.os_max_km_max}",
        f"  os_min_minutes: {args.os_min_minutes}  local_min_minutes: {args.local_min_minutes}",
        f"  cr_cap: {args.cr_cap}", "",
        "OpenSky data quality:",
    ]
    for k, v in os_quality.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.extend([
        "", f"Total minute records: {len(df)}",
        f"Minutes with local data: {matched.sum()}",
        f"Days total: {len(daily)}  used: {n_used}  skipped: {n_skip}",
        f"Phases (all): {sorted(df['phase'].unique())}",
    ])
    used_dates = set(daily.loc[daily["use_for_stats"], "date"].values)
    df_used = df[df["date"].isin(used_dates)]
    report_lines.append(f"Phases (used): {sorted(df_used['phase'].unique())}")
    report_lines.append(f"Distance bins (km): {DISTANCE_BIN_LABELS}")
    report_lines.append("")
    report_lines.extend(run_statistics(df, daily))

    report_path = os.path.join(out_dir, "opensky_comparison_stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log.info(f"  出力: {report_path}")

    if args.plots:
        daily_used = daily[daily["use_for_stats"]].copy()
        used_dates_set = set(daily_used["date"].values)
        df_for_plot = df[df["date"].isin(used_dates_set)].copy()
        daily_plot = daily[daily["pos_file_exists"]].copy()

        log.info("\n  プロット生成中...")
        if len(df_for_plot) > 0:
            plot_capture_ratio_by_phase(df_for_plot, os.path.join(out_dir, "capture_ratio_by_phase.png"))
            plot_capture_by_distance_bin(df_for_plot, os.path.join(out_dir, "capture_by_distance_bin.png"))
            plot_distance_heatmap(df_for_plot, os.path.join(out_dir, "capture_heatmap_phase_distance.png"))
        if len(daily_plot) >= 2:
            plot_daily_trend_compact(daily_plot, os.path.join(out_dir, "daily_capture_trend.png"))
            plot_daily_bin_trend_compact(daily_plot, os.path.join(out_dir, "daily_bin_capture_trend.png"))
        log.info("  完了。")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info(f"  Output dir: {out_dir}")
    log.info(f"  Valid days: {n_used} / {len(daily)}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
