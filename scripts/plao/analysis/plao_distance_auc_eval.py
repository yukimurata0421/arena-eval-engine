# <project>/scripts/plao/analysis/plao_distance_auc_eval.py
# -*- coding: utf-8 -*-
"""
PLAO Distance-bin AUC Evaluator (pos schema_ver=1)

Input line example:
{"type":"pos","schema_ver":1,"ts":...,"hex":"ab4bf7","seen":0.1,"lat":...,"lon":...,"alt":...}

Core:
- distance-bin AUC: sum over minutes of unique-aircraft count per bin
- minutes_covered normalize: auc_norm = auc * (1440/minutes_covered)
- auto-skip low-quality days for stats (still kept in daily CSV)
- stats: weekday Kruskal, (optional) intervention MWU+bootstrap, NB-GLM with offset log(minutes_covered)
- plots: compact (only used days, x-axis packed) OR calendar (real dates, gaps kept)

Outputs:
<project>/output/plao/distance_auc/
  - plao_daily_distance_auc_summary.csv
  - plao_daily_distance_auc_long.csv
  - plao_skipped_days.csv
  - plao_distance_auc_stats_report.txt
  - (optional) png plots
"""

from __future__ import annotations

import os
import re
import json
import math
import glob
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from arena.lib.config import get_site_latlon
from arena.lib.paths import DATA_DIR, OUTPUT_DIR

# ----------------------------
# Basic utilities
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def normalize_win_path_for_wsl(p: str) -> str:
    """
    Convert Windows-style paths like C:\\foo\\bar to /mnt/c/foo/bar when running on WSL.
    """
    if os.name == "nt":
        return p
    m = re.match(r"^([A-Za-z]):[\\\\/](.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def normalize_subdir_for_wsl(p: str) -> str:
    """
    Relative subdir may contain backslashes (Windows style). Convert to POSIX on WSL.
    """
    if os.name == "nt":
        return p
    return p.replace("\\", "/")

def parse_date_from_filename(path: str) -> Optional[str]:
    m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(path))
    return m.group(1) if m else None

def utc_dt_from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)

def weekday_name_from_date_yyyymmdd(s: str) -> str:
    dt = datetime.strptime(s, "%Y%m%d")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return names[dt.weekday()]

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def safe_str(x) -> Optional[str]:
    try:
        if x is None:
            return None
        s = str(x)
        return s if s else None
    except Exception:
        return None


# ----------------------------
# Bins
# ----------------------------

DEFAULT_BINS_KM = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 9999]

def make_bin_labels(edges: List[int]) -> List[str]:
    return [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]


# ----------------------------
# Data model
# ----------------------------

@dataclass
class DayResult:
    date: str
    minutes_covered: int
    auc_total: float
    auc_norm_total: float
    auc_by_bin: Dict[str, float]
    auc_norm_by_bin: Dict[str, float]
    n_lines: int
    n_used: int


# ----------------------------
# JSONL reader
# ----------------------------

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ----------------------------
# Core aggregation (PLAO schema_ver=1)
# ----------------------------

def compute_day_auc_plao_pos(
    jsonl_path: str,
    site_latlon: Tuple[float, float],
    bin_edges_km: List[int],
) -> DayResult:
    date = parse_date_from_filename(jsonl_path)
    if not date:
        date = datetime.fromtimestamp(os.path.getmtime(jsonl_path)).strftime("%Y%m%d")

    labels = make_bin_labels(bin_edges_km)

    # minute_key -> bin_label -> set(hex)
    minute_bin_sets: Dict[int, Dict[str, set]] = {}
    minutes_seen: set[int] = set()

    n_lines = 0
    n_used = 0

    site_lat, site_lon = site_latlon

    for rec in iter_jsonl(jsonl_path):
        n_lines += 1

        if rec.get("type") != "pos":
            continue
        if int(rec.get("schema_ver", 0) or 0) != 1:
            continue

        tsf = safe_float(rec.get("ts"))
        if tsf is None:
            continue

        hx = safe_str(rec.get("hex"))
        if hx is None:
            continue

        lat = safe_float(rec.get("lat"))
        lon = safe_float(rec.get("lon"))
        if lat is None or lon is None:
            continue

        # minute bucket (UTC minute)
        dt = utc_dt_from_ts(tsf)
        minute_key = int(dt.timestamp() // 60)
        minutes_seen.add(minute_key)

        # distance
        dist_km = haversine_km(site_lat, site_lon, lat, lon)

        # bin
        bidx = None
        for i in range(len(bin_edges_km) - 1):
            if bin_edges_km[i] <= dist_km < bin_edges_km[i + 1]:
                bidx = i
                break
        if bidx is None:
            continue

        blabel = labels[bidx]
        if minute_key not in minute_bin_sets:
            minute_bin_sets[minute_key] = {lab: set() for lab in labels}

        minute_bin_sets[minute_key][blabel].add(hx)
        n_used += 1

    minutes_covered = len(minutes_seen)

    auc_by_bin = {lab: 0.0 for lab in labels}
    for _, bins in minute_bin_sets.items():
        for lab, s in bins.items():
            auc_by_bin[lab] += float(len(s))

    auc_total = float(sum(auc_by_bin.values()))
    if minutes_covered <= 0:
        auc_norm_by_bin = {lab: 0.0 for lab in labels}
        auc_norm_total = 0.0
    else:
        factor = 1440.0 / float(minutes_covered)
        auc_norm_by_bin = {lab: v * factor for lab, v in auc_by_bin.items()}
        auc_norm_total = auc_total * factor

    return DayResult(
        date=date,
        minutes_covered=minutes_covered,
        auc_total=auc_total,
        auc_norm_total=auc_norm_total,
        auc_by_bin=auc_by_bin,
        auc_norm_by_bin=auc_norm_by_bin,
        n_lines=n_lines,
        n_used=n_used,
    )


# ----------------------------
# Stats helpers
# ----------------------------

def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 123) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)

    diffs = []
    for _ in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(bb) - np.mean(aa)))
    diffs = np.asarray(diffs)
    return (float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))

def run_nb_glm(df: pd.DataFrame, y_col: str) -> str:
    """
    Negative Binomial GLM with offset log(minutes_covered)
    predictors: post + C(weekday)
    """
    d = df[df["minutes_covered"] > 0].copy()
    d["log_minutes"] = np.log(d["minutes_covered"].astype(float))

    has_post = "post" in d.columns and d["post"].notna().any()
    if has_post:
        formula = f"{y_col} ~ post + C(weekday)"
    else:
        formula = f"{y_col} ~ C(weekday)"

    model = smf.glm(
        formula=formula,
        data=d,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=d["log_minutes"],
    )
    res = model.fit()
    return res.summary().as_text()

def detect_jax_backend() -> str:
    # informational only
    try:
        import jax  # type: ignore
        devs = jax.devices()
        if any(d.platform == "gpu" for d in devs):
            return "jax-gpu"
        return "jax-cpu"
    except Exception:
        return "no-jax"


# ----------------------------
# Plotting
# ----------------------------

def plot_compact_total(df_used: pd.DataFrame, out_path: str) -> None:
    """
    X-axis packed: 0..N-1; ticks are dates
    """
    d = df_used.reset_index(drop=True).copy()
    x = np.arange(len(d))
    labels = d["date"].tolist()

    plt.figure()
    plt.plot(x, d["auc_norm_total"].values)
    plt.title("PLAO: normalized total AUC (compact; used days only)")
    plt.xlabel("used-day index (packed)")
    plt.ylabel("auc_norm_total")
    # avoid overcrowding
    step = max(1, len(labels)//12)
    plt.xticks(x[::step], labels[::step], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_compact_bins(df_used: pd.DataFrame, bin_labels: List[str], out_path: str) -> None:
    d = df_used.reset_index(drop=True).copy()
    x = np.arange(len(d))
    labels = d["date"].tolist()

    plt.figure()
    for lab in bin_labels:
        plt.plot(x, d[f"auc_norm_{lab}"].values, label=lab)
    plt.title("PLAO: normalized distance-bin AUC (compact; used days only)")
    plt.xlabel("used-day index (packed)")
    plt.ylabel("auc_norm_bin")
    step = max(1, len(labels)//12)
    plt.xticks(x[::step], labels[::step], rotation=45, ha="right")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_calendar_total(df_all: pd.DataFrame, out_path: str) -> None:
    d = df_all.copy()
    d["dt"] = pd.to_datetime(d["date"], format="%Y%m%d")
    y = d["auc_norm_total"].astype(float).copy()
    # make skipped days NaN to show gaps
    y[~d["use_for_stats"].astype(bool)] = np.nan

    plt.figure()
    plt.plot(d["dt"], y.values)
    plt.title("PLAO: normalized total AUC (calendar; gaps for skipped days)")
    plt.xlabel("date")
    plt.ylabel("auc_norm_total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_calendar_bins(df_all: pd.DataFrame, bin_labels: List[str], out_path: str) -> None:
    d = df_all.copy()
    d["dt"] = pd.to_datetime(d["date"], format="%Y%m%d")

    plt.figure()
    for lab in bin_labels:
        y = d[f"auc_norm_{lab}"].astype(float).copy()
        y[~d["use_for_stats"].astype(bool)] = np.nan
        plt.plot(d["dt"], y.values, label=lab)

    plt.title("PLAO: normalized distance-bin AUC (calendar; gaps for skipped days)")
    plt.xlabel("date")
    plt.ylabel("auc_norm_bin")
    plt.xticks(rotation=45)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-dir", default=str(DATA_DIR / "plao_pos"))
    ap.add_argument("--pattern", default="pos_*.jsonl")

    ap.add_argument("--output-root", default=str(OUTPUT_DIR))
    ap.add_argument("--out-subdir", default=r"plao/distance_auc")

    ap.add_argument("--bin-edges", default="0,25,50,75,100,125,150,175,200,250,9999")

    # site fixed (your provided default)
    default_lat, default_lon = get_site_latlon()
    ap.add_argument("--site-lat", type=float, default=default_lat)
    ap.add_argument("--site-lon", type=float, default=default_lon)

    # intervention
    ap.add_argument("--intervention-date", default=None, help="YYYYMMDD optional")

    # quality filters
    ap.add_argument("--min-minutes-covered", type=int, default=60)
    ap.add_argument("--skip-threshold-ratio", type=float, default=0.8)

    # plots
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot-mode", choices=["compact", "calendar"], default="compact",
                    help="compact: pack x-axis to used days only; calendar: keep real dates with gaps")

    args = ap.parse_args()

    args.input_dir = normalize_win_path_for_wsl(args.input_dir)
    args.output_root = normalize_win_path_for_wsl(args.output_root)
    args.out_subdir = normalize_subdir_for_wsl(args.out_subdir)

    bin_edges = [int(x.strip()) for x in args.bin_edges.split(",") if x.strip()]
    bin_labels = make_bin_labels(bin_edges)

    out_dir = os.path.join(args.output_root, args.out_subdir)
    ensure_dir(out_dir)

    jax_backend = detect_jax_backend()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No input files matched: {os.path.join(args.input_dir, args.pattern)}")

    site_latlon = (float(args.site_lat), float(args.site_lon))

    print(f"[INFO] Input files: {len(paths)}  dir={args.input_dir} pattern={args.pattern}")
    print(f"[INFO] Site(lat,lon)=({site_latlon[0]:.8f},{site_latlon[1]:.8f})")
    print(f"[INFO] JAX backend detect: {jax_backend}")
    print(f"[INFO] bin_edges={bin_edges}")

    # ----------------------------
    # Aggregate all days
    # ----------------------------
    day_results: List[DayResult] = []
    for i, p in enumerate(paths, start=1):
        date = parse_date_from_filename(p) or "unknown"
        print(f"[RUN] ({i}/{len(paths)}) {os.path.basename(p)} date={date}")

        r = compute_day_auc_plao_pos(
            jsonl_path=p,
            site_latlon=site_latlon,
            bin_edges_km=bin_edges,
        )
        day_results.append(r)

        print(f"      lines={r.n_lines:,} used={r.n_used:,} minutes_covered={r.minutes_covered:,} auc_total={r.auc_total:,.0f} auc_norm_total={r.auc_norm_total:,.1f}")

    # build daily dataframe
    rows = []
    for r in day_results:
        row = {
            "date": r.date,
            "weekday": weekday_name_from_date_yyyymmdd(r.date),
            "minutes_covered": r.minutes_covered,
            "n_lines": r.n_lines,
            "n_used": r.n_used,
            "auc_total": r.auc_total,
            "auc_norm_total": r.auc_norm_total,
        }
        for lab in bin_labels:
            row[f"auc_{lab}"] = r.auc_by_bin.get(lab, 0.0)
            row[f"auc_norm_{lab}"] = r.auc_norm_by_bin.get(lab, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # add intervention flag
    if args.intervention_date:
        iv = args.intervention_date
        df["post"] = (df["date"] >= iv).astype(int)
    else:
        df["post"] = np.nan

    # filters
    df["hard_ok"] = df["minutes_covered"] >= int(args.min_minutes_covered)

    mean_total = float(df.loc[df["hard_ok"], "auc_norm_total"].mean()) if df["hard_ok"].any() else float(df["auc_norm_total"].mean())
    thr = args.skip_threshold_ratio * mean_total
    df["skip_by_low_quality"] = df["auc_norm_total"] < thr
    df["use_for_stats"] = df["hard_ok"] & (~df["skip_by_low_quality"])

    # write daily summary (all days)
    daily_csv = os.path.join(out_dir, "plao_daily_distance_auc_summary.csv")
    df.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    # long format
    long_rows = []
    for _, r in df.iterrows():
        for lab in bin_labels:
            long_rows.append({
                "date": r["date"],
                "weekday": r["weekday"],
                "minutes_covered": int(r["minutes_covered"]),
                "use_for_stats": bool(r["use_for_stats"]),
                "bin": lab,
                "auc": float(r[f"auc_{lab}"]),
                "auc_norm": float(r[f"auc_norm_{lab}"]),
            })
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(os.path.join(out_dir, "plao_daily_distance_auc_long.csv"), index=False, encoding="utf-8-sig")

    # skipped days csv (for auditability)
    df_skip = df[~df["use_for_stats"]].copy()
    df_skip["skip_reason"] = np.where(~df_skip["hard_ok"], "minutes_covered_below_min", "auc_norm_total_below_threshold")
    df_skip["threshold_auc_norm_total"] = thr
    df_skip.to_csv(os.path.join(out_dir, "plao_skipped_days.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # Stats report (only used days)
    # ----------------------------
    lines = []
    lines.append("PLAO Distance-bin AUC Evaluator (pos schema_ver=1)")
    lines.append(f"input_dir={args.input_dir} pattern={args.pattern}")
    lines.append(f"files={len(paths)} days={len(df)}")
    lines.append(f"site_latlon=({site_latlon[0]},{site_latlon[1]})")
    lines.append(f"jax_backend={jax_backend}")
    lines.append(f"bin_edges={bin_edges}")
    lines.append("")
    lines.append("Quality rules")
    lines.append(f"min_minutes_covered={args.min_minutes_covered}")
    lines.append(f"mean_auc_norm_total(hard_ok)={mean_total:.3f}")
    lines.append(f"skip_threshold_ratio={args.skip_threshold_ratio} => threshold={thr:.3f}")
    if args.intervention_date:
        lines.append(f"intervention_date={args.intervention_date}")
    lines.append("")
    lines.append("Counts")
    lines.append(f"hard_ok_days={int(df['hard_ok'].sum())}/{len(df)}")
    lines.append(f"skipped_low_quality_days={int(df['skip_by_low_quality'].sum())}/{len(df)}")
    lines.append(f"use_for_stats_days={int(df['use_for_stats'].sum())}/{len(df)}")
    lines.append("")

    dstat = df[df["use_for_stats"]].copy()
    if len(dstat) < 5:
        lines.append("[WARN] Too few days after filtering. Consider lowering --min-minutes-covered or --skip-threshold-ratio.")
    else:
        # weekday effect test
        lines.append("Weekday effect test (Kruskal-Wallis) on auc_norm_total")
        groups = []
        group_names = []
        for wd in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            v = dstat.loc[dstat["weekday"] == wd, "auc_norm_total"].values
            if len(v) >= 2:
                groups.append(v)
                group_names.append(f"{wd}(n={len(v)})")
        if len(groups) >= 2:
            st = kruskal(*groups)
            lines.append(f"groups: {', '.join(group_names)}")
            lines.append(f"H={st.statistic:.4f} p={st.pvalue:.6g}")
        else:
            lines.append("Not enough weekday groups with n>=2.")
        lines.append("")

        # intervention test
        if args.intervention_date:
            lines.append("Intervention test (Mann-Whitney U) on auc_norm_total (pre vs post)")
            pre = dstat.loc[dstat["date"] < args.intervention_date, "auc_norm_total"].values
            post = dstat.loc[dstat["date"] >= args.intervention_date, "auc_norm_total"].values
            if len(pre) >= 3 and len(post) >= 3:
                u = mannwhitneyu(pre, post, alternative="two-sided")
                md, lo, hi = bootstrap_mean_diff(pre, post, n=20000, seed=123)
                lines.append(f"pre n={len(pre)} mean={float(np.mean(pre)):.3f}")
                lines.append(f"post n={len(post)} mean={float(np.mean(post)):.3f}")
                lines.append(f"MWU U={u.statistic:.3f} p={u.pvalue:.6g}")
                lines.append(f"bootstrap mean_diff(post-pre)={md:.3f} 95%CI[{lo:.3f},{hi:.3f}]")
            else:
                lines.append("Not enough samples for pre/post (need >=3 each).")
            lines.append("")

        # NB-GLM (auc_total, offset=log(minutes_covered))
        lines.append("Negative Binomial GLM on auc_total with offset log(minutes_covered)")
        try:
            lines.append(run_nb_glm(dstat, "auc_total"))
        except Exception as e:
            lines.append(f"[ERROR] NB-GLM failed: {repr(e)}")
        lines.append("")

    report_txt = os.path.join(out_dir, "plao_distance_auc_stats_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # plots
    if args.plots:
        df_used = df[df["use_for_stats"]].copy()
        if args.plot_mode == "compact":
            plot_compact_total(df_used, os.path.join(out_dir, "plao_auc_norm_total_trend_compact.png"))
            plot_compact_bins(df_used, bin_labels, os.path.join(out_dir, "plao_auc_norm_bins_trend_compact.png"))
        else:
            plot_calendar_total(df, os.path.join(out_dir, "plao_auc_norm_total_trend_calendar.png"))
            plot_calendar_bins(df, bin_labels, os.path.join(out_dir, "plao_auc_norm_bins_trend_calendar.png"))

    print("")
    print(f"[OK] wrote: {daily_csv}")
    print(f"[OK] wrote: {report_txt}")
    print(f"[OK] wrote: {os.path.join(out_dir, 'plao_skipped_days.csv')}")
    print(f"[OK] out_dir: {out_dir}")
    print(f"[INFO] use_for_stats_days={int(df['use_for_stats'].sum())}/{len(df)}")
    print(f"[INFO] quality_threshold(auc_norm_total)={thr:.3f} (ratio={args.skip_threshold_ratio})")
    print(f"[INFO] plot_mode={args.plot_mode}")
    if args.intervention_date:
        print(f"[INFO] intervention_date={args.intervention_date}")


if __name__ == "__main__":
    main()
