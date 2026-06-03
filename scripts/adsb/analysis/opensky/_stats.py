"""Statistical tests for OpenSky vs Local ADS-B comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from _common import DISTANCE_BIN_LABELS
from scipy.stats import kruskal

from arena.lib.robust_stats import format_estimate_ci, mann_whitney_hodges_lehmann


def run_statistics(df_all: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    sep = "=" * 70

    used_dates = set(daily.loc[daily["use_for_stats"], "date"].values)
    df = df_all[df_all["date"].isin(used_dates)].copy()
    daily_used = daily[daily["use_for_stats"]].copy()

    phases = sorted(df["phase"].unique())

    lines.append(sep)
    lines.append("DATA QUALITY SUMMARY")
    lines.append(sep)
    lines.append(f"Total days in data:       {len(daily)}")
    lines.append(f"Days with pos file:       {int(daily['pos_file_exists'].sum())}")
    lines.append(f"Days used for stats:      {int(daily['use_for_stats'].sum())}")
    lines.append(f"Days skipped:             {int((~daily['use_for_stats']).sum())}")
    lines.append(f"Total minutes (all):      {len(df_all)}")
    lines.append(f"Total minutes (used):     {len(df)}")
    lines.append(f"Phases in used data:      {phases}")
    lines.append("")

    skip_summary = daily[~daily["use_for_stats"]].groupby("skip_reason").size()
    for reason, cnt in skip_summary.items():
        lines.append(f"  skip: {reason} = {cnt} days")
    lines.append("")

    if len(df) == 0:
        lines.append("[ERROR] No valid data after filtering. Cannot compute statistics.")
        return lines

    # 1. Descriptive Statistics
    lines.append(sep)
    lines.append("1. Descriptive Statistics (used days only)")
    lines.append(sep)
    lines.append("")

    for ph in phases:
        d = df[df["phase"] == ph]
        n_days = int(daily_used[daily_used["phase"] == ph].shape[0])
        lines.append(f"-- Phase: {ph} (n_days={n_days}, n_minutes={len(d)}) --")
        lines.append(f"  OpenSky  n_used:  mean={d['os_n_used'].mean():.1f}  median={d['os_n_used'].median():.1f}")
        lines.append(f"  Local    n_unique: mean={d['local_n_unique'].mean():.1f}  median={d['local_n_unique'].median():.1f}")
        cr = d["capture_ratio"].dropna()
        if len(cr) > 0:
            lines.append(f"  Capture ratio:     mean={cr.mean():.4f}  median={cr.median():.4f}  std={cr.std():.4f}")
        lines.append(f"  Reach: os_km_max={d['os_km_max'].max():.1f}  local_km_max={d['local_km_max'].max():.1f}")
        for lab in DISTANCE_BIN_LABELS:
            vals = d[f"capture_bin_{lab}"].dropna()
            if len(vals) > 0:
                lines.append(f"  Capture {lab:>8s} km:  mean={vals.mean():.4f}  median={vals.median():.4f}  n={len(vals)}")
        lines.append("")

    # 2. Mann-Whitney U (capture_ratio)
    lines.append(sep)
    lines.append("2. Phase comparison: Mann-Whitney U + Hodges-Lehmann CI (capture_ratio, minute-level)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2:
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                ph_a, ph_b = phases[i], phases[j]
                a = df.loc[df["phase"] == ph_a, "capture_ratio"].dropna().values
                b = df.loc[df["phase"] == ph_b, "capture_ratio"].dropna().values
                if len(a) >= 5 and len(b) >= 5:
                    mwu_hl = mann_whitney_hodges_lehmann(a, b, min_samples=5)
                    lines.append(f"  {ph_a} (n={len(a)}) vs {ph_b} (n={len(b)})")
                    lines.append(f"    MWU: U={mwu_hl.u_statistic:.1f}  p={mwu_hl.p_value:.6g}")
                    lines.append(
                        f"    HL shift({ph_b}-{ph_a}): "
                        f"{format_estimate_ci(mwu_hl.hl_estimate, mwu_hl.hl_ci_low, mwu_hl.hl_ci_high, digits=4)}"
                    )
                    lines.append(f"    rank_biserial_effect={mwu_hl.rank_biserial:.4f}")
                    lines.append("")
    else:
        lines.append("  Only 1 phase in valid data — no comparison possible.")
        lines.append("")

    # 3. Distance-bin phase comparison
    lines.append(sep)
    lines.append("3. Distance-bin phase comparison (Mann-Whitney U + Hodges-Lehmann CI on capture_bin)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2:
        for lab in DISTANCE_BIN_LABELS:
            col = f"capture_bin_{lab}"
            lines.append(f"-- {lab} km --")
            for i in range(len(phases)):
                for j in range(i + 1, len(phases)):
                    ph_a, ph_b = phases[i], phases[j]
                    a = df.loc[df["phase"] == ph_a, col].dropna().values
                    b = df.loc[df["phase"] == ph_b, col].dropna().values
                    if len(a) >= 5 and len(b) >= 5:
                        mwu_hl = mann_whitney_hodges_lehmann(a, b, min_samples=5)
                        lines.append(
                            f"  {ph_a} vs {ph_b}: MWU p={mwu_hl.p_value:.6g}  "
                            f"HL={format_estimate_ci(mwu_hl.hl_estimate, mwu_hl.hl_ci_low, mwu_hl.hl_ci_high, digits=4)}"
                        )
                    else:
                        lines.append(f"  {ph_a} vs {ph_b}: insufficient data (n={len(a)}/{len(b)})")
            lines.append("")

    # 4. Daily-level comparison
    lines.append(sep)
    lines.append("4. Daily-level comparison (MWU + Hodges-Lehmann CI on daily median_capture_ratio)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2 and len(daily_used) >= 4:
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                ph_a, ph_b = phases[i], phases[j]
                a = daily_used.loc[daily_used["phase"] == ph_a, "median_capture_ratio"].dropna().values
                b = daily_used.loc[daily_used["phase"] == ph_b, "median_capture_ratio"].dropna().values
                if len(a) >= 3 and len(b) >= 3:
                    mwu_hl = mann_whitney_hodges_lehmann(a, b, min_samples=3)
                    lines.append(f"  {ph_a} (n_days={len(a)}, mean={np.mean(a):.4f})")
                    lines.append(f"  vs {ph_b} (n_days={len(b)}, mean={np.mean(b):.4f})")
                    lines.append(f"    MWU: U={mwu_hl.u_statistic:.1f}  p={mwu_hl.p_value:.6g}")
                    lines.append(
                        "    HL shift: "
                        f"{format_estimate_ci(mwu_hl.hl_estimate, mwu_hl.hl_ci_low, mwu_hl.hl_ci_high, digits=4)}"
                    )
                    lines.append(f"    rank_biserial_effect={mwu_hl.rank_biserial:.4f}")
                    lines.append("")
                else:
                    lines.append(f"  {ph_a} (n={len(a)}) vs {ph_b} (n={len(b)}): need >=3 each")
                    lines.append("")
    else:
        lines.append("  Insufficient data for daily comparison.")
        lines.append("")

    # 5. NB-GLM
    lines.append(sep)
    lines.append("5. Negative Binomial GLM: local_n_unique ~ phase + offset(log(os_n_used))")
    lines.append(sep)
    lines.append("")

    try:
        d_glm = df[df["os_n_used"] > 0].copy()
        d_glm["log_os"] = np.log(d_glm["os_n_used"].astype(float))
        formula = "local_n_unique ~ C(phase)" if len(d_glm["phase"].unique()) >= 2 else "local_n_unique ~ 1"
        model = smf.glm(
            formula=formula,
            data=d_glm,
            family=sm.families.NegativeBinomial(alpha=1.0),
            offset=d_glm["log_os"],
        )
        res = model.fit()
        lines.append(res.summary().as_text())
    except Exception as e:
        lines.append(f"  [ERROR] NB-GLM failed: {e!r}")
    lines.append("")

    # 6. Distance-bin NB-GLM
    lines.append(sep)
    lines.append("6. Distance-bin NB-GLM: local_bin ~ phase + offset(log(os_bin))")
    lines.append(sep)
    lines.append("")

    for lab in DISTANCE_BIN_LABELS:
        lines.append(f"-- {lab} km --")
        try:
            d_bin = df[["phase", f"local_bin_{lab}", f"os_bin_{lab}"]].copy()
            d_bin.columns = ["phase", "local_count", "os_count"]
            d_bin = d_bin[d_bin["os_count"] > 0.5].copy()
            d_bin["local_count"] = d_bin["local_count"].astype(int)
            d_bin["log_os"] = np.log(d_bin["os_count"].astype(float))
            if len(d_bin) < 10:
                lines.append("  insufficient data")
                lines.append("")
                continue
            formula = "local_count ~ C(phase)" if len(d_bin["phase"].unique()) >= 2 else "local_count ~ 1"
            model = smf.glm(
                formula=formula,
                data=d_bin,
                family=sm.families.NegativeBinomial(alpha=1.0),
                offset=d_bin["log_os"],
            )
            res = model.fit()
            lines.append(res.summary().as_text())
        except Exception as e:
            lines.append(f"  [ERROR] {e!r}")
        lines.append("")

    # 7. Kruskal-Wallis (3+ phases)
    if len(phases) >= 3:
        lines.append(sep)
        lines.append("7. Kruskal-Wallis test (capture_ratio across all phases)")
        lines.append(sep)
        lines.append("")

        groups = []
        group_names = []
        for ph in phases:
            vals = df.loc[df["phase"] == ph, "capture_ratio"].dropna().values
            if len(vals) >= 3:
                groups.append(vals)
                group_names.append(f"{ph}(n={len(vals)})")
        if len(groups) >= 3:
            h_stat, h_p = kruskal(*groups)
            lines.append(f"  Groups: {', '.join(group_names)}")
            lines.append(f"  H={h_stat:.4f}  p={h_p:.6g}")
        elif len(groups) >= 2:
            lines.append(f"  Only {len(groups)} groups with n>=3 (need 3+). Using 2-group MWU instead.")
        else:
            lines.append("  Not enough groups with n>=3")
        lines.append("")

    return lines
