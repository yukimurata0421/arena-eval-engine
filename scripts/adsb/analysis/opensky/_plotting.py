"""Plotting functions for OpenSky vs Local ADS-B comparison."""
from __future__ import annotations

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import DISTANCE_BIN_LABELS


def _phase_colors() -> Dict[str, str]:
    return {
        "RTL-SDR": "#7f8c8d",
        "Airspy Mini": "#e74c3c",
        "Airspy+Cable": "#2980b9",
        "airspy_cable_v2": "#e67e22",
        "airspy_adapter": "#16a085",
    }


def plot_capture_ratio_by_phase(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    data = [df.loc[df["phase"] == ph, "capture_ratio"].dropna().values for ph in phases]
    colors = _phase_colors()

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=phases, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors.get(phases[i], "#cccccc"))
        patch.set_alpha(0.6)

    ax.set_title("Capture Ratio by Phase (Local / OpenSky)\n[valid days only, outliers hidden]")
    ax.set_ylabel("capture_ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1.0 (perfect)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_capture_by_distance_bin(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    n_phases = len(phases)
    n_bins = len(DISTANCE_BIN_LABELS)
    colors = _phase_colors()

    medians = np.zeros((n_phases, n_bins))
    q25 = np.zeros((n_phases, n_bins))
    q75 = np.zeros((n_phases, n_bins))

    for i, ph in enumerate(phases):
        d = df[df["phase"] == ph]
        for j, lab in enumerate(DISTANCE_BIN_LABELS):
            vals = d[f"capture_bin_{lab}"].dropna().values
            if len(vals) > 0:
                medians[i, j] = np.median(vals)
                q25[i, j] = np.quantile(vals, 0.25)
                q75[i, j] = np.quantile(vals, 0.75)

    x = np.arange(n_bins)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ph in enumerate(phases):
        offset = (i - n_phases / 2 + 0.5) * width
        yerr_lo = medians[i] - q25[i]
        yerr_hi = q75[i] - medians[i]
        ax.bar(x + offset, medians[i], width,
               yerr=[yerr_lo, yerr_hi],
               label=ph, color=colors.get(ph, "#cccccc"), alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} km" for lab in DISTANCE_BIN_LABELS])
    ax.set_ylabel("Median capture ratio (IQR)")
    ax.set_title("Capture Ratio by Distance Bin x Phase\n[valid days only]")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_daily_trend_compact(daily_plot: pd.DataFrame, out_path: str) -> None:
    d = daily_plot.sort_values("date").reset_index(drop=True)
    x = np.arange(len(d))
    phases = sorted(d["phase"].unique())
    colors = _phase_colors()

    fig, ax = plt.subplots(figsize=(12, 5))
    for ph in phases:
        mask = d["phase"] == ph
        ax.plot(x[mask], d.loc[mask, "median_capture_ratio"].values, "o-",
                label=ph, color=colors.get(ph, "#2980b9"), alpha=0.8, markersize=5)

    skipped_mask = ~d["use_for_stats"]
    if skipped_mask.any():
        ax.scatter(
            x[skipped_mask],
            d.loc[skipped_mask, "median_capture_ratio"].values,
            marker="x", color="#2c3e50", s=36, alpha=0.8, label="skipped day",
        )

    ax.set_title("Daily Median Capture Ratio (compact; pos-file days)")
    ax.set_ylabel("median_capture_ratio")
    ax.set_xlabel("Date (packed)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    step = max(1, len(d) // 15)
    tick_idx = list(range(0, len(d), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([d.loc[i, "date_iso"] for i in tick_idx], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_distance_heatmap(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    matrix = np.zeros((len(phases), len(DISTANCE_BIN_LABELS)))

    for i, ph in enumerate(phases):
        d = df[df["phase"] == ph]
        for j, lab in enumerate(DISTANCE_BIN_LABELS):
            vals = d[f"capture_bin_{lab}"].dropna()
            matrix[i, j] = vals.median() if len(vals) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, max(3, len(phases) * 1.2)))
    vmax = min(float(matrix.max()) * 1.2, 3.0) if matrix.max() > 0 else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(DISTANCE_BIN_LABELS)))
    ax.set_xticklabels([f"{l} km" for l in DISTANCE_BIN_LABELS])
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases)
    ax.set_title("Median Capture Ratio: Phase x Distance Bin\n[used days only]")

    for i in range(len(phases)):
        for j in range(len(DISTANCE_BIN_LABELS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if matrix[i, j] > 0.4 else "white")

    fig.colorbar(im, ax=ax, label="capture ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_daily_bin_trend_compact(daily_plot: pd.DataFrame, out_path: str) -> None:
    d = daily_plot.sort_values("date").reset_index(drop=True)
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.viridis
    for i, lab in enumerate(DISTANCE_BIN_LABELS):
        col = f"median_capture_{lab}"
        if col in d.columns:
            ax.plot(x, d[col].values, "o-", label=f"{lab} km",
                    color=cmap(i / len(DISTANCE_BIN_LABELS)), alpha=0.7, markersize=3)

    skipped_mask = ~d["use_for_stats"]
    if skipped_mask.any():
        for i, lab in enumerate(DISTANCE_BIN_LABELS):
            col = f"median_capture_{lab}"
            if col not in d.columns:
                continue
            vals = d.loc[skipped_mask, col].values
            ax.scatter(
                x[skipped_mask], vals,
                marker="x", s=20,
                color=plt.cm.viridis(i / len(DISTANCE_BIN_LABELS)), alpha=0.75,
            )

    ax.set_title("Daily Median Capture by Distance Bin (compact; pos-file days)")
    ax.set_ylabel("median capture ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    step = max(1, len(d) // 15)
    tick_idx = list(range(0, len(d), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([d.loc[i, "date_iso"] for i in tick_idx], rotation=45, ha="right")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
