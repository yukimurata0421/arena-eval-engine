"""
adsb_fringe_decoding_quality_stats.py module.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")


from arena.lib.paths import OUTPUT_DIR as OUT_ROOT

OUTPUT_DIR  = str(OUT_ROOT / "fringe_decoding")
FRINGE_CSV  = os.path.join(OUTPUT_DIR, "fringe_decoding_stats.csv")
REPORT_TXT  = os.path.join(OUTPUT_DIR, "statistical_report.txt")
REPORT_IMG  = os.path.join(OUTPUT_DIR, "fringe_decoding_trend_report.png")


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(means, [alpha * 100, (1 - alpha) * 100])


def run_fringe_quality_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FRINGE_CSV):
        print(f"  File not found: {FRINGE_CSV}")
        return

    df = pd.read_csv(FRINGE_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'phase' not in df.columns:
        print("  phase column is missing.")
        return

    # fringe_ratio = (dist_200_300 + dist_300_plus) / total * 100
    if 'fringe_ratio' not in df.columns:
        df['fringe_ratio'] = (df['dist_200_300'] + df['dist_300_plus']) / df['total'] * 100

    print(f"  fringe_ratio range: {df['fringe_ratio'].min():.2f}% ~ {df['fringe_ratio'].max():.2f}%")
    print(f"  ※ Older version displayed 100%, but the above is the correct range.\n")

    phases = df['phase'].unique()
    report_lines = []
    report_lines.append("=== ADS-B Fringe Decoding Statistical Report (v2) ===\n")

    print("=" * 70)
    print("  Fringe Decoding Quality Analysis (200km+ ratio)")
    print("=" * 70)

    phase_stats = {}
    print(f"\n  {'Phase':<22} {'N':>4} {'Mean%':>8} {'Std%':>8} {'95%CI':>20}")
    print("  " + "-" * 65)
    report_lines.append("--- Phase Summary (Fringe Ratio %) ---")

    for phase in sorted(phases):
        mask = df['phase'] == phase
        vals = df.loc[mask, 'fringe_ratio'].values
        n = len(vals)
        mean_val = np.mean(vals)
        std_val  = np.std(vals, ddof=1) if n > 1 else 0.0
        ci = bootstrap_ci(vals) if n >= 3 else (mean_val, mean_val)

        phase_stats[phase] = {'mean': mean_val, 'std': std_val, 'n': n, 'values': vals}

        print(f"  {phase:<22} {n:>4} {mean_val:>7.2f}% {std_val:>7.2f}% "
              f"[{ci[0]:.2f}, {ci[1]:.2f}]")
        report_lines.append(f"  {phase}: mean={mean_val:.4f}%, std={std_val:.4f}%, n={n}")

    print(f"\n  --- Pairwise Comparisons ---")
    report_lines.append("\n--- Pairwise Comparisons ---")

    sorted_phases = sorted(phases)
    for i in range(len(sorted_phases)):
        for j in range(i + 1, len(sorted_phases)):
            p1, p2 = sorted_phases[i], sorted_phases[j]
            v1 = phase_stats[p1]['values']
            v2 = phase_stats[p2]['values']

            if len(v1) >= 3 and len(v2) >= 3:
                u_stat, p_val = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
                sig = "Significant *" if p_val < 0.05 else "Not Significant"
            else:
                u_stat, p_val = np.nan, np.nan
                sig = "Insufficient N"

            diff = np.mean(v2) - np.mean(v1)
            rel_change = (diff / np.mean(v1)) * 100

            print(f"\n  {p1} vs {p2}:")
            print(f"    Δ (absolute): {diff:+.4f} pp")
            print(f"    Δ (relative): {rel_change:+.2f}%")
            print(f"    Mann-Whitney P: {p_val:.6f}")
            print(f"    Verdict: {sig}")

            report_lines.append(f"\n{p1} vs {p2}:")
            report_lines.append(f"  Delta: {diff:+.4f} pp ({rel_change:+.2f}%)")
            report_lines.append(f"  P-value: {p_val:.6f}")
            report_lines.append(f"  Result: {sig}")

    print(f"\n  --- Aggregate Chi-squared Test ---")
    report_lines.append("\n--- Aggregate Chi-squared Test ---")

    if len(sorted_phases) >= 2:
        p_first, p_last = sorted_phases[0], sorted_phases[-1]
        df_first = df[df['phase'] == p_first]
        df_last  = df[df['phase'] == p_last]

        fringe_first = (df_first['dist_200_300'] + df_first['dist_300_plus']).sum()
        total_first  = df_first['total'].sum()
        fringe_last  = (df_last['dist_200_300'] + df_last['dist_300_plus']).sum()
        total_last   = df_last['total'].sum()

        table = [
            [fringe_first, total_first - fringe_first],
            [fringe_last,  total_last  - fringe_last]
        ]
        chi2, p_chi2, _, _ = sp_stats.chi2_contingency(table)

        rate_first = fringe_first / total_first * 100
        rate_last  = fringe_last  / total_last  * 100

        print(f"  {p_first}: {rate_first:.4f}% (n={total_first:,})")
        print(f"  {p_last}:  {rate_last:.4f}% (n={total_last:,})")
        print(f"  Chi2={chi2:.4f}, P={p_chi2:.2e}")
        print(f"  ※ Note: with very large N, even small differences become significant.")

        report_lines.append(f"  {p_first}: {rate_first:.4f}% (N={total_first})")
        report_lines.append(f"  {p_last}: {rate_last:.4f}% (N={total_last})")
        report_lines.append(f"  Chi2={chi2:.4f}, P={p_chi2:.2e}")

    print("=" * 70)

    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Text report: {REPORT_TXT}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 3]})

    ax1 = axes[0]
    plot_dates = df['date']
    ax1.stackplot(
        plot_dates,
        df['dist_0_100'], df['dist_100_200'], df['dist_200_300'], df['dist_300_plus'],
        labels=['0-100km', '100-200km', '200-300km', '>300km'],
        colors=['#2c3e50', '#3498db', '#e67e22', '#e74c3c'],
        alpha=0.8
    )
    ax1.set_title("Decoding Volume & Range Distribution", fontsize=13)
    ax1.set_ylabel("Packet Count")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    ax2 = axes[1]

    colors_map = {
        '1_Old_Settings':   '#3498db',
        '2_New_Filter':     '#e67e22',
        '3_Post_Cable_Fix': '#e74c3c',
    }

    for phase in sorted(df['phase'].unique()):
        mask = df['phase'] == phase
        ax2.plot(df.loc[mask, 'date'], df.loc[mask, 'fringe_ratio'],
                 'o-', label=phase, color=colors_map.get(phase, 'gray'),
                 markersize=6, linewidth=1.5)

        if 'ci_95_err' in df.columns:
            ax2.fill_between(
                df.loc[mask, 'date'],
                df.loc[mask, 'fringe_ratio'] - df.loc[mask, 'ci_95_err'],
                df.loc[mask, 'fringe_ratio'] + df.loc[mask, 'ci_95_err'],
                alpha=0.15, color=colors_map.get(phase, 'gray')
            )

    ax2.set_title("Fringe Decoding Ratio (%) with 95% CI", fontsize=13)
    ax2.set_ylabel("Fringe Ratio (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    y_min = df['fringe_ratio'].min() - 2
    y_max = df['fringe_ratio'].max() + 2
    ax2.set_ylim(y_min, y_max)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_IMG, dpi=150, bbox_inches='tight')
    print(f"  Image report: {REPORT_IMG}")

    if os.getenv("SHOW_PLOT") == "1":
        plt.show()
    plt.close()


if __name__ == "__main__":
    run_fringe_quality_analysis()
