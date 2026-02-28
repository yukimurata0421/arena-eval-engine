"""
adsb_distance_nb_eval.py module.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")


from arena.lib.paths import OUTPUT_DIR as OUT_ROOT

OUTPUT_DIR  = str(OUT_ROOT / "performance")
FRINGE_CSV  = os.path.join(str(OUT_ROOT / "fringe_decoding"), "fringe_decoding_stats.csv")

PHASES = {
    '1_Old_Settings':   'Phase0_Old',
    '2_New_Filter':     'Phase1_Filter',
    '3_Post_Cable_Fix': 'Phase2_Cable',
}


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    return np.percentile(means, [alpha * 100, (1 - alpha) * 100])


def run_distance_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FRINGE_CSV):
        print(f"  File not found: {FRINGE_CSV}")
        print("  Run adsb_fringe_decoding_evaluator.py first.")
        return

    df = pd.read_csv(FRINGE_CSV)
    df['date'] = pd.to_datetime(df['date'])

    if 'phase' in df.columns:
        df['phase_label'] = df['phase'].map(PHASES).fillna(df['phase'])
    else:
        print("  phase column is missing.Check fringe_decoding_stats.csv.")
        return

    dist_cols = {
        '0-100km (Near)':    'dist_0_100',
        '100-200km (Mid)':   'dist_100_200',
        '200-300km (Far)':   'dist_200_300',
        '300km+ (Extreme)':  'dist_300_plus',
    }

    for label, col in dist_cols.items():
        df[f'ratio_{col}'] = df[col] / df['total'] * 100

    phases = sorted(df['phase_label'].unique())
    baseline_phase = phases[0]
    baseline_data = df[df['phase_label'] == baseline_phase]

    print("=" * 80)
    print("  Distance-wise Performance Analysis (Rate Comparison)")
    print(f"  Baseline: {baseline_phase} ({len(baseline_data)} days)")
    print("=" * 80)

    all_results = []

    for target_phase in phases[1:]:
        target_data = df[df['phase_label'] == target_phase]
        print(f"\n--- {target_phase} ({len(target_data)} days) vs {baseline_phase} ---")
        print(f"{'Distance Band':<22} {'Baseline %':>10} {'Target %':>10} "
              f"{'Change':>10} {'P-value':>10} {'Significance':>14}")
        print("-" * 80)

        for label, col in dist_cols.items():
            ratio_col = f'ratio_{col}'
            base_vals = baseline_data[ratio_col].values
            tgt_vals  = target_data[ratio_col].values

            base_mean = np.mean(base_vals)
            tgt_mean  = np.mean(tgt_vals)

            if base_mean > 0:
                relative_change = ((tgt_mean / base_mean) - 1) * 100
            else:
                relative_change = np.nan

            if len(base_vals) >= 3 and len(tgt_vals) >= 3:
                u_stat, p_value = sp_stats.mannwhitneyu(
                    base_vals, tgt_vals, alternative='two-sided'
                )
                sig = "significant *" if p_value < 0.05 else "not significant"
            else:
                p_value = np.nan
                sig = "N too small"

            # Bootstrap CI for target mean
            if len(tgt_vals) >= 3:
                ci_lo, ci_hi = bootstrap_ci(tgt_vals)
                ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
            else:
                ci_str = "---"

            print(f"{label:<22} {base_mean:>9.2f}% {tgt_mean:>9.2f}% "
                  f"{relative_change:>+9.1f}% {p_value:>10.4f} {sig:>14}")

            all_results.append({
                'Phase': target_phase,
                'Distance_Band': label,
                'Baseline_Mean_Pct': round(base_mean, 4),
                'Target_Mean_Pct': round(tgt_mean, 4),
                'Relative_Change_Pct': round(relative_change, 2) if not np.isnan(relative_change) else None,
                'Mann_Whitney_P': round(p_value, 6) if not np.isnan(p_value) else None,
                'Target_95CI': ci_str,
                'Significance': sig,
            })

    print("\n" + "=" * 80)

    print("\n--- Evaluation of aggregated long-range ratio (200km+) ---")
    df['ratio_fringe'] = (df['dist_200_300'] + df['dist_300_plus']) / df['total'] * 100

    for target_phase in phases[1:]:
        base_fringe = baseline_data['ratio_fringe' if 'ratio_fringe' in baseline_data else 'fringe_ratio'].values
        # Re-calculate from df
        tgt_mask = df['phase_label'] == target_phase
        base_mask = df['phase_label'] == baseline_phase
        base_fringe = df.loc[base_mask, 'ratio_fringe'].values
        tgt_fringe = df.loc[tgt_mask, 'ratio_fringe'].values

        u, p = sp_stats.mannwhitneyu(base_fringe, tgt_fringe, alternative='two-sided')
        base_ci = bootstrap_ci(base_fringe)
        tgt_ci = bootstrap_ci(tgt_fringe)

        print(f"  {baseline_phase}: {np.mean(base_fringe):.2f}% (95%CI [{base_ci[0]:.2f}, {base_ci[1]:.2f}])")
        print(f"  {target_phase}:  {np.mean(tgt_fringe):.2f}% (95%CI [{tgt_ci[0]:.2f}, {tgt_ci[1]:.2f}])")
        print(f"  Mann-Whitney P = {p:.6f}  → {'significant' if p < 0.05 else 'not significant'}")

    res_df = pd.DataFrame(all_results)
    save_path = os.path.join(OUTPUT_DIR, "distance_performance_summary.csv")
    res_df.to_csv(save_path, index=False)
    print(f"\n  Saved results: {save_path}")


if __name__ == "__main__":
    run_distance_analysis()
