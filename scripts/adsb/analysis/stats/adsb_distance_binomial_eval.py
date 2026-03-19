"""
adsb_distance_binomial_eval.py module.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

# FitWarning (statsmodels 収束警告) のみ抑制。他の DeprecationWarning は表示する。
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*", category=Warning)


from arena.lib.paths import OUTPUT_DIR as OUT_ROOT

from arena.log import get_script_logger


log = get_script_logger(__name__)
OUTPUT_DIR = str(OUT_ROOT / "performance")
FRINGE_CSV = os.path.join(str(OUT_ROOT / "fringe_decoding"), "fringe_decoding_stats.csv")

PHASES = {
    '1_Old_Settings':   'Phase0_Old',
    '2_New_Filter':     'Phase1_Filter',
    '3_Post_Cable_Fix': 'Phase2_Cable',
}


def two_proportion_z_test(n1, x1, n2, x2):
    """Two-sample proportion z-test (pooled)."""
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_value = 2 * sp_stats.norm.sf(abs(z))
    return z, p_value


def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def run_distance_binomial_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FRINGE_CSV):
        log.info(f"  ファイルが見つかりません: {FRINGE_CSV}")
        return

    df = pd.read_csv(FRINGE_CSV)
    df['date'] = pd.to_datetime(df['date'])

    if 'phase' not in df.columns:
        log.info("  phase 列がありません。")
        return

    df['phase_label'] = df['phase'].map(PHASES).fillna(df['phase'])

    dist_bands = {
        'Near (0-100km)':     'dist_0_100',
        'Mid (100-200km)':    'dist_100_200',
        'Far (200-300km)':    'dist_200_300',
        'Extreme (300km+)':   'dist_300_plus',
    }

    for label, col in dist_bands.items():
        df[f'pct_{col}'] = df[col] / df['total'] * 100

    phases = sorted(df['phase_label'].unique())
    baseline = phases[0]
    df_base = df[df['phase_label'] == baseline]

    log.info("=" * 85)
    log.info("  距離帯比率分析（2項ロジック v2）")
    log.info(f"  ベースライン: {baseline}（{len(df_base)} 日, "
          f"median total={df_base['total'].median():.0f} packets/day)")
    log.info("=" * 85)

    all_results = []

    for target in phases[1:]:
        df_tgt = df[df['phase_label'] == target]

        log.info(f"\n{'='*85}")
        log.info(f"  {target}（{len(df_tgt)} 日, "
              f"median total={df_tgt['total'].median():.0f} packets/day)")
        log.info(f"{'='*85}")

        log.info(f"\n  [A] 日次比率比較（Welch t-test）")
        log.info(f"  {'帯域':<20} {'基準%':>8} {'対象%':>8} {'差分':>8} "
              f"{'t-stat':>8} {'P':>10} {'d':>6} {'Judge':>8}")
        log.info("  " + "-" * 80)

        for label, col in dist_bands.items():
            pct_col = f'pct_{col}'
            base_pcts = df_base[pct_col].values
            tgt_pcts  = df_tgt[pct_col].values

            base_mean = np.mean(base_pcts)
            tgt_mean  = np.mean(tgt_pcts)
            diff = tgt_mean - base_mean

            if len(base_pcts) >= 2 and len(tgt_pcts) >= 2:
                t_stat, p_val = sp_stats.ttest_ind(base_pcts, tgt_pcts, equal_var=False)
                d = cohens_d(base_pcts, tgt_pcts)
                sig = "significant*" if p_val < 0.05 else "---"
            else:
                t_stat, p_val, d = np.nan, np.nan, np.nan
                sig = "N too small"

            log.info(f"  {label:<20} {base_mean:>7.2f}% {tgt_mean:>7.2f}% "
                  f"{diff:>+7.2f}% {t_stat:>8.3f} {p_val:>10.4f} {d:>6.2f} {sig:>8}")

            all_results.append({
                'Comparison': f'{target} vs {baseline}',
                'Band': label,
                'Test': 'Welch_t',
                'Base_Mean_Pct': round(base_mean, 4),
                'Target_Mean_Pct': round(tgt_mean, 4),
                'Diff_Pct': round(diff, 4),
                'Statistic': round(t_stat, 4) if not np.isnan(t_stat) else None,
                'P_Value': round(p_val, 6) if not np.isnan(p_val) else None,
                'Effect_Size_d': round(d, 3) if not np.isnan(d) else None,
                'Significance': sig,
            })

        log.info(f"\n  [B] 全体比率 z 検定（日単位プール）")
        log.info(f"  {'帯域':<20} {'基準率':>10} {'対象率':>10} {'z':>8} {'P':>12}")
        log.info("  " + "-" * 65)

        for label, col in dist_bands.items():
            n_base = df_base['total'].sum()
            x_base = df_base[col].sum()
            n_tgt  = df_tgt['total'].sum()
            x_tgt  = df_tgt[col].sum()

            rate_base = x_base / n_base * 100
            rate_tgt  = x_tgt / n_tgt * 100
            z, p_z = two_proportion_z_test(n_base, x_base, n_tgt, x_tgt)

            log.info(f"  {label:<20} {rate_base:>9.3f}% {rate_tgt:>9.3f}% "
                  f"{z:>8.3f} {p_z:>12.2e}")

            all_results.append({
                'Comparison': f'{target} vs {baseline}',
                'Band': label,
                'Test': 'z_proportion',
                'Base_Mean_Pct': round(rate_base, 4),
                'Target_Mean_Pct': round(rate_tgt, 4),
                'Diff_Pct': round(rate_tgt - rate_base, 4),
                'Statistic': round(z, 4),
                'P_Value': float(f"{p_z:.6e}"),
                'Effect_Size_d': None,
                'Significance': "significant*" if p_z < 0.05 else "---",
            })

    log.info(f"\n  注: 集計 z 検定は N が大きいと有意になりやすいです。")
    log.info(f"    日次比率の t 検定（A）が保守的で信頼できます。")

    res_df = pd.DataFrame(all_results)
    save_path = os.path.join(OUTPUT_DIR, "distance_binomial_summary.csv")
    res_df.to_csv(save_path, index=False)
    log.info(f"\n  結果を保存しました: {save_path}")


if __name__ == "__main__":
    run_distance_binomial_analysis()
