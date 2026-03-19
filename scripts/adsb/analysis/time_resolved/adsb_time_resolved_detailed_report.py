import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from arena.lib.paths import OUTPUT_DIR

from arena.log import get_script_logger


log = get_script_logger(__name__)
CSV_PATH = os.path.join(str(OUTPUT_DIR), "time_resolved", "adsb_timebin_summary.csv")
OUTPUT_DIR = str(Path(OUTPUT_DIR) / "performance")

def generate_detailed_report():
    if not os.path.exists(CSV_PATH):
        log.info("❌ データが見つかりません。")
        return

    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_data = []
    bins = sorted(df['time_bin'].unique())
    
    log.info(f"{'時間帯':<10} | {'旧平均':>10} | {'新平均':>10} | {'変化率%':>10} | {'P値'}")
    log.info("-" * 65)

    for b in bins:
        group1 = df[(df['time_bin'] == b) & (df['post'] == 0)]['auc_sum']
        group2 = df[(df['time_bin'] == b) & (df['post'] == 1)]['auc_sum']
        
        m1, m2 = group1.mean(), group2.mean()
        if m1 == 0 or len(group1) == 0:
            log.info(f"  [WARN] 時間帯 {b}: pre-intervention データが空またはゼロです。変化率を計算できません。")
            change = float('nan')
        else:
            change = ((m2 / m1) - 1) * 100

        if len(group1) < 2 or len(group2) < 2:
            log.info(f"  [WARN] 時間帯 {b}: サンプル数が不足しています (pre={len(group1)}, post={len(group2)})。")
            p_val = float('nan')
            t_stat = float('nan')
        else:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        p_str = f"{p_val:.4f}" if not (isinstance(p_val, float) and p_val != p_val) else "N/A"
        change_str = f"{change:>+9.2f}%" if not (isinstance(change, float) and change != change) else "     N/A%"
        log.info(f"{b:<10} | {m1:>10.1f} | {m2:>10.1f} | {change_str} | {p_str}")
        
        report_data.append({
            'time_bin': b, 'old_mean': m1, 'new_mean': m2, 'improvement_pct': change, 'p_value': p_val
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(OUTPUT_DIR, "time_bin_detailed_stats.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # join= was removed in seaborn 0.13; use linestyle='none' instead
    sns.pointplot(data=df, x='time_bin', y='auc_sum', hue='post',
                  dodge=True, linestyle='none', capsize=.1, palette="Set1", ax=ax1)
    ax1.set_title("AUC Raw Performance Comparison by Time Bin", fontsize=14)
    ax1.set_ylabel("Mean AUC (n_used sum)")
    ax1.legend(title="Phase", labels=["Old (Before)", "New (After)"])
    ax1.grid(True, alpha=0.3)

    # palette= with a plain list is deprecated in seaborn 0.13+; use hue + palette dict
    report_df['bar_color'] = ['positive' if x > 0 else 'negative' for x in report_df['improvement_pct']]
    sns.barplot(data=report_df, x='time_bin', y='improvement_pct',
                hue='bar_color', palette={'positive': '#2ecc71', 'negative': '#e74c3c'},
                legend=False, ax=ax2)

    for i, p in enumerate(report_df['p_value']):
        if isinstance(p, float) and not (p != p) and p < 0.05:
            imp = report_df['improvement_pct'].iloc[i]
            if isinstance(imp, float) and not (imp != imp):
                ax2.text(i, imp, '★', ha='center', va='bottom', fontsize=15, color='gold')

    ax2.set_title("Net Improvement Rate (%) per Time Bin", fontsize=14)
    ax2.set_ylabel("Improvement Rate (%)")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel("Time Bin (JST)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "time_resolved_detailed_plot.png")
    plt.savefig(img_path)
    log.info(f"\n✅ 詳細レポート図を保存しました: {img_path}")

if __name__ == "__main__":
    generate_detailed_report()
