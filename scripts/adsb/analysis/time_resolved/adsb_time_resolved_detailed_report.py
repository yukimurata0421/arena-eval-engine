import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from arena.lib.paths import OUTPUT_DIR

CSV_PATH = os.path.join(str(OUTPUT_DIR), "time_resolved", "adsb_timebin_summary.csv")
OUTPUT_DIR = str(Path(OUTPUT_DIR) / "performance")

def generate_detailed_report():
    if not os.path.exists(CSV_PATH):
        print("❌ Data not found.")
        return

    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_data = []
    bins = sorted(df['time_bin'].unique())

    print(f"{'Time Bin':<10} | {'Old Mean':>10} | {'New Mean':>10} | {'Change %':>10} | {'P-Value'}")
    print("-" * 65)

    for b in bins:
        group1 = df[(df['time_bin'] == b) & (df['post'] == 0)]['auc_sum']
        group2 = df[(df['time_bin'] == b) & (df['post'] == 1)]['auc_sum']

        m1, m2 = group1.mean(), group2.mean()
        change = ((m2 / m1) - 1) * 100

        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

        print(f"{b:<10} | {m1:>10.1f} | {m2:>10.1f} | {change:>+9.2f}% | {p_val:.4f}")

        report_data.append({
            'time_bin': b, 'old_mean': m1, 'new_mean': m2, 'improvement_pct': change, 'p_value': p_val
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(OUTPUT_DIR, "time_bin_detailed_stats.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.pointplot(data=df, x='time_bin', y='auc_sum', hue='post',
                  dodge=True, join=False, capsize=.1, palette="Set1", ax=ax1)
    ax1.set_title("AUC Raw Performance Comparison by Time Bin", fontsize=14)
    ax1.set_ylabel("Mean AUC (n_used sum)")
    ax1.legend(title="Phase", labels=["Old (Before)", "New (After)"])
    ax1.grid(True, alpha=0.3)

    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in report_df['improvement_pct']]
    sns.barplot(data=report_df, x='time_bin', y='improvement_pct', palette=colors, ax=ax2)

    for i, p in enumerate(report_df['p_value']):
        if p < 0.05:
            ax2.text(i, report_df['improvement_pct'][i], '★', ha='center', va='bottom', fontsize=15, color='gold')

    ax2.set_title("Net Improvement Rate (%) per Time Bin", fontsize=14)
    ax2.set_ylabel("Improvement Rate (%)")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel("Time Bin (JST)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "time_resolved_detailed_plot.png")
    plt.savefig(img_path)
    print(f"\n✅ Saved detailed report figure: {img_path}")

if __name__ == "__main__":
    generate_detailed_report()
