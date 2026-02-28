import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


from arena.lib.paths import OUTPUT_DIR

CSV_PATH = os.path.join(str(OUTPUT_DIR), "time_resolved", "adsb_timebin_summary.csv")
OUTPUT_DIR = str(Path(OUTPUT_DIR) / "performance")

def run_mixed_analysis():
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found. Run the aggregator first.")
        return

    df = pd.read_csv(CSV_PATH)
    df['date_factor'] = df['date'].astype(str)
    
    df['traffic_proxy'] = df['traffic_proxy'].replace(0, 1)

    print(f">>> Starting mixed model (GEE) analysis (samples: {len(df)})")

    formula = "auc_sum ~ post + C(time_bin) + np.log(traffic_proxy)"
    
    try:
        model = smf.gee(
            formula, 
            data=df, 
            groups=df['date_factor'], 
            family=sm.families.NegativeBinomial(),
            cov_struct=sm.cov_struct.Exchangeable()
        ).fit()

        print("\n" + "="*60)
        print("      ADS-B Time-Resolved Performance Report (GEE-NB)")
        print("="*60)
        print(model.summary().tables[1])
        
        post_coeff = model.params['post']
        p_val = model.pvalues['post']
        improvement = (np.exp(post_coeff) - 1) * 100

        print(f"\n✅ Overall improvement (pure hardware effect): {improvement:+.2f} %")
        print(f"✅ Significance (P-value)          : {p_val:.10e}")
        
        if p_val < 0.05:
            print(f"\nDecision: statistically significant performance improvement confirmed.")
            print("Significant difference remains even after accounting for time-of-day and traffic density.")
        else:
            print("\nDecision: no statistically significant change detected.")
        print("="*60)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df, x='time_bin', y='auc_sum', hue='post', palette="Set2")
        plt.title("AUC by Time Bin: Before vs After (Tsuchiura Station)", fontsize=14)
        plt.xlabel("Time Bin (JST)", fontsize=12)
        plt.ylabel("AUC (Aggregate n_used)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "time_resolved_performance.png")
        plt.savefig(save_path)
        print(f"\nSaved visualization report: {save_path}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    run_mixed_analysis()
