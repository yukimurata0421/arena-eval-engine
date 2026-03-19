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

# Suppress only statsmodels convergence warnings
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")


from arena.lib.paths import OUTPUT_DIR
from arena.log import get_script_logger

log = get_script_logger(__name__)

CSV_PATH = os.path.join(str(OUTPUT_DIR), "time_resolved", "adsb_timebin_summary.csv")
OUTPUT_DIR = str(Path(OUTPUT_DIR) / "performance")

def run_mixed_analysis():
    if not os.path.exists(CSV_PATH):
        log.info("❌ CSV not found. Please run the aggregator first.")
        return

    df = pd.read_csv(CSV_PATH)
    df['date_factor'] = df['date'].astype(str)
    
    df['traffic_proxy'] = df['traffic_proxy'].replace(0, 1)

    log.info(f">>> Start mixed model (GEE) analysis (number of samples: {len(df)})")

    formula = "auc_sum ~ post + C(time_bin) + np.log(traffic_proxy)"
    
    try:
        model = smf.gee(
            formula, 
            data=df, 
            groups=df['date_factor'], 
            family=sm.families.NegativeBinomial(),
            cov_struct=sm.cov_struct.Exchangeable()
        ).fit()

        log.info("\n" + "="*60)
        log.info("ADS-B Time Resolved Performance Report (GEE-NB)")
        log.info("="*60)
        log.info(model.summary().tables[1])
        
        if 'post' not in model.params:
            log.info(" [WARN] 'post' does not exist in model parameter."
                  "There may be no pre/post variation in the data.")
            return
        post_coeff = model.params['post']
        p_val = model.pvalues['post']
        improvement = (np.exp(post_coeff) - 1) * 100

        log.info(f"\n✅ Overall improvement (pure hardware effect): {improvement:+.2f} %")
        log.info(f"✅ Significance (P value) : {p_val:.10e}")
        
        if p_val < 0.05:
            log.info("\nVerdict: A statistically significant improvement was confirmed.")
            log.info("Even if time of day and traffic density are considered, significant differences remain.")
        else:
            log.info("\nVerdict: No statistically significant changes were detected.")
        log.info("="*60)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df, x='time_bin', y='auc_sum', hue='post', palette="Set2")
        plt.title("AUC by Time Bin: Before vs After (Tsuchiura Station)", fontsize=14)
        plt.xlabel("Time Bin (JST)", fontsize=12)
        plt.ylabel("AUC (Aggregate n_used)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "time_resolved_performance.png")
        plt.savefig(save_path)
        log.info(f"\nVisualization report saved: {save_path}")

    except Exception as e:
        import traceback
        log.info(f"❌ An error occurred while parsing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_mixed_analysis()
