import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary

def run_analysis():
    print(" ADS-B Statistical Evaluation Engine start(statsmodels 64-bit)")
    
    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return
    
    print("Step: Running negative binomial regression...")
    
    formula = "auc_n_used ~ post + np.log(local_traffic_proxy)"
    
    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()
    except Exception as e:
        print(f" Analysis error: {e}")
        print("Hint: too few days or is_post_change lacks both 0 and 1.")
        return

    print("\n" + "="*60)
    print(" Statistical summary")
    print("="*60)
    print(model.summary())

    gamma = model.params['post']
    p_value = model.pvalues['post']
    
    improvement_rate = (np.exp(gamma) - 1) * 100

    print("\n" + "="*60)
    print(f" Conclusion report (site)")
    print(f"1. Estimated pure improvement rate: {improvement_rate:+.2f} %")
    print(f"2. Statistical confidence (p-value): {p_value:.4f}")
    print("-" * 60)

    if p_value < 0.05:
        if gamma > 0:
            print(" Decision: statistically significant improvement confirmed!")
            print(f"   (Even after accounting for traffic/route variation, capture efficiency {improvement_rate:.1f}% has improved)")
        else:
            print(" Decision: statistically significant degradation detected.")
            print("   (Sensitivity may be too high or noise floor increased)")
    else:
        print(" Decision: no significant change detected.")
        print("   (Observed difference is within random variation; more days are needed)")
    print("="*60)

if __name__ == "__main__":
    run_analysis()
