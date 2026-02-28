"""
adsb_baseline_nb_eval.py module.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary, check_proxy_endogeneity
from arena.lib.paths import OUTPUT_DIR as OUT_ROOT

OUTPUT_DIR = str(OUT_ROOT / "performance")


def run_baseline_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return

    check_proxy_endogeneity(df)

    n = len(df)
    print(f"\n  Samples: {n}  days")
    print(f"  Parameters: Intercept + post + log_traffic")
    print(f"  Residual DOF: {n - 3}")

    formula = "auc_n_used ~ post + log_traffic"

    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()
    except Exception as e:
        print(f"  Model estimation error: {e}")
        return

    print("\n" + "=" * 70)
    print("      ADS-B Baseline NB Regression Report")
    print("=" * 70)
    print(model.summary())
    print("=" * 70)

    results = []
    if 'post' in model.params:
        coef = model.params['post']
        p_value = model.pvalues['post']
        ci = model.conf_int().loc['post']

        improvement = (np.exp(coef) - 1) * 100
        ci_lower = (np.exp(ci[0]) - 1) * 100
        ci_upper = (np.exp(ci[1]) - 1) * 100
        sig = "significant" if p_value < 0.05 else "not significant"

        results.append({
            'term': 'post',
            'improvement_pct': round(improvement, 2),
            'ci_lower': round(ci_lower, 2),
            'ci_upper': round(ci_upper, 2),
            'p_value': round(p_value, 6),
            'significance': sig
        })

        print(f"\n  post effect:")
        print(f"    improvement rate: {improvement:+.2f}%  95%CI [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
        print(f"    p-value: {p_value:.6f}  → {sig}")

    if 'log_traffic' in model.params:
        elasticity = model.params['log_traffic']
        p_traffic = model.pvalues['log_traffic']
        print(f"\n  Traffic elasticity: {elasticity:.4f}  (P={p_traffic:.4f})")

    print("\n" + "=" * 70)

    summary_path = os.path.join(OUTPUT_DIR, "baseline_nb_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(model.summary().as_text())
        f.write("\n\n--- Post Effect ---\n")
        for r in results:
            f.write(f"post: {r['improvement_pct']:+.2f}% "
                    f"[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}] "
                    f"P={r['p_value']:.6f}\n")
    print(f"  Text report: {summary_path}")

    json_path = os.path.join(OUTPUT_DIR, "baseline_nb_results.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump({
            'model': 'NegativeBinomial GLM',
            'n_observations': len(df),
            'df_residual': int(model.df_resid),
            'post_effect': results,
            'traffic_elasticity': round(elasticity, 4) if 'log_traffic' in model.params else None,
        }, f, indent=2, ensure_ascii=False)
    print(f"  JSON result: {json_path}")


if __name__ == "__main__":
    run_baseline_analysis()
