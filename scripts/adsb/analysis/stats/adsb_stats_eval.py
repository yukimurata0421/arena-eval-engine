import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary

from arena.log import get_script_logger


log = get_script_logger(__name__)
def run_analysis():
    log.info(" ADS-B 統計評価エンジン開始 (statsmodels 64-bit)")
    
    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return
    
    log.info("Step: 負の二項回帰を実行中...")
    
    formula = "auc_n_used ~ post + np.log(local_traffic_proxy)"
    
    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()
    except Exception as e:
        log.info(f" 解析エラー: {e}")
        log.info("Hint: too few days or is_post_change lacks both 0 and 1.")
        return

    log.info("\n" + "="*60)
    log.info(" Statistical summary")
    log.info("="*60)
    log.info(model.summary())

    if 'post' not in model.params:
        log.info("  [WARN] モデルパラメータに 'post' が存在しません。"
              " データに pre/post の変動がない可能性があります。解析を中止します。")
        return
    gamma = model.params['post']
    p_value = model.pvalues['post']

    improvement_rate = (np.exp(gamma) - 1) * 100

    log.info("\n" + "="*60)
    log.info(f" Conclusion report (site)")
    log.info(f"1. Estimated pure improvement rate: {improvement_rate:+.2f} %")
    log.info(f"2. Statistical confidence (p-value): {p_value:.4f}")
    log.info("-" * 60)

    if p_value < 0.05:
        if gamma > 0:
            log.info(" Decision: statistically significant improvement confirmed!")
            log.info(f"   (Even after accounting for traffic/route variation, capture efficiency {improvement_rate:.1f}% has improved)")
        else:
            log.info(" Decision: statistically significant degradation detected.")
            log.info("   (Sensitivity may be too high or noise floor increased)")
    else:
        log.info(" Decision: no significant change detected.")
        log.info("   (Observed difference is within random variation; more days are needed)")
    log.info("="*60)

if __name__ == "__main__":
    run_analysis()
