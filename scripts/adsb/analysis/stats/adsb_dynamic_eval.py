import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


from arena.lib.config import get_quality_thresholds
from arena.lib.input_utils import prompt_intervention_date
from arena.lib.data_loader import load_summary
from arena.log import get_script_logger

log = get_script_logger(__name__)

def run_analysis():
    log.info("ADS-B Dynamic Evaluation Engine")
    
    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return

    log.info("\nEnter the intervention date to evaluate")
    from arena.lib.phase_config import get_config as _get_cfg
    cutoff = prompt_intervention_date(_get_cfg().post_change_date)

    df['post'] = (df['date'] >= cutoff).astype(int)


    log.info(f"--- {cutoff.date()} Estimating effect with cutoff at ---")
    
    formula = "auc_n_used ~ post + np.log(local_traffic_proxy)"
    
    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()
    except Exception as e:
        log.info(f" parsing error: {e}")
        return

    if 'post' not in model.params:
        log.info(" [WARN] 'post' does not exist in model parameter."
              "There may be no pre/post fluctuations in the data. Analysis will be stopped.")
        return
    gamma = model.params['post']
    p_value = model.pvalues['post']
    improvement_rate = (np.exp(gamma) - 1) * 100

    log.info("\n" + "="*50)
    log.info(f" Parse target: {cutoff.date()}")
    log.info(f"Estimated pure improvement rate: {improvement_rate:+.2f} %")
    log.info(f"Statistical confidence (p-value)   : {p_value:.4f}")
    log.info("="*50)

    if p_value < 0.05:
        res = "[Significant difference]"
        status = " Performance change is statistically supported."
    else:
        res = "[No significant difference]"
        status = " Observed difference is within noise/traffic variation."
    
    log.info(f"{res}\n{status}")

if __name__ == "__main__":
    run_analysis()
