import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


from arena.lib.config import get_quality_thresholds
from arena.lib.input_utils import prompt_intervention_date
from arena.lib.data_loader import load_summary

def run_bayesian_analysis():
    print(" ADS-B Bayesian evaluation engine (Multi-Core Optimized)")

    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return

    print("\nEnter the configuration change date to evaluate")
    from arena.lib.phase_config import get_config as _get_cfg
    cutoff = prompt_intervention_date(_get_cfg().time_resolved_date)

    df['post'] = (df['date'] >= cutoff).astype(int)
    df['auc_n_used'] = pd.to_numeric(df['auc_n_used'], errors='coerce').fillna(0).clip(lower=0)
    df['log_traffic'] = pd.to_numeric(df['log_traffic'], errors='coerce')
    df = df.dropna(subset=['auc_n_used', 'log_traffic', 'post'])

    y = df['auc_n_used'].values.astype(float)
    post_flag = df['post'].values
    log_traffic = df['log_traffic'].values

    print(f"--- {cutoff.date()} Calculating improvement probability with cutoff at (6-core parallel) ---")

    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=10, sigma=5)
        beta_traffic = pm.Normal('beta_traffic', mu=1, sigma=1)
        gamma = pm.Normal('gamma', mu=0, sigma=1)
        alpha_inv = pm.Exponential('alpha_inv', 1.0)

        mu = pm.math.exp(intercept + beta_traffic * log_traffic + gamma * post_flag)

        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_inv, observed=y)

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=6,
            cores=6,
            random_seed=42,
            return_inferencedata=True
        )

    post_gamma = trace.posterior['gamma'].values.flatten()
    improvement_samples = (np.exp(post_gamma) - 1) * 100
    prob_improved = (improvement_samples > 0).mean() * 100
    mean_improvement = np.mean(improvement_samples)
    hdi_94 = az.hdi(improvement_samples, hdi_prob=0.94)

    print("\n" + "="*50)
    print(f" Bayesian analysis report: {cutoff.date()}")
    print(f"Estimated improvement (mean): {mean_improvement:+.2f} %")
    print(f"94% credible interval (HDI): [{hdi_94[0]:.1f}%, {hdi_94[1]:.1f}%]")
    print(f"� [Probability the change was effective]: {prob_improved:.1f} %")
    print("="*50)

if __name__ == "__main__":
    run_bayesian_analysis()
