"""
adsb_bayesian_eval.py module.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.paths import OUTPUT_DIR as OUT_ROOT

OUTPUT_DIR = str(OUT_ROOT / "performance")

from arena.lib.phase_config import get_config as _get_cfg
_cfg = _get_cfg()
PHASE_MAP = _cfg.get_hardware_map()
PHASE_NAMES = _cfg.get_phase_names()


def run_bayesian_analysis():
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("  PyMC is not installed.")
        print("  pip install pymc arviz")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return

    if 'hardware' in df.columns:
        df['phase_idx'] = df['hardware'].map(PHASE_MAP)
    else:
        df['phase_idx'] = 0
        _fallback = _cfg.get_phase_fallback_dates()
        for _i, _d in enumerate(_fallback):
            df.loc[df['date'] >= _d, 'phase_idx'] = _i + 1

    df = df.dropna(subset=['phase_idx']).copy()
    df['phase_idx'] = df['phase_idx'].astype(int)

    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    y = df['auc_n_used'].values.astype(float)
    log_traffic = df['log_traffic'].values.astype(float)
    phase_idx = df['phase_idx'].values
    is_weekend = df['is_weekend'].values.astype(float)
    num_phases = len(df['phase_idx'].unique())

    print(f"  Data: {len(df)} days, phases: {num_phases}")
    for i in range(num_phases):
        n = (phase_idx == i).sum()
        name = PHASE_NAMES.get(i, f"Phase{i}")
        print(f"    {name}: {n} days")

    print("\n  Building PyMC model...")

    n_cores = min(os.cpu_count() or 4, 4)

    with pm.Model() as model:
        alphas = pm.Normal('alphas', mu=10.0, sigma=3.0, shape=num_phases)

        beta_traffic = pm.Normal('beta_traffic', mu=0.5, sigma=0.5)

        beta_weekend = pm.Normal('beta_weekend', mu=0.0, sigma=0.5)

        phi = pm.Exponential('phi', 1.0)

        mu = pm.math.exp(
            alphas[phase_idx]
            + beta_traffic * log_traffic
            + beta_weekend * is_weekend
        )

        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=phi, observed=y)

        print(f"  Running MCMC (chains={n_cores}, cores={n_cores})...")
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=n_cores,
            cores=n_cores,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )

    print("\n" + "=" * 70)
    print("  ADS-B Bayesian Report (PyMC NegativeBinomial)")
    print("=" * 70)

    summary = az.summary(trace, var_names=['alphas', 'beta_traffic', 'beta_weekend', 'phi'])
    print("\n--- Parameter Summary ---")
    print(summary)

    alphas_samples = trace.posterior['alphas'].values
    # shape: (chains, draws, num_phases)
    alphas_flat = alphas_samples.reshape(-1, num_phases)

    print("\n--- Phase-to-Phase Improvement ---")
    print(f"{'Comparison':<35} {'Mean':>8} {'HDI 94%':>20} {'P(>0)':>8}")
    print("-" * 75)

    results = []

    for i in range(1, num_phases):
        # vs baseline (Phase 0)
        diff_vs_base = alphas_flat[:, i] - alphas_flat[:, 0]
        improvement_vs_base = (np.exp(diff_vs_base) - 1) * 100
        mean_imp = np.mean(improvement_vs_base)
        hdi = az.hdi(improvement_vs_base, hdi_prob=0.94)
        prob_positive = (improvement_vs_base > 0).mean() * 100

        name_i = PHASE_NAMES.get(i, f"Phase{i}")
        name_0 = PHASE_NAMES.get(0, "Phase0")
        label = f"{name_i} vs {name_0}"
        print(f"{label:<35} {mean_imp:>+7.1f}% [{hdi[0]:>+7.1f}, {hdi[1]:>+7.1f}]  {prob_positive:>6.1f}%")

        results.append({
            'comparison': label,
            'mean_improvement_pct': round(mean_imp, 2),
            'hdi_94_lower': round(float(hdi[0]), 2),
            'hdi_94_upper': round(float(hdi[1]), 2),
            'prob_positive_pct': round(prob_positive, 1),
        })

        # vs previous phase
        if i > 1:
            diff_vs_prev = alphas_flat[:, i] - alphas_flat[:, i - 1]
            improvement_vs_prev = (np.exp(diff_vs_prev) - 1) * 100
            mean_prev = np.mean(improvement_vs_prev)
            hdi_prev = az.hdi(improvement_vs_prev, hdi_prob=0.94)
            prob_prev = (improvement_vs_prev > 0).mean() * 100

            name_prev = PHASE_NAMES.get(i - 1, f"Phase{i-1}")
            label_prev = f"{name_i} vs {name_prev}"
            print(f"{label_prev:<35} {mean_prev:>+7.1f}% "
                  f"[{hdi_prev[0]:>+7.1f}, {hdi_prev[1]:>+7.1f}]  {prob_prev:>6.1f}%")

            results.append({
                'comparison': label_prev,
                'mean_improvement_pct': round(mean_prev, 2),
                'hdi_94_lower': round(float(hdi_prev[0]), 2),
                'hdi_94_upper': round(float(hdi_prev[1]), 2),
                'prob_positive_pct': round(prob_prev, 1),
            })

    beta_traffic_samples = trace.posterior['beta_traffic'].values.flatten()
    beta_weekend_samples = trace.posterior['beta_weekend'].values.flatten()

    print(f"\n--- Covariate Effects ---")
    print(f"  Traffic elasticity: {np.mean(beta_traffic_samples):.4f} "
          f"(94% HDI: {az.hdi(beta_traffic_samples, hdi_prob=0.94)})")
    weekend_pct = (np.exp(beta_weekend_samples) - 1) * 100
    print(f"  Weekend effect: {np.mean(weekend_pct):+.1f}% "
          f"(94% HDI: [{np.percentile(weekend_pct, 3):.1f}, {np.percentile(weekend_pct, 97):.1f}])")

    print("=" * 70)

    res_df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, "bayesian_phase_results.csv")
    res_df.to_csv(save_path, index=False)
    print(f"\n  Saved results: {save_path}")


if __name__ == "__main__":
    run_bayesian_analysis()
