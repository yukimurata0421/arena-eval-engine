"""
adsb_phase_evaluator.py (v2 — batch + file output)

Changes v1 → v2:
  1. Enable batch runs from master.py:
     - With ADSB_BATCH_MODE=1, skip interactive input and auto-load from phase_config
     - Output CSV + human-readable TXT
  2. Use platform_setup.py for GPU/CPU auto-selection
  3. Outputs: performance/phase_evaluator_results.csv
             performance/phase_evaluator_report.txt
             performance/phase_evaluator_boxplot.png (batch mode)

Required libraries: jax, numpyro, pandas, numpy, matplotlib
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


from arena.lib.paths import ADSB_DAILY_SUMMARY, OUTPUT_DIR as OUT_ROOT
from arena.lib.phase_config import get_config as _get_cfg
from arena.lib.platform_setup import init_numpyro_platform

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

OUTPUT_DIR = str(OUT_ROOT / "performance")

IS_BATCH = os.environ.get("ADSB_BATCH_MODE") == "1" or (not sys.stdin.isatty())
FORCE_INTERACTIVE = os.environ.get("ADSB_PHASE_INTERACTIVE") == "1"


def _get_phases_from_config():
    """Auto-load phases from phase_config.py (batch mode)."""
    cfg = _get_cfg()
    phases = []
    for ev in cfg.events:
        if ev.hardware:
            phases.append({"date": ev.date, "name": ev.label})
    if not phases:
        phases.append({"date": cfg.post_change_date, "name": "Baseline"})
    return phases


def _get_phases_interactive():
    """Prompt for phases interactively (legacy compatibility)."""
    from arena.lib.input_utils import prompt_phase_dates
    cfg = _get_cfg()
    default_labels = {d: n for d, n in cfg.phase_dates}
    return prompt_phase_dates(default_labels)


def get_phases():
    if FORCE_INTERACTIVE:
        return _get_phases_interactive()
    return _get_phases_from_config()


def run_phase_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_path = str(ADSB_DAILY_SUMMARY)
    if not os.path.exists(data_path):
        print(f"  Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['auc_n_used'] > 5000].copy()

    phases = get_phases()
    if len(phases) < 2:
        print("  Fewer than 2 phases. Cannot compare.")
        return

    df['phase_idx'] = -1
    for i, p in enumerate(phases):
        df.loc[df['date'] >= pd.Timestamp(p['date']), 'phase_idx'] = i
    df = df[df['phase_idx'] >= 0].reset_index(drop=True)

    if len(df) < 5:
        print("  Fewer than 5 valid days.")
        return

    init_numpyro_platform(n_data=len(df))

    df['auc_n_used'] = df['auc_n_used'].fillna(0).clip(lower=0)
    hnd = df['hnd_nrt_movements'].fillna(1).replace(0, 1)
    y = jnp.array(df['auc_n_used'].values, dtype=jnp.float32)
    log_traffic = jnp.array(np.log(hnd.values), dtype=jnp.float32)
    phase_idx = jnp.array(df['phase_idx'].values)
    num_phases = len(phases)

    print(f"  Data: {len(df)} days, phases: {num_phases}")
    for i, p in enumerate(phases):
        n = int((df['phase_idx'] == i).sum())
        print(f"    {p['name']}: {n} days")

    def model(y, log_traffic, phase_idx, num_phases):
        beta_traffic = numpyro.sample('beta_traffic', dist.Normal(1., 0.5))
        alphas = numpyro.sample('alphas', dist.Normal(10., 5.).expand([num_phases]))
        alpha_inv = numpyro.sample('alpha_inv', dist.Exponential(1.0))
        mu = jnp.exp(alphas[phase_idx] + beta_traffic * log_traffic)
        numpyro.sample('y_obs', dist.NegativeBinomial2(mu, alpha_inv), obs=y)

    # --- MCMC ---
    default_warmup = 1000
    default_samples = 2000
    n_warmup = int(os.environ.get("ADSB_PHASE_WARMUP", default_warmup))
    n_samples = int(os.environ.get("ADSB_PHASE_SAMPLES", default_samples))

    n_chains_env = int(os.environ.get("ADSB_PHASE_CHAINS", "0"))
    if n_chains_env > 0:
        n_chains = n_chains_env
    else:
        n_chains = 4

    mcmc = MCMC(NUTS(model), num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    print(f"\n  Running MCMC (warmup={n_warmup}, samples={n_samples}, chains={n_chains})...")
    mcmc.run(random.PRNGKey(42), y, log_traffic, phase_idx, num_phases)

    samples = mcmc.get_samples()
    alphas = samples['alphas']
    beta_traffic = samples['beta_traffic']

    results = []
    lines = []
    lines.append("=" * 80)
    lines.append("  ADS-B Phase Evaluator Report (NegBin + Traffic Control)")
    lines.append("=" * 80)
    lines.append(f"  Data: {len(df)} days, phases: {num_phases}")
    lines.append(f"  MCMC: warmup={n_warmup}, samples={n_samples}, chains={n_chains}")
    lines.append("")
    lines.append(f"  {'Phase Name':<25} | {'N':>4} | {'vs Previous':<18} | {'vs Baseline':<30}")
    lines.append("  " + "-" * 82)

    for i in range(num_phases):
        n_days = int((df['phase_idx'] == i).sum())

        # vs Baseline (Phase 0)
        rel_base = (jnp.exp(alphas[:, i] - alphas[:, 0]) - 1) * 100
        mean_base = float(jnp.mean(rel_base))
        hdi_base_lo = float(jnp.percentile(rel_base, 3.0))
        hdi_base_hi = float(jnp.percentile(rel_base, 97.0))
        prob_base = float((rel_base > 0).mean()) * 100

        # vs Previous
        if i == 0:
            mean_prev = 0.0
            hdi_prev_lo = 0.0
            hdi_prev_hi = 0.0
            prob_prev = 0.0
            vs_prev_str = "---"
        else:
            rel_prev = (jnp.exp(alphas[:, i] - alphas[:, i - 1]) - 1) * 100
            mean_prev = float(jnp.mean(rel_prev))
            hdi_prev_lo = float(jnp.percentile(rel_prev, 3.0))
            hdi_prev_hi = float(jnp.percentile(rel_prev, 97.0))
            prob_prev = float((rel_prev > 0).mean()) * 100
            vs_prev_str = f"{mean_prev:>+7.1f}% P(>0)={prob_prev:.0f}%"

        vs_base_str = f"{mean_base:>+7.1f}% [{hdi_base_lo:>+6.1f}, {hdi_base_hi:>+6.1f}] P(>0)={prob_base:.0f}%"

        lines.append(f"  {phases[i]['name']:<25} | {n_days:>4} | {vs_prev_str:<18} | {vs_base_str}")

        results.append({
            'phase_idx': i,
            'phase_name': phases[i]['name'],
            'phase_date': phases[i]['date'],
            'n_days': n_days,
            'vs_baseline_mean_pct': round(mean_base, 2),
            'vs_baseline_hdi94_lo': round(hdi_base_lo, 2),
            'vs_baseline_hdi94_hi': round(hdi_base_hi, 2),
            'vs_baseline_prob_positive_pct': round(prob_base, 1),
            'vs_previous_mean_pct': round(mean_prev, 2),
            'vs_previous_hdi94_lo': round(hdi_prev_lo, 2),
            'vs_previous_hdi94_hi': round(hdi_prev_hi, 2),
            'vs_previous_prob_positive_pct': round(prob_prev, 1),
        })

    beta_mean = float(jnp.mean(beta_traffic))
    beta_lo = float(jnp.percentile(beta_traffic, 3.0))
    beta_hi = float(jnp.percentile(beta_traffic, 97.0))
    lines.append("")
    lines.append(f"  Traffic elasticity (beta_traffic): {beta_mean:.4f} [94% HDI: {beta_lo:.4f}, {beta_hi:.4f}]")
    lines.append("=" * 80)

    print("\n" + "\n".join(lines))

    csv_path = os.path.join(OUTPUT_DIR, "phase_evaluator_results.csv")
    txt_path = os.path.join(OUTPUT_DIR, "phase_evaluator_report.txt")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Saved results: {csv_path}")
    print(f"  Report:   {txt_path}")

    if not IS_BATCH:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        data_to_plot = [np.array(jnp.exp(alphas[:, i] - alphas[:, 0])) for i in range(num_phases)]
        plt.boxplot(data_to_plot, tick_labels=[p['name'] for p in phases])
        plt.axhline(1.0, color='red', linestyle='--')
        plt.title("Hardware Performance Gain (Corrected Baseline)")
        plt.show()
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [np.array(jnp.exp(alphas[:, i] - alphas[:, 0])) for i in range(num_phases)]
        ax.boxplot(data_to_plot, tick_labels=[p['name'] for p in phases])
        ax.axhline(1.0, color='red', linestyle='--')
        ax.set_title("Hardware Performance Gain (Corrected Baseline)")
        png_path = os.path.join(OUTPUT_DIR, "phase_evaluator_boxplot.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot:   {png_path}")


if __name__ == "__main__":
    run_phase_analysis()
