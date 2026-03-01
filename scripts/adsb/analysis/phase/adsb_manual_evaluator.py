import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt


from arena.lib.input_utils import prompt_phase_dates
from arena.lib.paths import ADSB_DAILY_SUMMARY

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

def get_user_phases():
    from arena.lib.phase_config import get_config as _get_cfg
    _cfg = _get_cfg()
    default_labels = {d: n for d, n in _cfg.phase_dates}
    phases = prompt_phase_dates(default_labels)
    # prompt_phase_dates uses "Initial Baseline"; align label
    phases[0]["name"] = "Baseline"
    return phases

def run_manual_eval():
    input_file = str(ADSB_DAILY_SUMMARY)
    if not os.path.exists(input_file): return
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    median_val = df['auc_n_used'].median()
    df.loc[df['auc_n_used'] > 1000000, 'auc_n_used'] = median_val

    df = df[df['date'] < df['date'].max()].copy()

    phases = get_user_phases()
    df['phase_idx'] = -1
    for i, p in enumerate(phases):
        df.loc[df['date'] >= pd.Timestamp(p['date']), 'phase_idx'] = i

    df = df[df['phase_idx'] >= 0].reset_index(drop=True)
    y = jnp.array(df['auc_n_used'].values, dtype=jnp.float32)
    phase_idx = jnp.array(df['phase_idx'].values)
    num_phases = len(phases)

    def model(y, phase_idx, num_phases):
        alphas = numpyro.sample('alphas', dist.Normal(10., 5.).expand([num_phases]))
        phi = numpyro.sample('phi', dist.Exponential(0.1))
        mu = jnp.exp(alphas[phase_idx])
        numpyro.sample('y_obs', dist.NegativeBinomial2(mu, phi), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=1)
    mcmc.run(random.PRNGKey(42), y, phase_idx, num_phases)

    samples = mcmc.get_samples()
    alphas = samples['alphas']

    print("\n" + "="*80)
    print(f"{'Phase':<20} | {'Start date':<12} | {'Avg aircraft/day':<18} | {'Improvement'}")
    print("-" * 80)
    for i in range(num_phases):
        p_mean = np.mean(np.exp(alphas[:, i]))
        if i == 0: imp = "---"
        else:
            prev_mean = np.mean(np.exp(alphas[:, i-1]))
            imp = f"{((p_mean / prev_mean) - 1) * 100:+.1f}%"
        print(f"{phases[i]['name']:<20} | {phases[i]['date']:<12} | {int(p_mean):>12,} | {imp}")
    print("="*80)

if __name__ == "__main__":
    run_manual_eval()
