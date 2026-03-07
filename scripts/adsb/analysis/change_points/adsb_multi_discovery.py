import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.paths import ADSB_DAILY_SUMMARY

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
numpyro.set_platform("cpu")
numpyro.set_host_device_count(min(6, os.cpu_count() or 6))

def run_clean_discovery():
    input_file = str(ADSB_DAILY_SUMMARY)
    if not os.path.exists(input_file): return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f">>> Exclusion: first day ({min_date.strftime('%Y-%m-%d')})  and last day ({max_date.strftime('%Y-%m-%d')})  will be excluded.")
    df = df[(df['date'] > min_date) & (df['date'] < max_date)].copy()

    df = df[df['auc_n_used'] > 5000].reset_index(drop=True)
    if len(df) < 10:
        print(" Not enough days for analysis.")
        return

    y = jnp.array(df['auc_n_used'].values, dtype=jnp.float32)
    n_days = len(df)
    K = 3

    def model(y, n_days, K):
        taus = numpyro.sample('taus', dist.DiscreteUniform(2, n_days - 3).expand([K]))
        alphas = numpyro.sample('alphas', dist.Normal(10., 5.).expand([K + 1]))
        phi = numpyro.sample('phi', dist.Exponential(0.1))

        idx = jnp.arange(n_days)[:, None]
        sorted_taus = jnp.sort(taus)
        phase_idx = jnp.sum(idx >= sorted_taus, axis=-1)

        mu = jnp.exp(alphas[phase_idx])
        numpyro.sample('y_obs', dist.NegativeBinomial2(mu, phi), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1500, num_samples=3000, num_chains=1)

    print("\n>>> Running inference algorithm...")
    mcmc.run(random.PRNGKey(42), y, n_days, K)

    samples = mcmc.get_samples()
    tau_samples = jnp.sort(samples['taus'], axis=-1)

    unique_indices = []
    for i in range(K):
        vals, counts = np.unique(tau_samples[:, i], return_counts=True)
        idx = int(vals[np.argmax(counts)])
        if idx > 0:
            unique_indices.append(idx)

    unique_indices = sorted(list(set(unique_indices)))

    print("\n" + "="*80)
    print(f"{'Phase':<10} | {'Estimated start':<12} | {'Avg aircraft/day (est.)':<20} | {'Improvement'}")
    print("-" * 80)

    alphas_samples = samples['alphas']
    for i in range(len(unique_indices) + 1):
        phase_mean = np.mean(np.exp(alphas_samples[:, i]))
        start_date = df.iloc[unique_indices[i-1]]['date'] if i > 0 else df.iloc[0]['date']

        if i == 0:
            imp_str = "Baseline"
        else:
            prev_mean = np.mean(np.exp(alphas_samples[:, i-1]))
            improvement = ((phase_mean / prev_mean) - 1) * 100
            imp_str = f"{improvement:+.1f}%"

        print(f"Phase {i+1:<2} | {start_date.strftime('%Y-%m-%d'):<12} | {int(phase_mean):>15,} aircraft | {imp_str}")
    print("="*80)

if __name__ == "__main__":
    run_clean_discovery()
