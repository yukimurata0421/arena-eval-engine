"""
adsb_detect_multi_change_points.py module.
"""
import os
import sys
from pathlib import Path

# ============================================================
# ============================================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
print(f">>> Platform: CPU ({jax.devices()})")


def run_multi_discovery_analysis():
    min_auc, _min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=None, require_proxy=True)
    if df is None or df.empty:
        print("Data file not found.")
        return
    df = df.sort_values("date").reset_index(drop=True)

    y = jnp.array(df["auc_n_used"].values, dtype=jnp.float32)
    log_traffic = jnp.array(df["log_traffic"].values, dtype=jnp.float32)
    n_days = len(df)

    K = 3

    def model(y, log_traffic, n_days, K):
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        alpha_inv = numpyro.sample("alpha_inv", dist.Exponential(1.0))
        taus = numpyro.sample(
            "taus", dist.DiscreteUniform(0, n_days - 1).expand([K])
        )
        alphas = numpyro.sample("alphas", dist.Normal(10.0, 5.0).expand([K + 1]))

        idx = jnp.arange(n_days)[:, None]
        phase_idx = jnp.sum(idx >= jnp.sort(taus), axis=-1)

        intercept = alphas[phase_idx]
        mu = jnp.exp(intercept + beta_traffic * log_traffic)

        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, alpha_inv), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1500, num_samples=3000, num_chains=1)

    print(f"\n>>> {K} change points being inferred (CPU)...")
    mcmc.run(random.PRNGKey(0), y, log_traffic, n_days, K)

    samples = mcmc.get_samples()
    tau_samples = jnp.sort(samples["taus"], axis=-1)

    print("\n" + "=" * 40)
    print(" Major inferred intervention dates")
    for i in range(K):
        t_vals, t_counts = np.unique(tau_samples[:, i], return_counts=True)
        best_t = t_vals[np.argmax(t_counts)]
        detected_date = df.iloc[int(best_t)]["date"]
        prob = np.max(t_counts) / len(tau_samples) * 100
        print(
            f"Change point {i+1}: {detected_date.strftime('%Y-%m-%d')} (Confidence: {prob:.1f}%)"
        )
    print("=" * 40)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df["date"], df["auc_n_used"], label="Daily AUC", color="gray", alpha=0.4)
    for i in range(K):
        plt.hist(
            df["date"].values[tau_samples[:, i].astype(int)],
            bins=n_days,
            alpha=0.6,
            label=f"Change Point {i+1}",
        )
    plt.title("Detected Multiple Structural Breaks")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["date"], df["auc_n_used"], color="gray", alpha=0.2)
    plt.title("Performance Phase Shift (Traffic Corrected)")
    plt.show()


if __name__ == "__main__":
    run_multi_discovery_analysis()
