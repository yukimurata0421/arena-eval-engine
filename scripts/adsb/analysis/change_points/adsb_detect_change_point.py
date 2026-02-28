"""
adsb_detect_change_point.py module.
"""
import os
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

import pandas as pd
import numpy as np
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
print(f"Platform: CPU (4 devices)")


def run_discovery_analysis():
    min_auc, _min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=None, require_proxy=True)
    if df is None or df.empty:
        print("No data. Exiting.")
        return
    df = df.sort_values("date").reset_index(drop=True)

    y = jnp.array(df["auc_n_used"].values)
    log_traffic = jnp.array(df["log_traffic"].values)
    n_days = len(df)

    def model(y, log_traffic, n_days):
        tau = numpyro.sample("tau", dist.DiscreteUniform(0, n_days - 1))
        alpha_before = numpyro.sample("alpha_before", dist.Normal(10.0, 5.0))
        alpha_after = numpyro.sample("alpha_after", dist.Normal(10.0, 5.0))
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        alpha_inv = numpyro.sample("alpha_inv", dist.Exponential(1.0))

        idx = jnp.arange(n_days)
        intercept = jnp.where(idx < tau, alpha_before, alpha_after)
        mu = jnp.exp(intercept + beta_traffic * log_traffic)

        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, alpha_inv), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=3000, num_chains=1)

    print("Searching change points on CPU...")
    mcmc.run(random.PRNGKey(42), y, log_traffic, n_days)

    samples = mcmc.get_samples()
    tau_samples = samples["tau"]

    vals, counts = np.unique(tau_samples, return_counts=True)
    best_tau_idx = vals[np.argmax(counts)]
    detected_date = df.iloc[int(best_tau_idx)]["date"]

    improvement = (np.exp(samples["alpha_after"] - samples["alpha_before"]) - 1) * 100

    print("-" * 30)
    print(f"[Detected change point]: {detected_date.strftime('%Y-%m-%d')}")
    print(f"  Estimated improvement: {np.mean(improvement):+.2f}%")
    print(f"  Confidence: {np.max(counts)/len(tau_samples)*100:.1f}%")
    print("-" * 30)

    plt.figure(figsize=(10, 5))
    plt.hist(
        df["date"].values[tau_samples.astype(int)],
        bins=n_days,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Detected Change Point Probability (Tsuchiura Station)")
    plt.xlabel("Date")
    plt.ylabel("Probability Density")
    plt.show()


if __name__ == "__main__":
    run_discovery_analysis()
