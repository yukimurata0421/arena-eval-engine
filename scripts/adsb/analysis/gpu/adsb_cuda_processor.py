"""
adsb_cuda_processor.py module.
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


from arena.lib.paths import OUTPUT_DIR
from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.platform_setup import resolve_workers

from arena.log import get_script_logger



log = get_script_logger(__name__)
CPU_HOST = resolve_workers(default_cap=12)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(CPU_HOST)


def run_cuda_analysis():
    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes, require_proxy=True)
    if df is None or df.empty:
        log.info("Error: No input data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["auc_n_used", "log_traffic"])
    if len(df) < 5:
        log.info("Warning: Skipping change point estimation due to lack of valid data.")
        return

    y = jnp.array(df["auc_n_used"].values, dtype=jnp.float32)
    log_traffic = jnp.array(df["log_traffic"].values, dtype=jnp.float32)
    n_days = len(df)
    log.info(
        f">>> Parsed: {n_days} days ({df['date'].min().date()} ~ {df['date'].max().date()})"
        f" [min_auc>{min_auc}, minutes>={min_minutes}]"
    )

    def model(y, log_traffic, n_days):
        tau = numpyro.sample("tau", dist.DiscreteUniform(0, n_days - 1))
        alpha_before = numpyro.sample("alpha_before", dist.Normal(10.0, 5.0))
        alpha_after = numpyro.sample("alpha_after", dist.Normal(10.0, 5.0))
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        phi = numpyro.sample("phi", dist.Exponential(1.0))

        idx = jnp.arange(n_days)
        intercept = jnp.where(idx < tau, alpha_before, alpha_after)
        mu = jnp.exp(intercept + beta_traffic * log_traffic)
        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, phi), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model))
    if os.getenv("MCMC_FULL") == "1":
        num_warmup, num_samples = 1000, 2000
    else:
        num_warmup, num_samples = 50, 200
    n_chains = max(1, min(CPU_HOST, 4))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=n_chains)

    log.info(f">>> Running Bayesian inference (CPU, warmup={num_warmup}, samples={num_samples}, chains={n_chains})...")
    mcmc.run(random.PRNGKey(42), y, log_traffic, n_days)

    samples = mcmc.get_samples()
    tau_samples = samples["tau"]
    vals, counts = jnp.unique(tau_samples, return_counts=True)
    best_tau_idx = int(vals[jnp.argmax(counts)])
    detected_date = df.iloc[best_tau_idx]["date"]

    log.info("\n" + "=" * 40)
    log.info(f" Estimated change point: {detected_date}")
    log.info("=" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df["date"]), df["auc_n_used"], label="Daily Packets")
    plt.axvline(pd.to_datetime(detected_date), color="red", linestyle="--", label="Break Point")
    plt.title("ADSB Performance Change Detection (CUDA)")
    plt.legend()

    out_dir = os.path.join(str(OUTPUT_DIR), "performance")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adsb_cuda_processor_change_point.png")
    plt.savefig(out_path, dpi=150)
    if os.getenv("SHOW_PLOT") == "1":
        plt.show()
    plt.close()


if __name__ == "__main__":
    run_cuda_analysis()
