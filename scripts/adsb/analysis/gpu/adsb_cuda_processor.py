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
from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.paths import OUTPUT_DIR
from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.nb2_models import clean_nb2_df, prepare_nb2_inputs, make_single_change_point_model

CPU_HOST = min(6, os.cpu_count() or 6)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(CPU_HOST)


def run_cuda_analysis():
    min_auc, _min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=None, require_proxy=True)
    if df is None or df.empty:
        print(" Error: input data is missing.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df = clean_nb2_df(df, require_log_traffic=True)
    if len(df) < 5:
        print("  WARNING: Insufficient valid data. Skipping change-point estimation.")
        return

    inputs = prepare_nb2_inputs(df)
    n_days = inputs.n_days
    model = make_single_change_point_model(alpha_name="phi")

    kernel = DiscreteHMCGibbs(NUTS(model))
    if os.getenv("MCMC_FULL") == "1":
        num_warmup, num_samples = 1000, 2000
    else:
        num_warmup, num_samples = 50, 200
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)

    print(f">>> Running Bayesian inference (CPU, warmup={num_warmup}, samples={num_samples})...")
    mcmc.run(random.PRNGKey(42), inputs.y, inputs.log_traffic, inputs.n_days)

    samples = mcmc.get_samples()
    tau_samples = samples["tau"]
    vals, counts = jnp.unique(tau_samples, return_counts=True)
    best_tau_idx = int(vals[jnp.argmax(counts)])
    detected_date = df.iloc[best_tau_idx]["date"]

    print("\n" + "=" * 40)
    print(f" Estimated change point: {detected_date}")
    print("=" * 40)

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
