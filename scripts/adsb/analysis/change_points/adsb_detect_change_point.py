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
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.nb2_models import clean_nb2_df, prepare_nb2_inputs, make_single_change_point_model

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
    df = clean_nb2_df(df, require_log_traffic=True)
    if len(df) < 5:
        print("  警告: 有効データが不足しているため、変化点検出をスキップします。")
        return
    inputs = prepare_nb2_inputs(df)
    n_days = inputs.n_days
    model = make_single_change_point_model(alpha_name="alpha_inv")

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=3000, num_chains=1)

    print("Searching change points on CPU...")
    mcmc.run(random.PRNGKey(42), inputs.y, inputs.log_traffic, inputs.n_days)

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
