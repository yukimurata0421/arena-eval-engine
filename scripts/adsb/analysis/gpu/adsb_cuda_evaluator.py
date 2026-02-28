"""
adsb_cuda_evaluator.py module.
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
import matplotlib.dates as mdates


from arena.lib.paths import OUTPUT_DIR
from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


def run_cuda_analysis():
    min_auc, _min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=None, require_proxy=True)
    if df is None or df.empty:
        print("  Error: input data is missing.")
        return
    df = df.sort_values("date").reset_index(drop=True)

    n_days = len(df)
    dates = df["date"].values

    print(f"  Analysis target: {n_days}  days ({df['date'].min().date()} ~ {df['date'].max().date()})")

    y = jnp.array(df["auc_n_used"].values, dtype=jnp.float32)
    log_traffic = jnp.array(df["log_traffic"].values, dtype=jnp.float32)

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

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
    print(f"  Starting MCMC (warmup={num_warmup}, samples={num_samples}, CPU)...")
    mcmc.run(random.PRNGKey(42), y, log_traffic, n_days)

    samples = mcmc.get_samples()
    tau_samples = np.array(samples["tau"])
    alpha_before_samples = np.array(samples["alpha_before"])
    alpha_after_samples = np.array(samples["alpha_after"])

    vals, counts = np.unique(tau_samples, return_counts=True)
    best_tau_idx = int(vals[np.argmax(counts)])
    detected_date = pd.Timestamp(dates[best_tau_idx])

    improvement_samples = (np.exp(alpha_after_samples - alpha_before_samples) - 1) * 100
    mean_improvement = np.mean(improvement_samples)
    hdi_lo, hdi_hi = np.percentile(improvement_samples, [3, 97])

    print(f"\n  Most likely change point: {detected_date.date()} (index={best_tau_idx})")
    print(f"  Estimated improvement: {mean_improvement:+.1f}% (94% HDI: [{hdi_lo:+.1f}%, {hdi_hi:+.1f}%])")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 2, 2]})

    ax1 = axes[0]
    plot_dates = pd.to_datetime(dates)
    ax1.plot(plot_dates, df["auc_n_used"].values, color="steelblue", alpha=0.7, linewidth=1.5)
    ax1.fill_between(plot_dates, df["auc_n_used"].values, alpha=0.15, color="steelblue")
    ax1.axvline(detected_date, color="red", linestyle="--", linewidth=2,
                label=f"Detected Change: {detected_date.date()}")

    pre_mean = df.loc[df["date"] < detected_date, "auc_n_used"].mean()
    post_mean = df.loc[df["date"] >= detected_date, "auc_n_used"].mean()
    pre_dates = plot_dates[plot_dates < detected_date]
    post_dates = plot_dates[plot_dates >= detected_date]
    if len(pre_dates) > 0:
        ax1.hlines(pre_mean, pre_dates[0], pre_dates[-1],
                   colors="gray", linestyles=":", linewidth=1.5, label=f"Pre mean: {pre_mean:.0f}")
    if len(post_dates) > 0:
        ax1.hlines(post_mean, post_dates[0], post_dates[-1],
                   colors="orange", linestyles=":", linewidth=1.5, label=f"Post mean: {post_mean:.0f}")

    ax1.set_title("ADS-B Performance Change Detection (CUDA)", fontsize=14)
    ax1.set_ylabel("Daily AUC (n_used)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    ax2 = axes[1]
    tau_dates = pd.to_datetime([dates[int(t)] for t in tau_samples])
    ax2.hist(tau_dates, bins=n_days, color="orange", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax2.axvline(detected_date, color="red", linestyle="--", linewidth=2)
    ax2.set_title("Change Point Posterior Distribution", fontsize=12)
    ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    ax3 = axes[2]
    ax3.hist(improvement_samples, bins=60, color="seagreen", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax3.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax3.axvline(mean_improvement, color="red", linestyle="-", linewidth=2,
                label=f"Mean: {mean_improvement:+.1f}%")
    ax3.axvspan(hdi_lo, hdi_hi, alpha=0.15, color="red",
                label=f"94% HDI: [{hdi_lo:+.1f}, {hdi_hi:+.1f}]")
    ax3.set_title("Improvement Rate Posterior", fontsize=12)
    ax3.set_xlabel("Improvement (%)")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=9)

    plt.tight_layout()

    perf_dir = os.path.join(str(OUTPUT_DIR), "performance")
    os.makedirs(perf_dir, exist_ok=True)
    out_path = os.path.join(perf_dir, "adsb_cuda_evaluator_change_point.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    if os.getenv("SHOW_PLOT") == "1":
        plt.show()
    plt.close()


if __name__ == "__main__":
    run_cuda_analysis()
