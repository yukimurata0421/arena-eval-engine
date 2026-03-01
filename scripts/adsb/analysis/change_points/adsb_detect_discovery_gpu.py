import os
import sys
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ and os.getenv("FORCE_CUDA") == "1":
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def link_nvidia_dlls():
    if os.name == 'nt':
        import site
        packages = site.getsitepackages()
        if hasattr(site, 'getusersitepackages'):
            packages.append(site.getusersitepackages())
        for s in packages:
            nvidia_base = os.path.join(s, 'nvidia')
            if os.path.exists(nvidia_base):
                for root, dirs, files in os.walk(nvidia_base):
                    if 'bin' in dirs:
                        bin_path = os.path.abspath(os.path.join(root, 'bin'))
                        if os.path.exists(bin_path):
                            os.add_dll_directory(bin_path)

link_nvidia_dlls()

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.nb2_models import clean_nb2_df, prepare_nb2_inputs, make_single_change_point_model

try:
    numpyro.set_platform("cuda")
    print(f">>>  GPU detected: {jax.devices()}")
except:
    print(">>> GPU not detected. Running in CPU parallel mode (4 cores).")
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

def run_discovery_analysis():
    min_auc, _min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=None, require_proxy=True)
    if df is None or df.empty:
        print("Data file not found.")
        return
    df = df.sort_values('date').reset_index(drop=True)
    df = clean_nb2_df(df, require_log_traffic=True)
    if len(df) < 5:
        print("  警告: 有効データが不足しているため、変化点検出をスキップします。")
        return

    inputs = prepare_nb2_inputs(df)
    
    print(f">>> Analysis target: {df['date'].min().date()} ～ {df['date'].max().date()} ({inputs.n_days} days)")

    model = make_single_change_point_model(alpha_name="alpha_inv")

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=1)
    
    print("\n>>> MCMC sampling (inferring change points)...")
    mcmc.run(random.PRNGKey(42), inputs.y, inputs.log_traffic, inputs.n_days)
    
    samples = mcmc.get_samples()
    tau_samples = samples['tau']
    improvement = (jnp.exp(samples['alpha_after'] - samples['alpha_before']) - 1) * 100
    
    vals, counts = np.unique(tau_samples, return_counts=True)
    best_tau_idx = vals[np.argmax(counts)]
    detected_date = df.iloc[int(best_tau_idx)]['date']
    
    print("\n" + "="*40)
    print(f" Analysis complete")
    print(f"[Detected structural change date]: {detected_date.strftime('%Y-%m-%d')}")
    print(f"[Estimated improvement (mean)]: {jnp.mean(improvement):+.2f}%")
    print(f"[Confidence]: {np.max(counts)/len(tau_samples)*100:.1f}%")
    print("="*40)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['date'], df['auc_n_used'], label='AUC', color='black', alpha=0.3)
    plt.axvline(detected_date, color='red', linestyle='--', label='Structural Break')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(df['date'].values[tau_samples.astype(int)], bins=inputs.n_days, color='orange')
    plt.show()

if __name__ == "__main__":
    run_discovery_analysis()
