import os
import sys
from pathlib import Path

def solve_gpu_initialization():
    if os.name == 'nt':
        import site
        venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
        paths = [
            os.path.join(venv_site, 'nvidia', 'cublas', 'bin'),
            os.path.join(venv_site, 'nvidia', 'cudnn', 'bin'),
            os.path.join(venv_site, 'nvidia', 'cuda_nvcc', 'bin'),
            os.path.join(venv_site, 'nvidia', 'cuda_nvcc', 'nvvm', 'bin')
        ]
        for p in paths:
            if os.path.exists(p):
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        nvvm_path = os.path.join(venv_site, 'nvidia', 'cuda_nvcc')
        if os.path.exists(nvvm_path):
            os.environ["XLA_FLAGS"] = f'--xla_gpu_cuda_data_dir="{nvvm_path}"'
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        # WSL/Linux: nvidia-* wheels install libs under site-packages/nvidia/*/lib
        try:
            import site
            lib_paths = []
            bin_paths = []
            for base in site.getsitepackages():
                nvidia_base = os.path.join(base, "nvidia")
                if not os.path.isdir(nvidia_base):
                    continue
                for name in os.listdir(nvidia_base):
                    p = os.path.join(nvidia_base, name, "lib")
                    if os.path.isdir(p):
                        lib_paths.append(p)
                    b = os.path.join(nvidia_base, name, "bin")
                    if os.path.isdir(b):
                        bin_paths.append(b)
            if lib_paths:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([existing] if existing else []))
            if bin_paths:
                existing = os.environ.get("PATH", "")
                os.environ["PATH"] = ":".join(bin_paths + ([existing] if existing else []))
                os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + bin_paths[0]
        except Exception:
            pass

solve_gpu_initialization()
# ---------------------------------------

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt


from arena.lib.paths import ADSB_DAILY_SUMMARY

try:
    gpu_devices = jax.devices("cuda")
    print(f" GTX 1060 initialized: {gpu_devices}")
    numpyro.set_platform("cuda")
except Exception as e:
    print(f" GPU detection error: {e}\nRunning in CPU mode.")
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

def normalize_path(path):
    if os.name != "nt" and len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path

def run_multi_point_analysis():
    input_file = normalize_path(str(ADSB_DAILY_SUMMARY))
    if not os.path.exists(input_file): return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    y = jnp.array(df['auc_n_used'].values, dtype=jnp.float32)
    n_days = len(df)

    K = 3

    def model(y, n_days, K):
        taus = numpyro.sample('taus', dist.DiscreteUniform(0, n_days - 1).expand([K]))
        alphas = numpyro.sample('alphas', dist.Normal(10., 5.).expand([K + 1]))
        phi = numpyro.sample('phi', dist.Exponential(1.0))

        idx = jnp.arange(n_days)[:, None]
        phase_idx = jnp.sum(idx >= jnp.sort(taus), axis=-1)
        mu = jnp.exp(alphas[phase_idx])

        numpyro.sample('y_obs', dist.NegativeBinomial2(mu, phi), obs=y)

    kernel = DiscreteHMCGibbs(NUTS(model))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=1)

    print(f"\n>>> From all past data, {K}  change points being inferred...")
    mcmc.run(random.PRNGKey(42), y, n_days, K)

    samples = mcmc.get_samples()
    tau_samples = jnp.sort(samples['taus'], axis=-1)

    print("\n" + "="*45)
    print(" Estimated performance change points")
    print("-" * 45)
    for i in range(K):
        t_vals, t_counts = np.unique(tau_samples[:, i], return_counts=True)
        best_t = t_vals[np.argmax(t_counts)]
        detected_date = df.iloc[int(best_t)]['date']
        print(f"Change point {i+1}: {detected_date.strftime('%Y-%m-%d')} (Confidence: {np.max(t_counts)/len(tau_samples)*100:.1f}%)")
    print("="*45)

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['auc_n_used'], color='black', alpha=0.3, label='AUC (Daily Packets)')
    for i in range(K):
        t_vals, t_counts = np.unique(tau_samples[:, i], return_counts=True)
        best_t = t_vals[np.argmax(t_counts)]
        plt.axvline(df.iloc[int(best_t)]['date'], linestyle='--', label=f'Point {i+1}')
    plt.title("ADSB Multi-Point Change Detection")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_multi_point_analysis()
