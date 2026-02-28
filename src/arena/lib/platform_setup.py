"""
platform_setup.py — unified platform setup

Consolidates setup_cuda_environment() that was duplicated across 3 files and
addresses the "CPU is much faster for small datasets" issue.

Measured results:
  GPU (GTX 1060) Bayesian phase comparison: 658s → CPU: 17s (38x)
  GPU (GTX 1060) change points K=3:         748s → CPU: 15s (50x)

Cause: with n=59, running DiscreteHMCGibbs on GPU is dominated by kernel launch
      and transfer overhead.

Usage:
  from platform_setup import init_numpyro_platform
  init_numpyro_platform(n_data=len(df))
"""

import os
import sys

# GPU_THRESHOLD: force CPU when data size is below this
# Heuristic: on GTX 1060, DiscreteHMCGibbs beats CPU around n > 10,000
GPU_THRESHOLD = 5000

# CPU parallel chains (i7-8700K = 6C/12T → 4-6 is optimal)
CPU_HOST_DEVICE_COUNT = min(6, os.cpu_count() or 4)


def _link_nvidia_dlls():
    """Add NVIDIA DLL paths on Windows/WSL."""
    if os.name == "nt":
        venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
        nvcc_base = os.path.join(venv_site, "nvidia", "cuda_nvcc")
        paths = [
            os.path.join(venv_site, "nvidia", "cublas", "bin"),
            os.path.join(venv_site, "nvidia", "cudnn", "bin"),
            os.path.join(nvcc_base, "bin"),
            os.path.join(nvcc_base, "nvvm", "bin"),
        ]
        for p in paths:
            if os.path.exists(p):
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if os.path.exists(nvcc_base):
            os.environ["XLA_FLAGS"] = f'--xla_gpu_cuda_data_dir="{nvcc_base}"'
    else:
        try:
            import site
            lib_paths, bin_paths = [], []
            nvcc_base = None
            for base in site.getsitepackages():
                nvidia_base = os.path.join(base, "nvidia")
                if not os.path.isdir(nvidia_base):
                    continue
                for name in os.listdir(nvidia_base):
                    if name == "cuda_nvcc":
                        nvcc_base = os.path.join(nvidia_base, name)
                    lib_p = os.path.join(nvidia_base, name, "lib")
                    if os.path.isdir(lib_p):
                        lib_paths.append(lib_p)
                    bin_p = os.path.join(nvidia_base, name, "bin")
                    if os.path.isdir(bin_p):
                        bin_paths.append(bin_p)
                    nvvm_p = os.path.join(nvidia_base, name, "nvvm", "bin")
                    if os.path.isdir(nvvm_p):
                        bin_paths.append(nvvm_p)
            if lib_paths:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(
                    lib_paths + ([existing] if existing else [])
                )
            if bin_paths:
                existing = os.environ.get("PATH", "")
                os.environ["PATH"] = ":".join(
                    bin_paths + ([existing] if existing else [])
                )
            if nvcc_base and os.path.isdir(nvcc_base):
                os.environ["XLA_FLAGS"] = (
                    "--xla_gpu_cuda_data_dir=" + nvcc_base
                )
        except Exception:
            pass

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def init_numpyro_platform(n_data: int = 0, force_cpu: bool = False):
    """
    Auto-select NumPyro platform.

    Args:
        n_data: number of data points. Force CPU if <= GPU_THRESHOLD.
        force_cpu: if True, force CPU.

    Returns:
        str: selected platform ("cpu" or "cuda")
    """
    import numpyro

    use_cpu = force_cpu or (0 < n_data <= GPU_THRESHOLD)

    if use_cpu:
        reason = "force_cpu=True" if force_cpu else f"n={n_data} <= {GPU_THRESHOLD}"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(CPU_HOST_DEVICE_COUNT)
        print(f"  Platform: CPU ({CPU_HOST_DEVICE_COUNT} devices) [{reason}]")
        return "cpu"

    # Try GPU
    _link_nvidia_dlls()
    try:
        import jax

        numpyro.set_platform("cuda")
        devs = jax.devices("cuda")
        if not devs:
            raise RuntimeError("No CUDA devices")
        print(f"  Platform: CUDA ({devs})")
        return "cuda"
    except Exception as e:
        print(f"  CUDA unavailable ({e}), falling back to CPU")
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(CPU_HOST_DEVICE_COUNT)
        return "cpu"
