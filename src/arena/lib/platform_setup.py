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

from arena.log import get_logger


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def gpu_threshold() -> int:
    # Keep this conservative; GTX 10xx-class GPUs lose on small ARENA daily-N models.
    return _env_int("ADSB_GPU_MIN_N", _env_int("ARENA_GPU_MIN_N", 5000))


GPU_THRESHOLD = gpu_threshold()

logger = get_logger(__name__)


def resolve_workers(default_cap: int = 12) -> int:
    raw = os.environ.get("ADSB_MAX_WORKERS") or os.environ.get("ARENA_MAX_WORKERS") or ""
    cpu = os.cpu_count() or default_cap
    if raw:
        try:
            return max(1, min(int(raw), cpu))
        except ValueError:
            pass
    return max(1, min(default_cap, cpu))


CPU_HOST_DEVICE_COUNT = resolve_workers(default_cap=12)


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
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([existing] if existing else []))
            if bin_paths:
                existing = os.environ.get("PATH", "")
                os.environ["PATH"] = ":".join(bin_paths + ([existing] if existing else []))
            if nvcc_base and os.path.isdir(nvcc_base):
                os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + nvcc_base
        except Exception as exc:
            logger.debug("Failed to link NVIDIA paths on POSIX; falling back to defaults: %s", exc)

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

    force_gpu = _env_bool("ADSB_FORCE_GPU") or _env_bool("ARENA_FORCE_GPU")
    disable_gpu = _env_bool("ADSB_DISABLE_GPU") or _env_bool("ARENA_DISABLE_GPU")
    threshold = gpu_threshold()
    use_cpu = force_cpu or disable_gpu or ((not force_gpu) and 0 < n_data <= threshold)

    if use_cpu:
        if force_cpu:
            reason = "force_cpu=True"
        elif disable_gpu:
            reason = "GPU disabled by env"
        else:
            reason = f"n={n_data} <= {threshold}"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(CPU_HOST_DEVICE_COUNT)
        print(f" Platform: CPU ({CPU_HOST_DEVICE_COUNT} devices) [{reason}]")
        return "cpu"

    # Try GPU
    _link_nvidia_dlls()
    try:
        import jax

        numpyro.set_platform("cuda")
        devs = jax.devices("cuda")
        if not devs:
            raise RuntimeError("No CUDA devices")
        print(f" Platform: CUDA ({devs})")
        return "cuda"
    except Exception as e:
        print(f" CUDA is unavailable ({e}). Falling back to CPU")
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(CPU_HOST_DEVICE_COUNT)
        return "cpu"
