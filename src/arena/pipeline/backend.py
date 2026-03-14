from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

BATCH_ENV = {
    "ADSB_BATCH_MODE": "1",
    "MPLBACKEND": "Agg",  # matplotlib non-interactive backend
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # Prevent UnicodeEncodeError when child scripts print non-CP932 characters on Windows.
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_windows() -> bool:
    return os.name == "nt"


def default_roots_native() -> tuple[Path, Path, Path]:
    """
    Paths accessible from *current Python runtime*.
    - Default layout: <project>/scripts, <project>/output, <project>/data
    """
    from arena.lib.paths import resolve_runtime_roots

    return resolve_runtime_roots()


def _windows_to_wsl_path(p: Path) -> str:
    if os.name != "nt":
        return str(p)
    drive = p.drive[0].lower() if p.drive else "c"
    rest = p.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def default_roots_exec_for_wsl(
    scripts_root_native: Path,
    output_root_native: Path,
    data_root_native: Path,
) -> tuple[str, str, str]:
    """
    Paths used *inside WSL bash*.
    """
    return (
        _windows_to_wsl_path(scripts_root_native),
        _windows_to_wsl_path(output_root_native),
        _windows_to_wsl_path(data_root_native),
    )


def wsl_available() -> bool:
    if not is_windows():
        return False
    try:
        p = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "command -v python3 >/dev/null 2>&1; echo $?"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return p.returncode == 0 and p.stdout.strip().endswith("0")
    except Exception:
        return False


def tail_text(s: str, max_chars: int = 1200) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def resolve_default_workers() -> int:
    """Return default worker count (all logical CPU threads).

    Unlike ``platform_setup.resolve_workers`` this intentionally ignores
    ADSB_MAX_WORKERS / ARENA_MAX_WORKERS and the default_cap ceiling because
    the pipeline orchestrator itself is not CPU-bound — the cap is applied
    per-child-script via env vars passed to subprocess.
    """
    return max(1, os.cpu_count() or 1)


# =========================
# Backend abstraction
# =========================
@dataclass
class Backend:
    """
    - native: run scripts with current python (sys.executable) and native file paths.
    - wsl:    run scripts via `wsl bash -lc` using /mnt/<drive>/... paths for execution,
              while validating outputs via Windows native paths (E:\\...) if master runs on Windows.
    """

    kind: str  # "native" or "wsl"
    scripts_root_native: Path
    output_root_native: Path
    data_root_native: Path

    # execution roots used in commands (posix paths inside WSL)
    scripts_root_exec: str | None = None
    output_root_exec: str | None = None
    data_root_exec: str | None = None
    pythonpath_exec: str | None = None

    python_native: str = field(default_factory=lambda: sys.executable)

    def describe(self) -> str:
        if self.kind == "native":
            return f"native ({self.python_native})"
        return "wsl (python3)"

    def ensure_output_dirs(self, subdirs: Sequence[str]) -> None:
        """
        Create output directories on *native filesystem* (Windows path when master runs on Windows).
        """
        for d in subdirs:
            (self.output_root_native / d).mkdir(parents=True, exist_ok=True)

    def run_python_snippet(self, code: str, env: dict[str, str], timeout_s: int = 30) -> subprocess.CompletedProcess:
        """
        Run a python -c snippet in the execution environment (native or wsl).
        """
        if self.kind == "native":
            cmd = [self.python_native, "-c", code]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)

        # WSL: run python3 -c ...
        code_q = shlex.quote(code)
        cmd = ["wsl", "-e", "bash", "-lc", f"python3 -c {code_q}"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)

    def build_script_cmd(self, script_rel_posix: str, extra_args: list[str] | None = None) -> tuple[list[str], Optional[str]]:
        """
        Return (cmd, cwd) for subprocess.run.
        - script_rel_posix must be posix-style relative path, e.g. "adsb/aggregators/adsb_aggregator.py"
        - extra_args: additional CLI arguments to pass to the script
        """
        if self.kind == "native":
            script_path = self.scripts_root_native / script_rel_posix
            cmd = [self.python_native, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)
            return cmd, str(self.scripts_root_native)

        assert self.scripts_root_exec is not None
        rel_q = shlex.quote(script_rel_posix)
        args_str = " ".join(shlex.quote(a) for a in extra_args) if extra_args else ""
        env_prefix = ""
        if self.pythonpath_exec:
            env_prefix = f"PYTHONPATH={shlex.quote(self.pythonpath_exec)} "
        payload = f"cd {shlex.quote(self.scripts_root_exec)} && {env_prefix}python3 {rel_q}"
        if args_str:
            payload += f" {args_str}"
        cmd = ["wsl", "-e", "bash", "-lc", payload]
        return cmd, None

BASE_MODULES = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "folium", "requests"]
STAGE4_MODULES = ["jax", "numpyro"]         # phase eval (NumPyro NUTS)
STAGE5_MODULES = ["jax", "numpyro"]         # change point / numpyro
STAGE5_PYMC = ["pymc", "arviz"]             # PyMC bayesian scripts


def missing_modules(backend: Backend, modules: Sequence[str], env: dict[str, str]) -> list[str]:
    mods = list(modules)
    code = (
        "import importlib.util as u; "
        f"mods={mods!r}; "
        "miss=[m for m in mods if u.find_spec(m) is None]; "
        "print('\\n'.join(miss))"
    )
    try:
        proc = backend.run_python_snippet(code, env=env, timeout_s=30)
        out = (proc.stdout or "").strip()
        if not out:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return list(modules)


# =========================
# GPU detection (JAX)
# =========================
def detect_gpu_jax(backend: Backend, env: dict[str, str]) -> dict:
    """
    Detect GPU availability for JAX in the *execution environment*.
    """
    info = {"available": False, "device": "CPU only", "jax": False}
    code = (
        "import jax; "
        "ds=jax.devices(); "
        "gpu=[d for d in ds if d.platform=='gpu']; "
        "print('1' if len(gpu)>0 else '0'); "
        "print(gpu[0].device_kind if gpu else 'none')"
    )
    try:
        proc = backend.run_python_snippet(code, env={**env, "JAX_PLATFORMS": "cuda,cpu"}, timeout_s=30)
        lines = (proc.stdout or "").strip().splitlines()
        if lines and lines[0].strip() == "1":
            info["available"] = True
            info["jax"] = True
            info["device"] = lines[1].strip() if len(lines) > 1 else "GPU"
    except Exception:
        pass
    return info
