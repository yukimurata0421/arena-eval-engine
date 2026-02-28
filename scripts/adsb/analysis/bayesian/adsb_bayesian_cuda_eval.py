"""
adsb_bayesian_cuda_eval.py module.
"""
import runpy
import sys
from pathlib import Path


from arena.lib.paths import SCRIPTS_ROOT

GPU_SCRIPT = str(Path(SCRIPTS_ROOT) / "adsb" / "analysis" / "gpu" / "adsb_bayesian_phase_cuda_eval.py")

if __name__ == "__main__":
    runpy.run_path(GPU_SCRIPT, run_name="__main__")
