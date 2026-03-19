"""
NOTE:
Unified the old version of CUDA Bayes evaluation (intervention date input type) into a phase comparison model.
The main body of the GPU version has been moved to:
  <project>\\scripts\\adsb\\analysis\\gpu\\adsb_bayesian_phase_cuda_eval.py
"""
import runpy
import sys
from pathlib import Path


from arena.lib.paths import SCRIPTS_ROOT

GPU_SCRIPT = str(Path(SCRIPTS_ROOT) / "adsb" / "analysis" / "gpu" / "adsb_bayesian_phase_cuda_eval.py")

if __name__ == "__main__":
    runpy.run_path(GPU_SCRIPT, run_name="__main__")
