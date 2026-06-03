"""
adsb_bayesian_phase_cuda_eval.py module.
"""

import os

import numpy as np
import pandas as pd

from arena.lib.arviz_compat import hdi_bounds
from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.paths import OUTPUT_DIR as OUT_ROOT
from arena.lib.platform_setup import init_numpyro_platform
from arena.log import get_script_logger

log = get_script_logger(__name__)

OUTPUT_DIR = str(OUT_ROOT / "performance")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bayesian_phase_results_cuda.csv")


from arena.lib.phase_config import get_config as _get_cfg

_cfg = _get_cfg()
PHASE_MAP = _cfg.get_hardware_map()
PHASE_NAMES = _cfg.get_phase_names()


def _env_posint(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


CHAINS = _env_posint("ADSB_BAYES_PHASE_CHAINS", 4)
DRAWS = 2000
TUNE = 1000
RANDOM_SEED = 42


def run_bayesian_phase_cuda_analysis():
    try:
        import arviz as az
    except ImportError as e:
        log.info(f" A required library is missing: {e}")
        return

    n_cores = min(os.cpu_count() or 4, CHAINS)

    log.info(f" ADS-B Bayesian evaluation (NumPyro backend, chains={CHAINS})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["auc_n_used", "log_traffic"])

    if "hardware" in df.columns:
        df["phase_idx"] = df["hardware"].map(PHASE_MAP)
    else:
        df["phase_idx"] = 0
        _fallback = _cfg.get_phase_fallback_dates()
        for _i, _d in enumerate(_fallback):
            df.loc[df["date"] >= _d, "phase_idx"] = _i + 1

    df = df.dropna(subset=["phase_idx"]).copy()
    df["phase_idx"] = df["phase_idx"].astype(int)
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    y = df["auc_n_used"].values.astype(float)
    log_traffic = df["log_traffic"].values.astype(float)
    phase_idx = df["phase_idx"].values
    is_weekend = df["is_weekend"].values.astype(float)
    num_phases = len(df["phase_idx"].unique())
    if len(df) < 5 or num_phases < 2:
        log.info("Warning: Skipping Bayesian analysis due to insufficient valid data.")
        return

    log.info(f" data: {len(df)} days, number of phases: {num_phases}")
    for i in range(num_phases):
        n = (phase_idx == i).sum()
        name = PHASE_NAMES.get(i, f"Phase{i}")
        log.info(f" {name}: {n} days")

    platform = init_numpyro_platform(n_data=len(df))
    if platform == "cuda":
        n_cores = 1
    try:
        import pymc as pm
    except ImportError as e:
        log.info(f" A required library is missing: {e}")
        return

    log.info(f"\n Building PyMC model (NumPyro/{platform}, cores={n_cores})...")
    with pm.Model():
        alphas = pm.Normal("alphas", mu=10.0, sigma=3.0, shape=num_phases)
        beta_traffic = pm.Normal("beta_traffic", mu=0.5, sigma=0.5)
        beta_weekend = pm.Normal("beta_weekend", mu=0.0, sigma=0.5)
        phi = pm.Exponential("phi", 1.0)

        mu = pm.math.exp(alphas[phase_idx] + beta_traffic * log_traffic + beta_weekend * is_weekend)
        pm.NegativeBinomial("y_obs", mu=mu, alpha=phi, observed=y)

        log.info(f" MCMC running (chains={CHAINS}, draws={DRAWS}, tune={TUNE})...")
        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=n_cores,
            nuts_sampler="numpyro",
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            progressbar=True,
        )

    log.info("\n" + "=" * 70)
    log.info(f"ADS-B Bayesian report ({platform} / PyMC+NumPyro)")
    log.info("=" * 70)

    summary = az.summary(trace, var_names=["alphas", "beta_traffic", "beta_weekend", "phi"])
    log.info("\n--- Parameter summary ---")
    log.info(summary)

    alphas_samples = trace.posterior["alphas"].values
    alphas_flat = alphas_samples.reshape(-1, num_phases)

    log.info("\n--- Improvements between phases ---")
    log.info(f"{'Comparison':<35} {'Average':>8} {'HDI 94%':>20} {'P(>0)':>8}")
    log.info("-" * 75)

    results = []
    for i in range(1, num_phases):
        diff_vs_base = alphas_flat[:, i] - alphas_flat[:, 0]
        improvement_vs_base = (np.exp(diff_vs_base) - 1) * 100
        mean_imp = np.mean(improvement_vs_base)
        hdi_lo, hdi_hi = hdi_bounds(improvement_vs_base, hdi_prob=0.94)
        prob_positive = (improvement_vs_base > 0).mean() * 100

        name_i = PHASE_NAMES.get(i, f"Phase{i}")
        name_0 = PHASE_NAMES.get(0, "Phase0")
        label = f"{name_i} vs {name_0}"
        log.info(f"{label:<35} {mean_imp:>+7.1f}% [{hdi_lo:>+7.1f}, {hdi_hi:>+7.1f}]  {prob_positive:>6.1f}%")
        results.append(
            {
                "comparison": label,
                "mean_improvement_pct": round(mean_imp, 2),
                "hdi_94_lower": round(hdi_lo, 2),
                "hdi_94_upper": round(hdi_hi, 2),
                "prob_positive_pct": round(prob_positive, 1),
            }
        )

        if i > 1:
            diff_vs_prev = alphas_flat[:, i] - alphas_flat[:, i - 1]
            improvement_vs_prev = (np.exp(diff_vs_prev) - 1) * 100
            mean_prev = np.mean(improvement_vs_prev)
            hdi_prev_lo, hdi_prev_hi = hdi_bounds(improvement_vs_prev, hdi_prob=0.94)
            prob_prev = (improvement_vs_prev > 0).mean() * 100

            name_prev = PHASE_NAMES.get(i - 1, f"Phase{i - 1}")
            label_prev = f"{name_i} vs {name_prev}"
            log.info(f"{label_prev:<35} {mean_prev:>+7.1f}% [{hdi_prev_lo:>+7.1f}, {hdi_prev_hi:>+7.1f}]  {prob_prev:>6.1f}%")
            results.append(
                {
                    "comparison": label_prev,
                    "mean_improvement_pct": round(mean_prev, 2),
                    "hdi_94_lower": round(hdi_prev_lo, 2),
                    "hdi_94_upper": round(hdi_prev_hi, 2),
                    "prob_positive_pct": round(prob_prev, 1),
                }
            )

    beta_traffic_samples = trace.posterior["beta_traffic"].values.flatten()
    beta_weekend_samples = trace.posterior["beta_weekend"].values.flatten()

    log.info("\n--- covariate effects ---")
    log.info(f"  Traffic elasticity: {np.mean(beta_traffic_samples):.4f} (94% HDI: {hdi_bounds(beta_traffic_samples, hdi_prob=0.94)})")
    weekend_pct = (np.exp(beta_weekend_samples) - 1) * 100
    log.info(
        f"  Weekend effect: {np.mean(weekend_pct):+.1f}% "
        f"(94% HDI: [{np.percentile(weekend_pct, 3):.1f}, {np.percentile(weekend_pct, 97):.1f}])"
    )
    log.info("=" * 70)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"\nResult saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_bayesian_phase_cuda_analysis()
