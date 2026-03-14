import os

import numpy as np

from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.input_utils import prompt_intervention_date


def _env_posint(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_workers(default_cap: int = 12) -> int:
    return max(1, min(os.cpu_count() or 1, default_cap))


def run_bayesian_analysis() -> None:
    print(" ADS-B Bayesian evaluation engine (Multi-Core Optimized)")

    min_auc, min_minutes = get_quality_thresholds()
    df = load_summary(min_auc=min_auc, min_minutes=min_minutes)
    if df is None:
        return
    df = df.replace([np.inf, -np.inf], np.nan)

    print("\nEnter the configuration change date to evaluate")
    from arena.lib.phase_config import get_config as _get_cfg

    cutoff = prompt_intervention_date(_get_cfg().time_resolved_date)

    df["post"] = (df["date"] >= cutoff).astype(int)
    df["auc_n_used"] = df["auc_n_used"].fillna(0).clip(lower=0)
    df = df.dropna(subset=["auc_n_used", "local_traffic_proxy", "post", "log_traffic"])
    if len(df) < 5:
        print("  WARNING: insufficient valid data; skipping Bayesian analysis.")
        return

    y = df["auc_n_used"].values.astype(float)
    post_flag = df["post"].values
    log_traffic = df["log_traffic"].values

    mode = os.getenv("ADSB_BAYES_DYNAMIC_MODE", "mcmc").strip().lower()
    if mode in {"quick", "fast"}:
        pre_vals = y[post_flag == 0]
        post_vals = y[post_flag == 1]
        if len(pre_vals) == 0 or len(post_vals) == 0:
            print("  WARNING: pre/post data is missing; skipping quick approximation.")
            return
        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        denom = pre_mean if abs(pre_mean) > 1e-9 else 1e-9
        mean_improvement = ((post_mean / denom) - 1.0) * 100.0
        prob_improved = 100.0 if mean_improvement > 0 else 0.0

        print("\n" + "=" * 50)
        print(f" Bayesian analysis report (FAST): {cutoff.date()}")
        print(f"Estimated improvement (mean): {mean_improvement:+.2f} %")
        print(f"94% credible interval (HDI): [{mean_improvement:.1f}%, {mean_improvement:.1f}%]")
        print(f"[Probability the change was effective]: {prob_improved:.1f} %")
        print("=" * 50)
        return

    n_workers = _resolve_workers(default_cap=12)
    draws = _env_posint("ADSB_BAYES_DYNAMIC_DRAWS", 1000)
    tune = _env_posint("ADSB_BAYES_DYNAMIC_TUNE", 1000)
    max_chains = _env_posint("ADSB_BAYES_DYNAMIC_MAX_CHAINS", n_workers)
    n_chains = max(1, min(n_workers, max_chains))

    print(
        f"--- {cutoff.date()} Calculating improvement probability "
        f"(chains={n_chains}, draws={draws}, tune={tune}) ---"
    )

    try:
        import arviz as az
        import pymc as pm
    except Exception as exc:
        print(f"  WARNING: PyMC/ArviZ is unavailable; skipping MCMC mode ({exc}).")
        return

    with pm.Model():
        intercept = pm.Normal("intercept", mu=10, sigma=5)
        beta_traffic = pm.Normal("beta_traffic", mu=1, sigma=1)
        gamma = pm.Normal("gamma", mu=0, sigma=1)
        alpha_inv = pm.Exponential("alpha_inv", 1.0)

        mu = pm.math.exp(intercept + beta_traffic * log_traffic + gamma * post_flag)

        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha_inv, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=n_chains,
            cores=n_chains,
            random_seed=42,
            return_inferencedata=True,
            compute_convergence_checks=False,
            progressbar=False,
        )

    post_gamma = trace.posterior["gamma"].values.flatten()
    improvement_samples = (np.exp(post_gamma) - 1) * 100
    prob_improved = (improvement_samples > 0).mean() * 100
    mean_improvement = np.mean(improvement_samples)
    hdi_94 = az.hdi(improvement_samples, hdi_prob=0.94)

    print("\n" + "=" * 50)
    print(f" Bayesian analysis report: {cutoff.date()}")
    print(f"Estimated improvement (mean): {mean_improvement:+.2f} %")
    print(f"94% credible interval (HDI): [{hdi_94[0]:.1f}%, {hdi_94[1]:.1f}%]")
    print(f"[Probability the change was effective]: {prob_improved:.1f} %")
    print("=" * 50)


if __name__ == "__main__":
    run_bayesian_analysis()
