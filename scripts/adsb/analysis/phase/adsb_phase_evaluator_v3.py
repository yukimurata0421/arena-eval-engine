"""
adsb_phase_evaluator.py (v3.1 — dual-baseline + data quality gates)

Changes v3 → v3.1:
  1. Uptime gate: exclude days with minutes_covered < MIN_MINUTES_COVERED
     (prevents misclassifying short-uptime days as regressions)
  2. minutes_covered offset: add log(minutes_covered) as an offset
     → properly normalizes short-uptime days
  3. Auto-detect low-N phases + warnings
     → N < MIN_DAYS_FOR_CONCLUSION marked as preliminary in reports
  4. Data quality summary section (excluded days + reasons)

Environment variables:
  ADSB_ALT_BASELINE_IDX  — set alt baseline phase_idx directly (0-origin)
  ADSB_BATCH_MODE=1      — batch mode
  ADSB_PHASE_INTERACTIVE=1 — force interactive mode
  ADSB_MIN_MINUTES=1296  — uptime gate threshold (default: 1296 = 1440*0.9)
  ADSB_MIN_PHASE_DAYS=7  — minimum days for quantitative conclusions (default: 7)

Required libraries: jax, numpyro, pandas, numpy, matplotlib
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


from arena.lib.paths import ADSB_DAILY_SUMMARY, OUTPUT_DIR as OUT_ROOT
from arena.lib.phase_config import get_config as _get_cfg
from arena.lib.config import get_quality_thresholds
from arena.lib.platform_setup import init_numpyro_platform

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

OUTPUT_DIR = str(OUT_ROOT / "performance")

IS_BATCH = os.environ.get("ADSB_BATCH_MODE") == "1" or (not sys.stdin.isatty())
FORCE_INTERACTIVE = os.environ.get("ADSB_PHASE_INTERACTIVE") == "1"

MIN_MINUTES_COVERED = int(os.environ.get("ADSB_MIN_MINUTES", "1296"))
MIN_DAYS_FOR_CONCLUSION = int(os.environ.get("ADSB_MIN_PHASE_DAYS", "7"))
MIN_AUC_N_USED, _MIN_MINUTES = get_quality_thresholds()


def _get_phases_from_config():
    """Auto-load phases from phase_config.py (batch mode)."""
    cfg = _get_cfg()
    phases = []
    for ev in cfg.events:
        if ev.hardware:
            phases.append({"date": ev.date, "name": ev.label, "hardware": ev.hardware})
    if not phases:
        phases.append({"date": cfg.post_change_date, "name": "Baseline", "hardware": ""})
    return phases


def _get_phases_interactive():
    """Prompt for phases interactively (legacy compatibility)."""
    from arena.lib.input_utils import prompt_phase_dates
    cfg = _get_cfg()
    default_labels = {d: n for d, n in cfg.phase_dates}
    return prompt_phase_dates(default_labels)


def get_phases():
    if FORCE_INTERACTIVE:
        return _get_phases_interactive()
    return _get_phases_from_config()


def _read_setting_from_phases_txt(key: str) -> str | None:
    """
    Read a key directly from phases.txt [settings].
    Fallback when lib/phase_config lacks new fields such as alt_baseline_date.
    """
    candidates = [
        Path(os.environ.get("ADSB_PHASE_CONFIG", "")),
        Path(__file__).resolve().parents[3] / "config" / "phases.txt",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            in_settings = False
            for line in p.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("[settings]"):
                    in_settings = True
                    continue
                if stripped.startswith("[") and in_settings:
                    break
                if in_settings and "=" in stripped and not stripped.startswith("#"):
                    k, v = stripped.split("=", 1)
                    if k.strip() == key:
                        return v.strip()
        except Exception:
            continue
    return None


def _detect_alt_baseline_idx(phases: list) -> int:
    """
    Auto-detect alt baseline phase_idx.

    Priority order:
      1. ADSB_ALT_BASELINE_IDX if set
      2. Phase containing phases.txt [settings].alt_baseline_date
      3. First phase with hardware == "airspy_mini"
      4. Fallback: min(1, len(phases)-1)
    """
    env_val = os.environ.get("ADSB_ALT_BASELINE_IDX", "").strip()
    if env_val:
        try:
            idx = int(env_val)
            if 0 <= idx < len(phases):
                return idx
        except ValueError:
            pass

    try:
        cfg = _get_cfg()
        alt_date_str = getattr(cfg, "alt_baseline_date", None)

        if not alt_date_str:
            alt_date_str = _read_setting_from_phases_txt("alt_baseline_date")

        if alt_date_str:
            from datetime import date as date_cls
            if isinstance(alt_date_str, str):
                alt_date = date_cls.fromisoformat(alt_date_str)
            else:
                alt_date = alt_date_str

            for i in range(len(phases)):
                phase_start = date_cls.fromisoformat(str(phases[i]["date"]))
                if i + 1 < len(phases):
                    next_start = date_cls.fromisoformat(str(phases[i + 1]["date"]))
                    if phase_start <= alt_date < next_start:
                        return i
                else:
                    if phase_start <= alt_date:
                        return i
    except Exception as e:
        print(f"  [WARN] failed to read alt_baseline_date ({e}); using fallback")

    for i, p in enumerate(phases):
        hw = p.get("hardware", "")
        if hw and "airspy" in hw.lower() and "cable" not in hw.lower():
            return i

    return min(1, len(phases) - 1)


def _compute_relative(alphas_samples, idx_target, idx_reference):
    """
    Compute relative improvement from Bayesian posterior samples.

    Returns: (mean_pct, hdi_lo, hdi_hi, prob_positive_pct)
    """
    rel = (jnp.exp(alphas_samples[:, idx_target] - alphas_samples[:, idx_reference]) - 1) * 100
    mean_pct = float(jnp.mean(rel))
    hdi_lo = float(jnp.percentile(rel, 3.0))
    hdi_hi = float(jnp.percentile(rel, 97.0))
    prob_pos = float((rel > 0).mean()) * 100
    return mean_pct, hdi_lo, hdi_hi, prob_pos


def _phase_reliability_tag(n_days: int) -> str:
    """Return phase reliability tag."""
    if n_days >= MIN_DAYS_FOR_CONCLUSION:
        return ""
    if n_days >= 3:
        return " [prelim: low N]"
    if n_days >= 1:
        return " [reference: N<3]"
    return " [no data]"


def run_phase_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_path = str(ADSB_DAILY_SUMMARY)
    if not os.path.exists(data_path):
        print(f"  Error: {data_path} not found.")
        return

    df_raw = pd.read_csv(data_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('date').reset_index(drop=True)

    # ================================================================
    # ================================================================
    n_total = len(df_raw)

    mask_auc = df_raw['auc_n_used'] > MIN_AUC_N_USED
    n_auc_excluded = int((~mask_auc).sum())

    has_minutes = 'minutes_covered' in df_raw.columns
    if has_minutes:
        mask_minutes = df_raw['minutes_covered'] >= MIN_MINUTES_COVERED
        n_minutes_excluded = int((mask_auc & ~mask_minutes).sum())
        mask_all = mask_auc & mask_minutes
    else:
        mask_minutes = pd.Series(True, index=df_raw.index)
        n_minutes_excluded = 0
        mask_all = mask_auc

    df = df_raw[mask_all].copy().reset_index(drop=True)

    print(f"\n  Data quality gate:")
    print(f"    Total days:           {n_total}")
    print(f"    AUC excluded (<{MIN_AUC_N_USED}):  {n_auc_excluded}")
    if has_minutes:
        print(f"    Minutes excluded (<{MIN_MINUTES_COVERED}m): {n_minutes_excluded}")
    print(f"    Valid days:         {len(df)}")

    excluded = df_raw[~mask_all].copy()
    if len(excluded) > 0:
        print(f"\n    Excluded days (latest 10):")
        for _, row in excluded.tail(10).iterrows():
            reason = []
            if row['auc_n_used'] <= MIN_AUC_N_USED:
                reason.append(f"AUC={row['auc_n_used']:.0f}")
            if has_minutes and row['minutes_covered'] < MIN_MINUTES_COVERED:
                reason.append(f"minutes={row['minutes_covered']:.0f}m")
            print(f"      {row['date'].strftime('%Y-%m-%d')} — {', '.join(reason)}")

    phases = get_phases()
    if len(phases) < 2:
        print("  Fewer than 2 phases. Cannot compare.")
        return

    df['phase_idx'] = -1
    for i, p in enumerate(phases):
        df.loc[df['date'] >= pd.Timestamp(p['date']), 'phase_idx'] = i
    df = df[df['phase_idx'] >= 0].reset_index(drop=True)

    if len(df) < 5:
        print("  Fewer than 5 valid days.")
        return

    init_numpyro_platform(n_data=len(df))

    alt_baseline_idx = _detect_alt_baseline_idx(phases)
    print(f"  Original baseline: Phase 0 ({phases[0]['name']})")
    print(f"  Alt baseline:      Phase {alt_baseline_idx} ({phases[alt_baseline_idx]['name']})")

    num_phases = len(phases)
    phase_n_days = {}
    low_n_phases = []

    print(f"\n  Data: {len(df)} days, phases: {num_phases}")
    for i, p in enumerate(phases):
        n = int((df['phase_idx'] == i).sum())
        phase_n_days[i] = n
        tag = _phase_reliability_tag(n)
        marker = " ← original baseline" if i == 0 else (" ← alt baseline" if i == alt_baseline_idx else "")
        print(f"    {p['name']}: {n} days{marker}{tag}")
        if n < MIN_DAYS_FOR_CONCLUSION:
            low_n_phases.append((i, p['name'], n))

    if low_n_phases:
        print(f"\n  ⚠ Note: {len(low_n_phases)} phases below minimum days ({MIN_DAYS_FOR_CONCLUSION})")
        print(f"    → These phases are preliminary/reference; no quantitative conclusions.")

    df['auc_n_used'] = df['auc_n_used'].fillna(0).clip(lower=0)
    hnd = df['hnd_nrt_movements'].fillna(1).replace(0, 1)
    y = jnp.array(df['auc_n_used'].values, dtype=jnp.float32)
    log_traffic = jnp.array(np.log(hnd.values), dtype=jnp.float32)
    phase_idx = jnp.array(df['phase_idx'].values)

    # minutes_covered offset
    if has_minutes:
        minutes = df['minutes_covered'].fillna(1440).clip(lower=60)
        log_minutes = jnp.array(np.log(minutes.values), dtype=jnp.float32)
        use_minutes_offset = True
    else:
        log_minutes = None
        use_minutes_offset = False

    def model(y, log_traffic, phase_idx, num_phases, log_minutes=None):
        beta_traffic = numpyro.sample('beta_traffic', dist.Normal(1., 0.5))
        alphas = numpyro.sample('alphas', dist.Normal(10., 5.).expand([num_phases]))
        alpha_inv = numpyro.sample('alpha_inv', dist.Exponential(1.0))

        log_mu = alphas[phase_idx] + beta_traffic * log_traffic
        if log_minutes is not None:
            beta_minutes = numpyro.sample('beta_minutes', dist.Normal(1., 0.3))
            log_mu = log_mu + beta_minutes * log_minutes

        mu = jnp.exp(log_mu)
        numpyro.sample('y_obs', dist.NegativeBinomial2(mu, alpha_inv), obs=y)

    # --- MCMC ---
    default_warmup = 1000
    default_samples = 2000
    n_warmup = int(os.environ.get("ADSB_PHASE_WARMUP", default_warmup))
    n_samples = int(os.environ.get("ADSB_PHASE_SAMPLES", default_samples))

    n_chains_env = int(os.environ.get("ADSB_PHASE_CHAINS", "0"))
    n_chains = n_chains_env if n_chains_env > 0 else 4

    mcmc = MCMC(NUTS(model), num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    print(f"\n  Running MCMC (warmup={n_warmup}, samples={n_samples}, chains={n_chains})...")
    print(f"  Model: NegBin + traffic_control" + (" + minutes_offset" if use_minutes_offset else ""))
    mcmc.run(random.PRNGKey(42), y, log_traffic, phase_idx, num_phases,
             log_minutes if use_minutes_offset else None)

    samples = mcmc.get_samples()
    alphas = samples['alphas']
    beta_traffic = samples['beta_traffic']

    # ================================================================
    # ================================================================
    results = []
    lines = []

    lines.append("=" * 100)
    lines.append("  ADS-B Phase Evaluator Report v3.1 (NegBin + Traffic + Minutes Offset, Dual Baseline)")
    lines.append("=" * 100)
    lines.append(f"  Data: {len(df)} days (excluded: AUC<{MIN_AUC_N_USED}={n_auc_excluded} days"
                 + (f", minutes<{MIN_MINUTES_COVERED}m={n_minutes_excluded} days" if has_minutes else "")
                 + f"), phases: {num_phases}")
    lines.append(f"  MCMC: warmup={n_warmup}, samples={n_samples}, chains={n_chains}")
    lines.append(f"  Model: NegBin + traffic_control" + (" + minutes_offset" if use_minutes_offset else ""))
    lines.append(f"  Original baseline: Phase 0 = {phases[0]['name']}")
    lines.append(f"  Alt baseline:      Phase {alt_baseline_idx} = {phases[alt_baseline_idx]['name']}")
    lines.append(f"  Minimum days for conclusions: {MIN_DAYS_FOR_CONCLUSION}")
    lines.append("")

    # ── Section 1: vs Original baseline (Phase 0) ──
    lines.append("-" * 100)
    lines.append("  SECTION 1: vs Original Baseline (Phase 0)")
    lines.append("-" * 100)
    lines.append(f"  {'Phase Name':<30} | {'N':>4} | {'vs Previous':<22} | {'vs Baseline(Ph0)':<35}")
    lines.append("  " + "-" * 96)

    for i in range(num_phases):
        n_days = phase_n_days[i]
        tag = _phase_reliability_tag(n_days)

        mean_base, hdi_base_lo, hdi_base_hi, prob_base = _compute_relative(alphas, i, 0)

        if i == 0:
            vs_prev_str = "---"
            mean_prev, hdi_prev_lo, hdi_prev_hi, prob_prev = 0.0, 0.0, 0.0, 0.0
        else:
            mean_prev, hdi_prev_lo, hdi_prev_hi, prob_prev = _compute_relative(alphas, i, i - 1)
            vs_prev_str = f"{mean_prev:>+7.1f}% P(>0)={prob_prev:.0f}%"

        vs_base_str = f"{mean_base:>+7.1f}% [{hdi_base_lo:>+7.1f}, {hdi_base_hi:>+7.1f}] P(>0)={prob_base:.0f}%"
        lines.append(f"  {phases[i]['name']:<30} | {n_days:>4} | {vs_prev_str:<22} | {vs_base_str}{tag}")

        results.append({
            'phase_idx': i,
            'phase_name': phases[i]['name'],
            'phase_date': phases[i]['date'],
            'n_days': n_days,
            'reliability': 'definitive' if n_days >= MIN_DAYS_FOR_CONCLUSION
                           else ('preliminary' if n_days >= 3 else 'reference'),
            'vs_baseline_mean_pct': round(mean_base, 2),
            'vs_baseline_hdi94_lo': round(hdi_base_lo, 2),
            'vs_baseline_hdi94_hi': round(hdi_base_hi, 2),
            'vs_baseline_prob_positive_pct': round(prob_base, 1),
            'vs_previous_mean_pct': round(mean_prev, 2),
            'vs_previous_hdi94_lo': round(hdi_prev_lo, 2),
            'vs_previous_hdi94_hi': round(hdi_prev_hi, 2),
            'vs_previous_prob_positive_pct': round(prob_prev, 1),
        })

    lines.append("")

    # ── Section 2: vs Alt baseline ──
    lines.append("-" * 100)
    lines.append(f"  SECTION 2: vs Alt Baseline (Phase {alt_baseline_idx}: {phases[alt_baseline_idx]['name']})")
    lines.append("-" * 100)
    lines.append(f"  {'Phase Name':<30} | {'N':>4} | {'vs Alt Baseline':<40}")
    lines.append("  " + "-" * 80)

    for i in range(num_phases):
        n_days = phase_n_days[i]
        tag = _phase_reliability_tag(n_days)

        mean_alt, hdi_alt_lo, hdi_alt_hi, prob_alt = _compute_relative(alphas, i, alt_baseline_idx)
        vs_alt_str = f"{mean_alt:>+7.1f}% [{hdi_alt_lo:>+7.1f}, {hdi_alt_hi:>+7.1f}] P(>0)={prob_alt:.0f}%"

        lines.append(f"  {phases[i]['name']:<30} | {n_days:>4} | {vs_alt_str}{tag}")

        results[i]['alt_baseline_idx'] = alt_baseline_idx
        results[i]['alt_baseline_name'] = phases[alt_baseline_idx]['name']
        results[i]['vs_alt_mean_pct'] = round(mean_alt, 2)
        results[i]['vs_alt_hdi94_lo'] = round(hdi_alt_lo, 2)
        results[i]['vs_alt_hdi94_hi'] = round(hdi_alt_hi, 2)
        results[i]['vs_alt_prob_positive_pct'] = round(prob_alt, 1)

    lines.append("")

    lines.append("-" * 100)
    lines.append("  SECTION 3: Interpretation Guide")
    lines.append("-" * 100)
    lines.append("")
    lines.append("  Section 1 (vs Phase 0) measures improvement from the global baseline.")
    lines.append("  The RTL-SDR → Airspy jump dominates and can mask Cable/Adapter effects.")
    lines.append("")
    lines.append("  Section 2 (vs Alt Baseline) captures post-Airspy fine-tuning effects.")
    lines.append("  It can resolve 5–15% improvements from cable/adapter changes.")
    lines.append("")
    lines.append("  Reliability tags:")
    lines.append(f"    (no tag)      — N >= {MIN_DAYS_FOR_CONCLUSION}. Supports conclusions.")
    lines.append(f"    [prelim: low N]   — 3 <= N < {MIN_DAYS_FOR_CONCLUSION}. Trend only; no conclusion.")
    lines.append(f"    [reference: N<3]  — N < 3. Not enough data for statistical judgment.")
    lines.append("")
    lines.append("  Decision criteria (no-tag phases only):")
    lines.append("    P(>0) >= 95%  → improvement is very likely")
    lines.append("    P(>0) >= 80%  → improvement likely")
    lines.append("    P(>0) <  80%  → effect unclear (need more data)")
    lines.append("    94% HDI excludes 0 → statistically significant")
    lines.append("")

    beta_mean = float(jnp.mean(beta_traffic))
    beta_lo = float(jnp.percentile(beta_traffic, 3.0))
    beta_hi = float(jnp.percentile(beta_traffic, 97.0))
    lines.append(f"  Traffic elasticity (beta_traffic): {beta_mean:.4f} [94% HDI: {beta_lo:.4f}, {beta_hi:.4f}]")

    if use_minutes_offset and 'beta_minutes' in samples:
        bm = samples['beta_minutes']
        bm_mean = float(jnp.mean(bm))
        bm_lo = float(jnp.percentile(bm, 3.0))
        bm_hi = float(jnp.percentile(bm, 97.0))
        lines.append(f"  Minutes elasticity (beta_minutes): {bm_mean:.4f} [94% HDI: {bm_lo:.4f}, {bm_hi:.4f}]")

    lines.append("=" * 100)

    print("\n" + "\n".join(lines))

    csv_path = os.path.join(OUTPUT_DIR, "phase_evaluator_results.csv")
    txt_path = os.path.join(OUTPUT_DIR, "phase_evaluator_report.txt")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Saved results: {csv_path}")
    print(f"  Report:   {txt_path}")

    _plot_dual_boxplot(alphas, phases, num_phases, alt_baseline_idx, phase_n_days)


def _plot_dual_boxplot(alphas, phases, num_phases, alt_baseline_idx, phase_n_days):
    """Generate 2-panel boxplots vs Phase 0 and Alt baseline."""
    import matplotlib
    if IS_BATCH:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Prefer Japanese-capable fonts on Windows to avoid mojibake.
    preferred_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "MS Mincho"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            plt.rcParams["font.family"] = name
            break

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    labels = [f"{p['name'][:20]}" for p in phases]

    label_display = []
    for i, lbl in enumerate(labels):
        n = phase_n_days.get(i, 0)
        if n < MIN_DAYS_FOR_CONCLUSION:
            label_display.append(f"{lbl}\n(N={n}, prelim)")
        else:
            label_display.append(f"{lbl}\n(N={n})")

    ax = axes[0]
    data_orig = [np.array(jnp.exp(alphas[:, i] - alphas[:, 0])) for i in range(num_phases)]
    bp = ax.boxplot(data_orig, tick_labels=label_display, patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        n = phase_n_days.get(j, 0)
        if n < MIN_DAYS_FOR_CONCLUSION:
            patch.set_facecolor('#ffe0e0')
            patch.set_linestyle('--')
        else:
            patch.set_facecolor('#e8e8e8')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7)
    ax.set_title(f"vs Phase 0: {phases[0]['name'][:25]}", fontsize=11)
    ax.set_ylabel("Performance Ratio")
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    ax = axes[1]
    data_alt = [np.array(jnp.exp(alphas[:, i] - alphas[:, alt_baseline_idx])) for i in range(num_phases)]
    bp = ax.boxplot(data_alt, tick_labels=label_display, patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        n = phase_n_days.get(j, 0)
        if j == alt_baseline_idx:
            patch.set_facecolor('#ffdddd')
        elif n < MIN_DAYS_FOR_CONCLUSION:
            patch.set_facecolor('#ffe0e0')
            patch.set_linestyle('--')
        else:
            patch.set_facecolor('#dde8ff')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7)
    ax.set_title(f"vs Phase {alt_baseline_idx}: {phases[alt_baseline_idx]['name'][:25]}", fontsize=11)
    ax.set_ylabel("Performance Ratio")
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    fig.suptitle("ADS-B Phase Evaluator — Dual Baseline Comparison (v3.1)", fontsize=13)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "phase_evaluator_boxplot.png")
    if IS_BATCH or not hasattr(plt, 'show'):
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot:   {png_path}")
    else:
        plt.show()


if __name__ == "__main__":
    run_phase_analysis()
