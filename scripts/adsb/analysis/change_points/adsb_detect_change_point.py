"""
adsb_detect_change_point.py module.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HAS_MCMC_DEPS = True
MCMC_IMPORT_ERROR = ""
try:
    from jax import random
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import DiscreteHMCGibbs, MCMC, NUTS
except Exception as exc:  # pragma: no cover - runtime dependency guard
    HAS_MCMC_DEPS = False
    MCMC_IMPORT_ERROR = str(exc)
    random = None
    jnp = None
    numpyro = None
    dist = None
    DiscreteHMCGibbs = None
    MCMC = None
    NUTS = None

from arena.lib.config import get_quality_thresholds
from arena.lib.data_loader import load_summary
from arena.lib.paths import resolve_output_dir


def _resolve_workers(default_cap: int = 12) -> int:
    return max(1, min(os.cpu_count() or 1, default_cap))


SCRIPT_NAME = Path(__file__).name
METRIC_NAME = "auc_n_used"
CPU_HOST = _resolve_workers(default_cap=12)
if HAS_MCMC_DEPS:
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(CPU_HOST)
    print(f"Platform: CPU ({CPU_HOST} devices)")
else:
    print(f"[WARN] change point dependencies unavailable: {MCMC_IMPORT_ERROR}")


def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[OK] report: {path}")
    except Exception as exc:
        print(f"[WARN] failed to save report: {path} ({exc})")


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] json: {path}")
    except Exception as exc:
        print(f"[WARN] failed to save json: {path} ({exc})")


def _segment_summary(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def _compute_base_drop_reasons(source_path: Path, min_auc: int, min_minutes: int) -> tuple[int, dict[str, int], list[str]]:
    warnings: list[str] = []
    reasons: dict[str, int] = {}
    if not source_path.exists():
        warnings.append(f"source_csv_not_found: {source_path}")
        return 0, reasons, warnings

    try:
        raw_df = pd.read_csv(source_path)
    except Exception as exc:
        warnings.append(f"source_csv_read_failed: {exc}")
        return 0, reasons, warnings

    total_rows = int(len(raw_df))
    if "date" not in raw_df.columns:
        reasons["missing_date_column"] = total_rows
    else:
        parsed_date = pd.to_datetime(raw_df["date"], errors="coerce")
        reasons["invalid_date_rows"] = int(parsed_date.isna().sum())

    if "auc_n_used" not in raw_df.columns:
        reasons["missing_auc_n_used_column"] = total_rows
    else:
        auc = pd.to_numeric(raw_df["auc_n_used"], errors="coerce")
        reasons["auc_n_used_non_numeric_rows"] = int(auc.isna().sum())
        reasons["auc_n_used_below_or_equal_threshold_rows"] = int((auc <= min_auc).fillna(False).sum())

    if "minutes_covered" in raw_df.columns:
        minutes = pd.to_numeric(raw_df["minutes_covered"], errors="coerce")
        reasons["minutes_covered_non_numeric_rows"] = int(minutes.isna().sum())
        reasons["minutes_covered_below_threshold_rows"] = int((minutes < min_minutes).fillna(False).sum())
    else:
        warnings.append("minutes_covered column not found")

    if "local_traffic_proxy" not in raw_df.columns:
        reasons["missing_local_traffic_proxy_column"] = total_rows
    else:
        proxy = pd.to_numeric(raw_df["local_traffic_proxy"], errors="coerce")
        reasons["local_traffic_proxy_non_numeric_rows"] = int(proxy.isna().sum())
        reasons["local_traffic_proxy_zero_rows"] = int((proxy == 0).fillna(False).sum())

    return total_rows, reasons, warnings


def _build_report_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Change Point Report",
        "",
        f"script_name: {payload['script_name']}",
        f"generated_at: {payload['generated_at']}",
        f"input file / source path: {payload['source_path']}",
        f"metric_name: {payload['metric_name']}",
        f"quality.min_auc_n_used: {payload['thresholds']['min_auc_n_used']}",
        f"quality.min_minutes_covered: {payload['thresholds']['min_minutes_covered']}",
        f"total_rows: {payload['total_rows']}",
        f"usable_rows: {payload['usable_rows']}",
        f"dropped_rows: {payload['dropped_rows']}",
        "",
        "dropped_reasons:",
    ]
    if payload["dropped_reasons"]:
        for key, value in payload["dropped_reasons"].items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("detected_change_points:")
    if payload["detected_change_points"]:
        for cp in payload["detected_change_points"]:
            lines.append(
                "- index={index} date={date} confidence_pct={confidence_pct:.2f}".format(
                    index=cp.get("index"),
                    date=cp.get("date"),
                    confidence_pct=float(cp.get("confidence_pct", 0.0)),
                )
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("summary_before_after:")
    summary = payload.get("summary_before_after", {})
    if summary:
        lines.append(
            f"- before: n={summary.get('before', {}).get('n')} "
            f"mean={summary.get('before', {}).get('mean')} "
            f"median={summary.get('before', {}).get('median')}"
        )
        lines.append(
            f"- after:  n={summary.get('after', {}).get('n')} "
            f"mean={summary.get('after', {}).get('mean')} "
            f"median={summary.get('after', {}).get('median')}"
        )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("warnings:")
    if payload["warnings"]:
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("note:")
    lines.append("- thresholds are derived from get_quality_thresholds() and therefore settings.toml.")
    return "\n".join(lines) + "\n"


def run_discovery_analysis() -> None:
    out_root = resolve_output_dir()
    out_dir = out_root / "change_point"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "change_point_report.txt"
    json_path = out_dir / "change_point_result.json"
    plot_path = out_dir / "change_point_histogram.png"
    source_path = out_root / "adsb_daily_summary_v2.csv"

    min_auc, min_minutes = get_quality_thresholds()
    generated_at = datetime.now().isoformat(timespec="seconds")
    total_rows, dropped_reasons, warnings = _compute_base_drop_reasons(source_path, min_auc, min_minutes)

    payload: dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "generated_at": generated_at,
        "source_path": str(source_path),
        "metric_name": METRIC_NAME,
        "thresholds": {
            "min_auc_n_used": int(min_auc),
            "min_minutes_covered": int(min_minutes),
        },
        "total_rows": int(total_rows),
        "usable_rows": 0,
        "dropped_rows": int(total_rows),
        "dropped_reasons": dropped_reasons,
        "detected_change_points": [],
        "summary_before_after": {},
        "warnings": warnings,
    }

    df = load_summary(path=str(source_path), min_auc=min_auc, min_minutes=min_minutes, require_proxy=True)
    if df is None or df.empty:
        payload["warnings"].append("no_data_after_load_summary")
        _safe_write_text(report_path, _build_report_text(payload))
        _safe_write_json(json_path, payload)
        print("No data. Exiting.")
        return

    before_dropna_rows = int(len(df))
    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["auc_n_used", "log_traffic"])
    after_dropna_rows = int(len(df))
    payload["dropped_reasons"]["dropna_auc_or_log_traffic_after_preprocess"] = before_dropna_rows - after_dropna_rows
    payload["usable_rows"] = after_dropna_rows
    payload["dropped_rows"] = max(payload["total_rows"] - payload["usable_rows"], 0)

    if not HAS_MCMC_DEPS:
        payload["warnings"].append(f"mcmc_dependencies_unavailable: {MCMC_IMPORT_ERROR}")
        _safe_write_text(report_path, _build_report_text(payload))
        _safe_write_json(json_path, payload)
        print("[WARN] MCMC dependencies unavailable; skipping inference.")
        return

    if len(df) < 5:
        payload["warnings"].append("usable_rows_lt_5_skip_change_point")
        _safe_write_text(report_path, _build_report_text(payload))
        _safe_write_json(json_path, payload)
        print("  WARNING: insufficient valid data; skipping change-point detection.")
        return

    y = jnp.array(df["auc_n_used"].values)
    log_traffic = jnp.array(df["log_traffic"].values)
    n_days = len(df)

    def model(y_values, log_traffic_values, n_days_value):
        tau = numpyro.sample("tau", dist.DiscreteUniform(0, n_days_value - 1))
        alpha_before = numpyro.sample("alpha_before", dist.Normal(10.0, 5.0))
        alpha_after = numpyro.sample("alpha_after", dist.Normal(10.0, 5.0))
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        alpha_inv = numpyro.sample("alpha_inv", dist.Exponential(1.0))

        idx = jnp.arange(n_days_value)
        intercept = jnp.where(idx < tau, alpha_before, alpha_after)
        mu = jnp.exp(intercept + beta_traffic * log_traffic_values)

        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, alpha_inv), obs=y_values)

    kernel = DiscreteHMCGibbs(NUTS(model))
    n_warmup = int(os.environ.get("ADSB_CP_WARMUP", "1000"))
    n_samples = int(os.environ.get("ADSB_CP_SAMPLES", "3000"))
    n_chains = int(os.environ.get("ADSB_CP_CHAINS", str(max(1, min(CPU_HOST, 4)))))
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=False,
    )

    print(
        "Searching change points on CPU... "
        f"(chains={n_chains}, devices={CPU_HOST}, warmup={n_warmup}, samples={n_samples})"
    )
    mcmc.run(random.PRNGKey(42), y, log_traffic, n_days)

    samples = mcmc.get_samples()
    tau_samples = np.asarray(samples["tau"])
    vals, counts = np.unique(tau_samples, return_counts=True)
    best_tau_idx = int(vals[np.argmax(counts)])
    detected_date = df.iloc[best_tau_idx]["date"]
    detected_date_str = detected_date.strftime("%Y-%m-%d") if pd.notna(detected_date) else ""
    confidence_pct = float(np.max(counts) / len(tau_samples) * 100)

    improvement = (np.exp(np.asarray(samples["alpha_after"] - samples["alpha_before"])) - 1) * 100
    before_series = df.loc[: best_tau_idx - 1, "auc_n_used"] if best_tau_idx > 0 else pd.Series(dtype=float)
    after_series = df.loc[best_tau_idx:, "auc_n_used"]

    payload["detected_change_points"] = [
        {
            "index": best_tau_idx,
            "date": detected_date_str,
            "confidence_pct": confidence_pct,
            "estimated_improvement_mean_pct": float(np.mean(improvement)),
        }
    ]
    payload["summary_before_after"] = {
        "before": _segment_summary(before_series),
        "after": _segment_summary(after_series),
    }

    print("-" * 30)
    print(f"[Detected change point]: {detected_date_str}")
    print(f"  Estimated improvement: {np.mean(improvement):+.2f}%")
    print(f"  Confidence: {confidence_pct:.1f}%")
    print("-" * 30)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df["date"].values[tau_samples.astype(int)],
        bins=n_days,
        color="skyblue",
        edgecolor="black",
    )
    ax.set_title("Detected Change Point Probability (Tsuchiura Station)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability Density")
    fig.tight_layout()
    try:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[OK] plot: {plot_path}")
    except Exception as exc:
        payload["warnings"].append(f"plot_save_failed: {exc}")
        print(f"[WARN] failed to save plot: {plot_path} ({exc})")
    finally:
        plt.close(fig)

    _safe_write_text(report_path, _build_report_text(payload))
    _safe_write_json(json_path, payload)


if __name__ == "__main__":
    run_discovery_analysis()
