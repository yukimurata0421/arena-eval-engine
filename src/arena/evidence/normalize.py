from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from arena.evidence.schema import EvidenceRow, clean_float, clean_int
from arena.evidence.scoring import score_evidence_row


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _source(path: Path) -> str:
    return path.as_posix()


def _phase_label(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    return raw if raw else fallback


def _comparison_parts(label: str) -> tuple[str, str]:
    if " vs " not in label:
        return "", label.strip()
    left, right = label.split(" vs ", 1)
    return right.strip(), left.strip()


def _rank_probability_from_biserial(value: Any) -> float | None:
    effect = clean_float(value)
    if effect is None:
        return None
    return max(0.0, min(1.0, (effect + 1.0) / 2.0))


def _rank_probability_direction(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > 0.5:
        return "positive"
    if value < 0.5:
        return "negative"
    return "neutral"


def normalize_phase_evaluator(path: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for item in _read_csv(path):
        phase_idx = clean_int(item.get("phase_idx"))
        phase_name = _phase_label(item.get("phase_name"), f"phase_{phase_idx}")
        n_days = clean_int(item.get("n_days"))
        reliability = str(item.get("reliability") or "").casefold()
        base_warnings = []
        if reliability in {"preliminary", "reference"}:
            base_warnings.append(f"phase_reliability_{reliability}")

        if phase_idx not in (None, 0):
            rows.append(
                EvidenceRow(
                    comparison_id=f"bayes_phase:{phase_name}:vs_original_baseline",
                    source_file=_source(path),
                    source_metric="auc_n_used",
                    baseline_id="original_baseline",
                    phase_a="Phase 0",
                    phase_b=phase_name,
                    model_name="bayes_phase",
                    model_family="bayesian",
                    metric_family="coverage_auc",
                    effect_scale="posterior_ratio",
                    effect=clean_float(item.get("vs_baseline_mean_pct")),
                    ci_low=clean_float(item.get("vs_baseline_hdi94_lo")),
                    ci_high=clean_float(item.get("vs_baseline_hdi94_hi")),
                    ci_level=0.94,
                    prob_positive=clean_float(item.get("vs_baseline_prob_positive_pct")),
                    n_b=n_days,
                    n_effective=clean_float(n_days),
                    independence_unit="day",
                    autocorrelation_risk="medium",
                    warnings=list(base_warnings),
                    assumptions={
                        "family": "NegativeBinomial2",
                        "traffic_control": True,
                        "minutes_offset": True,
                    },
                    next_data_needed=[] if n_days and n_days >= 21 else ["more stable phase days"],
                )
            )

        alt_idx = clean_int(item.get("alt_baseline_idx"))
        alt_name = _phase_label(item.get("alt_baseline_name"), f"phase_{alt_idx}")
        if alt_idx is not None and phase_idx is not None and phase_idx != alt_idx:
            rows.append(
                EvidenceRow(
                    comparison_id=f"bayes_phase:{phase_name}:vs_alt_baseline",
                    source_file=_source(path),
                    source_metric="auc_n_used",
                    baseline_id=f"alt_baseline:{alt_name}",
                    phase_a=alt_name,
                    phase_b=phase_name,
                    model_name="bayes_phase",
                    model_family="bayesian",
                    metric_family="coverage_auc",
                    effect_scale="posterior_ratio",
                    effect=clean_float(item.get("vs_alt_mean_pct")),
                    ci_low=clean_float(item.get("vs_alt_hdi94_lo")),
                    ci_high=clean_float(item.get("vs_alt_hdi94_hi")),
                    ci_level=0.94,
                    prob_positive=clean_float(item.get("vs_alt_prob_positive_pct")),
                    n_b=n_days,
                    n_effective=clean_float(n_days),
                    independence_unit="day",
                    autocorrelation_risk="medium",
                    warnings=list(base_warnings),
                    caveats=["alt_baseline_comparison"],
                    assumptions={
                        "family": "NegativeBinomial2",
                        "traffic_control": True,
                        "minutes_offset": True,
                    },
                    next_data_needed=[] if n_days and n_days >= 21 else ["more days for post-baseline fine tuning"],
                )
            )
    return rows


def normalize_baseline_nb(path: Path) -> list[EvidenceRow]:
    data = _read_json(path)
    post_effect = data.get("post_effect")
    if not isinstance(post_effect, list):
        return []
    n = clean_int(data.get("n_observations"))
    rows: list[EvidenceRow] = []
    for item in post_effect:
        if not isinstance(item, dict):
            continue
        rows.append(
            EvidenceRow(
                comparison_id="baseline_nb_glm:post_vs_pre",
                source_file=_source(path),
                source_metric="auc_n_used",
                baseline_id="pre_post",
                phase_a="pre",
                phase_b="post",
                model_name="baseline_nb_glm",
                model_family="regression",
                metric_family="coverage_auc",
                effect_scale="coverage_auc_delta_pct",
                effect=clean_float(item.get("improvement_pct")),
                ci_low=clean_float(item.get("ci_lower")),
                ci_high=clean_float(item.get("ci_upper")),
                ci_level=0.95,
                p_value=clean_float(item.get("p_value")),
                n_effective=clean_float(n),
                independence_unit="day",
                autocorrelation_risk="medium",
                role="corroborating",
                assumptions={"family": "NegativeBinomial", "log_link": True, "traffic_control": True},
                caveats=["binary pre_post split collapses phase structure"],
            )
        )
    return rows


def normalize_bayesian_phase_2group(path: Path, *, model_name: str) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for item in _read_csv(path):
        comparison = str(item.get("comparison") or "").strip()
        phase_a, phase_b = _comparison_parts(comparison)
        rows.append(
            EvidenceRow(
                comparison_id=f"{model_name}:{comparison}",
                source_file=_source(path),
                source_metric="auc_n_used",
                baseline_id=phase_a or "comparison_baseline",
                phase_a=phase_a,
                phase_b=phase_b,
                model_name=model_name,
                model_family="bayesian",
                metric_family="coverage_auc",
                effect_scale="posterior_ratio",
                effect=clean_float(item.get("mean_improvement_pct")),
                ci_low=clean_float(item.get("hdi_94_lower")),
                ci_high=clean_float(item.get("hdi_94_upper")),
                ci_level=0.94,
                prob_positive=clean_float(item.get("prob_positive_pct")),
                independence_unit="day",
                autocorrelation_risk="medium",
                role="corroborating",
                assumptions={"family": "NegativeBinomial", "traffic_control": False},
                caveats=["uncontrolled two-group comparison", "n not embedded in source artifact"],
            )
        )
    return rows


def normalize_bayesian_phase(path: Path) -> list[EvidenceRow]:
    return normalize_bayesian_phase_2group(path, model_name="bayesian_2group")


def normalize_bayesian_phase_cuda(path: Path) -> list[EvidenceRow]:
    return normalize_bayesian_phase_2group(path, model_name="bayesian_2group_cuda")


def normalize_distance_performance(path: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for item in _read_csv(path):
        phase = _phase_label(item.get("Phase"), "target_phase")
        band = _phase_label(item.get("Distance_Band"), "distance_band")
        rank_probability = _rank_probability_from_biserial(item.get("Rank_Biserial_Effect"))
        common = {
            "source_file": _source(path),
            "source_metric": band,
            "comparison_window": band,
            "baseline_id": "distance_baseline",
            "phase_a": "baseline",
            "phase_b": phase,
            "model_family": "nonparametric",
            "metric_family": "distance_bin",
            "independence_unit": "day",
            "autocorrelation_risk": "medium",
            "role": "diagnostic",
            "assumptions": {"comparison_method": item.get("Comparison_Method")},
            "caveats": ["distance-band ratio diagnostic, not global performance conclusion"],
        }
        rows.append(
            EvidenceRow(
                comparison_id=f"distance_mwu_bootstrap:{phase}:{band}",
                model_name="distance_mwu_bootstrap",
                effect_scale="mean_delta_pct",
                effect=clean_float(item.get("Relative_Change_Pct")),
                p_value=clean_float(item.get("Mann_Whitney_P")),
                diagnostics={
                    "baseline_mean_pct": clean_float(item.get("Baseline_Mean_Pct")),
                    "target_mean_pct": clean_float(item.get("Target_Mean_Pct")),
                    "target_95ci": item.get("Target_95CI"),
                },
                warnings=["target_ci_not_delta_ci"],
                **common,
            )
        )
        rows.append(
            EvidenceRow(
                comparison_id=f"distance_mwu_hl_shift:{phase}:{band}",
                model_name="distance_mwu_hl",
                effect_scale="location_shift_pct",
                effect=clean_float(item.get("HL_Diff_Pct")),
                ci_low=clean_float(item.get("HL_95CI_Low_Pct")),
                ci_high=clean_float(item.get("HL_95CI_High_Pct")),
                ci_level=0.95,
                p_value=clean_float(item.get("Mann_Whitney_P")),
                diagnostics={
                    "rank_biserial_effect": clean_float(item.get("Rank_Biserial_Effect")),
                    "baseline_mean_pct": clean_float(item.get("Baseline_Mean_Pct")),
                    "target_mean_pct": clean_float(item.get("Target_Mean_Pct")),
                },
                **common,
            )
        )
        if rank_probability is not None:
            rows.append(
                EvidenceRow(
                    comparison_id=f"distance_mwu_rank_probability:{phase}:{band}",
                    model_name="distance_mwu_rank_probability",
                    effect_scale="rank_probability",
                    effect=rank_probability,
                    effect_direction=_rank_probability_direction(rank_probability),
                    p_value=clean_float(item.get("Mann_Whitney_P")),
                    diagnostics={
                        "rank_biserial_effect": clean_float(item.get("Rank_Biserial_Effect")),
                    },
                    **common,
                )
            )
    return rows


def normalize_time_bin_details(path: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for item in _read_csv(path):
        time_bin = _phase_label(item.get("time_bin"), "time_bin")
        rows.append(
            EvidenceRow(
                comparison_id=f"time_bin_detail:{time_bin}",
                source_file=_source(path),
                source_metric=time_bin,
                comparison_window=time_bin,
                baseline_id="old_new",
                phase_a="old",
                phase_b="new",
                model_name="time_bin_test",
                model_family="nonparametric",
                metric_family="time_bin",
                effect_scale="mean_delta_pct",
                effect=clean_float(item.get("improvement_pct")),
                p_value=clean_float(item.get("p_value")),
                independence_unit="time_bin_day",
                autocorrelation_risk="medium",
                role="diagnostic",
                assumptions={"time_of_day_stratified": True},
                caveats=["old_new split collapses phase structure"],
            )
        )
    return rows


def normalize_opensky_daily_capture_ratio(path: Path) -> list[EvidenceRow]:
    raw_rows = _read_csv(path)
    grouped: dict[str, list[float]] = {}
    for item in raw_rows:
        use_value = str(item.get("use_for_stats", "true")).strip().casefold()
        if use_value in {"false", "0", "no"}:
            continue
        phase = _phase_label(item.get("phase"), "")
        value = clean_float(item.get("median_capture_ratio"))
        if not phase or value is None:
            continue
        grouped.setdefault(phase, []).append(value)

    phases = sorted(grouped)
    rows: list[EvidenceRow] = []
    for i in range(len(phases) - 1):
        phase_a = phases[i]
        phase_b = phases[i + 1]
        vals_a = grouped[phase_a]
        vals_b = grouped[phase_b]
        rows.append(
            EvidenceRow(
                comparison_id=f"opensky_capture_ratio:{phase_b}:vs_{phase_a}",
                source_file=_source(path),
                source_metric="median_capture_ratio",
                baseline_id=phase_a,
                phase_a=phase_a,
                phase_b=phase_b,
                model_name="opensky_daily_capture_ratio",
                model_family="proxy",
                metric_family="capture_ratio",
                effect_scale="capture_ratio_delta",
                effect=mean(vals_b) - mean(vals_a),
                n_a=len(vals_a),
                n_b=len(vals_b),
                independence_unit="day",
                autocorrelation_risk="medium",
                role="proxy",
                diagnostics={
                    "phase_a_mean": mean(vals_a),
                    "phase_b_mean": mean(vals_b),
                },
                warnings=["proxy_metric"],
                caveats=["OpenSky capture ratio can contradict coverage AUC and should remain diagnostic"],
                next_data_needed=["traffic-normalized distance-bin evidence"],
            )
        )
    return rows


def _latest_daily_signature_dir(performance: Path) -> Path | None:
    root = performance / "change_points"
    dirs = sorted(path for path in root.glob("daily_signature_cp_*") if path.is_dir())
    return dirs[-1] if dirs else None


def _signature_metric_family(series_name: str, *, source_kind: str) -> str:
    if source_kind == "traffic_test" or series_name == "traffic_count":
        return "traffic_proxy"
    if series_name in {"daily_auc", "daily_auc_adjusted"}:
        return "coverage_auc"
    if series_name.startswith("quantile_signature."):
        return "coverage_signature"
    return "coverage_signature"


def _signature_role(metric_family: str) -> str:
    if metric_family == "coverage_auc":
        return "corroborating"
    if metric_family == "traffic_proxy":
        return "proxy"
    return "diagnostic"


def normalize_daily_signature_change_points(performance: Path) -> list[EvidenceRow]:
    latest = _latest_daily_signature_dir(performance)
    if latest is None:
        return []

    rows: list[EvidenceRow] = []
    detector_path = latest / "change_point_results.csv"
    for item in _read_csv(detector_path):
        series_name = _phase_label(item.get("series_name"), "series")
        change_date = _phase_label(item.get("detected_change_date"), "unknown_date")
        metric_family = _signature_metric_family(series_name, source_kind=str(item.get("adjustment") or "detector"))
        rows.append(
            EvidenceRow(
                comparison_id=f"daily_signature_change_detector:{series_name}:{change_date}:{item.get('adjustment')}",
                source_file=_source(detector_path),
                source_metric=series_name,
                comparison_window=change_date,
                baseline_id=f"change_point:{change_date}",
                phase_a="before_change",
                phase_b="after_change",
                model_name="daily_signature_change_detector",
                model_family="temporal",
                metric_family=metric_family,
                effect_scale="change_score",
                effect=clean_float(item.get("score")),
                effect_direction="neutral",
                n_effective=clean_float(item.get("n_obs")),
                independence_unit="day",
                autocorrelation_risk="high",
                role="diagnostic",
                diagnostics={
                    "series_type": item.get("series_type"),
                    "adjustment": item.get("adjustment"),
                    "detection_method": item.get("detection_method"),
                    "split_index": clean_int(item.get("split_index")),
                    "notes": item.get("notes"),
                },
                caveats=["change point score locates candidate boundary; it is not an improvement direction"],
            )
        )

    result_files = {
        "traffic_adjusted": latest / "traffic_adjusted_results.csv",
        "traffic_unadjusted": latest / "traffic_unadjusted_results.csv",
        "traffic_test": latest / "traffic_test_results.csv",
    }
    for source_kind, path in result_files.items():
        for item in _read_csv(path):
            series_name = _phase_label(item.get("series_name"), "series")
            change_date = _phase_label(item.get("detected_change_date"), "unknown_date")
            metric_family = _signature_metric_family(series_name, source_kind=source_kind)
            warnings = ["autocorrelation_risk_high"]
            if source_kind == "traffic_unadjusted":
                warnings.append("traffic_unadjusted")
            if source_kind == "traffic_test":
                warnings.append("proxy_metric")
            rows.append(
                EvidenceRow(
                    comparison_id=f"daily_signature_mwu_hl:{source_kind}:{series_name}:{change_date}",
                    source_file=_source(path),
                    source_metric=series_name,
                    comparison_window=change_date,
                    baseline_id=f"change_point:{change_date}",
                    phase_a="before_change",
                    phase_b="after_change",
                    model_name=f"daily_signature_mwu_hl_{source_kind}",
                    model_family="nonparametric" if metric_family != "traffic_proxy" else "proxy",
                    metric_family=metric_family,
                    effect_scale="location_shift",
                    effect=clean_float(item.get("hl_or_pim_shift")),
                    ci_low=clean_float(item.get("ci_low")),
                    ci_high=clean_float(item.get("ci_high")),
                    ci_level=0.95,
                    p_value=clean_float(item.get("p_value")),
                    n_a=clean_int(item.get("n_before")),
                    n_b=clean_int(item.get("n_after")),
                    independence_unit="day",
                    autocorrelation_risk="high",
                    role=_signature_role(metric_family),
                    assumptions={"comparison_method": item.get("comparison_method"), "source_kind": source_kind},
                    diagnostics={
                        "rank_biserial_effect": clean_float(item.get("effect_size")),
                        "notes": item.get("notes"),
                    },
                    warnings=warnings,
                    caveats=["change-point split is data-selected; treat as validation target unless corroborated"],
                    next_data_needed=["confirm change point on held-out days or phase-defined windows"],
                )
            )
            rank_probability = _rank_probability_from_biserial(item.get("effect_size"))
            if rank_probability is not None:
                rows.append(
                    EvidenceRow(
                        comparison_id=f"daily_signature_rank_probability:{source_kind}:{series_name}:{change_date}",
                        source_file=_source(path),
                        source_metric=series_name,
                        comparison_window=change_date,
                        baseline_id=f"change_point:{change_date}",
                        phase_a="before_change",
                        phase_b="after_change",
                        model_name=f"daily_signature_rank_probability_{source_kind}",
                        model_family="nonparametric" if metric_family != "traffic_proxy" else "proxy",
                        metric_family=metric_family,
                        effect_scale="rank_probability",
                        effect=rank_probability,
                        effect_direction=_rank_probability_direction(rank_probability),
                        p_value=clean_float(item.get("p_value")),
                        n_a=clean_int(item.get("n_before")),
                        n_b=clean_int(item.get("n_after")),
                        independence_unit="day",
                        autocorrelation_risk="high",
                        role=_signature_role(metric_family),
                        assumptions={"comparison_method": item.get("comparison_method"), "source_kind": source_kind},
                        diagnostics={"rank_biserial_effect": clean_float(item.get("effect_size"))},
                        warnings=warnings,
                        caveats=["rank probability derived from MWU; not independent from HL shift"],
                        next_data_needed=["confirm change point on held-out days or phase-defined windows"],
                    )
                )
    return rows


def load_evidence_rows(output_root: Path) -> list[EvidenceRow]:
    performance = output_root / "performance"
    normalizers = [
        (normalize_phase_evaluator, performance / "phase_evaluator_results.csv"),
        (normalize_baseline_nb, performance / "baseline_nb_results.json"),
        (normalize_bayesian_phase, performance / "bayesian_phase_results.csv"),
        (normalize_bayesian_phase_cuda, performance / "bayesian_phase_results_cuda.csv"),
        (normalize_distance_performance, performance / "distance_performance_summary.csv"),
        (normalize_time_bin_details, performance / "time_bin_detailed_stats.csv"),
        (normalize_opensky_daily_capture_ratio, output_root / "opensky_comparison" / "opensky_comparison_daily_summary.csv"),
    ]

    rows: list[EvidenceRow] = []
    for normalize, path in normalizers:
        rows.extend(normalize(path))
    rows.extend(normalize_daily_signature_change_points(performance))
    return [score_evidence_row(row) for row in rows]
