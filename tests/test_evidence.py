from __future__ import annotations

import csv
import json
from pathlib import Path

from arena.evidence.claim_router import build_claim_routes, render_disagreement_report
from arena.evidence.normalize import load_evidence_rows
from arena.evidence.schema import EvidenceRow
from arena.evidence.scoring import n_reliability_tag, score_evidence_row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_n_reliability_tags_do_not_use_definitive() -> None:
    assert n_reliability_tag(2) == "reference_only"
    assert n_reliability_tag(3) == "trend_only"
    assert n_reliability_tag(7) == "usable"
    assert n_reliability_tag(14) == "likely"
    assert n_reliability_tag(21) == "strong"


def test_score_keeps_component_breakdown() -> None:
    row = score_evidence_row(
        EvidenceRow(
            comparison_id="x",
            model_name="bayes_phase",
            model_family="bayesian",
            metric_family="coverage_auc",
            effect_scale="posterior_ratio",
            effect=12.0,
            ci_low=4.0,
            ci_high=20.0,
            prob_positive=97.0,
            n_effective=12,
            independence_unit="day",
        )
    )
    assert row.role == "primary"
    assert row.reliability_tag in {"usable", "likely", "strong"}
    assert row.evidence_score > 0
    assert set(row.score_components) == {
        "applicability_score",
        "precision_score",
        "assumption_score",
        "proxy_score",
        "practical_effect_score",
    }


def test_claim_router_preserves_cross_metric_conflict() -> None:
    rows = [
        score_evidence_row(
            EvidenceRow(
                comparison_id="coverage",
                model_name="bayes_phase",
                model_family="bayesian",
                metric_family="coverage_auc",
                effect_scale="posterior_ratio",
                effect=20.0,
                ci_low=5.0,
                ci_high=35.0,
                prob_positive=98.0,
                n_effective=14,
                baseline_id="airspy",
                phase_a="Airspy",
                phase_b="Adapter",
                independence_unit="day",
            )
        ),
        score_evidence_row(
            EvidenceRow(
                comparison_id="capture",
                model_name="opensky_daily_capture_ratio",
                model_family="proxy",
                metric_family="capture_ratio",
                effect_scale="capture_ratio_delta",
                effect=-0.04,
                n_a=8,
                n_b=8,
                baseline_id="airspy",
                phase_a="Airspy",
                phase_b="Adapter",
                independence_unit="day",
                role="proxy",
                warnings=["proxy_metric"],
            )
        ),
    ]
    routes = build_claim_routes(rows)
    assert routes["validation_targets"]
    assert routes["claims"][0]["status"] == "supported_but_qualified"
    assert routes["claims"][0]["counter_evidence"]
    assert routes["claims"][0]["primary_metric_family"] == "coverage_auc"


def test_load_evidence_rows_from_existing_outputs(tmp_path: Path) -> None:
    out = tmp_path / "output"
    performance = out / "performance"
    _write_csv(
        performance / "phase_evaluator_results.csv",
        [
            {
                "phase_idx": 1,
                "phase_name": "Airspy",
                "phase_date": "2026-01-14",
                "n_days": 12,
                "reliability": "definitive",
                "vs_baseline_mean_pct": 45.0,
                "vs_baseline_hdi94_lo": 10.0,
                "vs_baseline_hdi94_hi": 80.0,
                "vs_baseline_prob_positive_pct": 99.0,
                "vs_previous_mean_pct": 45.0,
                "vs_previous_hdi94_lo": 10.0,
                "vs_previous_hdi94_hi": 80.0,
                "vs_previous_prob_positive_pct": 99.0,
                "alt_baseline_idx": 1,
                "alt_baseline_name": "Airspy",
                "vs_alt_mean_pct": 0.0,
                "vs_alt_hdi94_lo": 0.0,
                "vs_alt_hdi94_hi": 0.0,
                "vs_alt_prob_positive_pct": 0.0,
            }
        ],
    )
    (performance / "baseline_nb_results.json").write_text(
        json.dumps(
            {
                "model": "NegativeBinomial GLM",
                "n_observations": 30,
                "post_effect": [
                    {
                        "improvement_pct": 35.0,
                        "ci_lower": 5.0,
                        "ci_upper": 65.0,
                        "p_value": 0.02,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        out / "opensky_comparison" / "opensky_comparison_daily_summary.csv",
        [
            {"date": "2026-01-01", "phase": "Airspy", "use_for_stats": "true", "median_capture_ratio": 1.2},
            {"date": "2026-01-02", "phase": "Airspy", "use_for_stats": "true", "median_capture_ratio": 1.1},
            {"date": "2026-01-03", "phase": "Adapter", "use_for_stats": "true", "median_capture_ratio": 1.0},
            {"date": "2026-01-04", "phase": "Adapter", "use_for_stats": "true", "median_capture_ratio": 1.0},
        ],
    )

    rows = load_evidence_rows(out)
    assert {row.metric_family for row in rows} >= {"coverage_auc", "capture_ratio"}
    assert all(row.reliability_tag != "definitive" for row in rows)


def test_evidence_synthesizer_writes_artifacts(tmp_path: Path) -> None:
    from scripts.adsb.analysis.meta.adsb_model_evidence_synthesizer import main, run_evidence_synthesis

    out = tmp_path / "output"
    result = run_evidence_synthesis(output_root=out)
    for path in result.values():
        assert Path(path).exists()
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    assert summary["summary"]["evidence_row_count"] == 0
    assert main(["--output-root", str(out)]) == 0


def test_render_disagreement_report_for_empty_and_conflicting_routes() -> None:
    assert "No metric-family disagreements detected." in render_disagreement_report({"validation_targets": []})

    report = render_disagreement_report(
        {
            "validation_targets": [
                {
                    "conflict_group": "baseline|old|new|default_window",
                    "reason": "metric_family_direction_conflict",
                    "needed_data": "re-check proxy assumptions",
                    "metric_signs": {"coverage_auc": 1, "capture_ratio": -1},
                    "counter_evidence": [
                        {
                            "model_name": "opensky_daily_capture_ratio",
                            "metric_family": "capture_ratio",
                            "effect_direction": "negative",
                            "effect": -0.04,
                        }
                    ],
                }
            ]
        }
    )
    assert "coverage_auc: positive" in report
    assert "capture_ratio: negative" in report
    assert "opensky_daily_capture_ratio capture_ratio negative effect=-0.04" in report


def test_scoring_edge_cases_are_reviewable() -> None:
    invalid = score_evidence_row(
        EvidenceRow(
            comparison_id="invalid",
            model_name="failed_model",
            model_family="invalid",
            metric_family="coverage_auc",
            effect_scale="posterior_ratio",
            effect=10.0,
        )
    )
    assert invalid.role == "invalid"
    assert invalid.reliability_tag == "invalid"
    assert invalid.evidence_score == 0.0

    weak = score_evidence_row(
        EvidenceRow(
            comparison_id="weak",
            model_name="weak_model",
            model_family="regression",
            metric_family="coverage_signature",
            effect_scale="rank_probability",
            effect=0.51,
            p_value=0.8,
            ci_low=-1.0,
            ci_high=1.0,
            n_a=2,
            n_b=2,
            warnings=["autocorrelation", "prior_sensitivity_medium", "target_ci_not_delta_ci"],
            caveats=["wide interval crosses zero"],
        )
    )
    assert weak.role == "diagnostic"
    assert weak.reliability_tag in {"reference_only", "trend_only"}
    assert 0.0 < weak.evidence_score < 0.1


def test_load_evidence_rows_covers_distance_time_bayes_and_change_points(tmp_path: Path) -> None:
    out = tmp_path / "output"
    performance = out / "performance"
    _write_csv(
        performance / "bayesian_phase_results.csv",
        [
            {
                "comparison": "Airspy vs Adapter",
                "mean_improvement_pct": 12.5,
                "hdi_94_lower": 2.0,
                "hdi_94_upper": 25.0,
                "prob_positive_pct": 96.0,
            }
        ],
    )
    _write_csv(
        performance / "bayesian_phase_results_cuda.csv",
        [
            {
                "comparison": "Airspy vs Adapter",
                "mean_improvement_pct": 11.0,
                "hdi_94_lower": 1.0,
                "hdi_94_upper": 23.0,
                "prob_positive_pct": 94.0,
            }
        ],
    )
    _write_csv(
        performance / "distance_performance_summary.csv",
        [
            {
                "Phase": "Adapter",
                "Distance_Band": "0-50km",
                "Comparison_Method": "mwu_hl",
                "Relative_Change_Pct": 8.0,
                "Mann_Whitney_P": 0.01,
                "Baseline_Mean_Pct": 40.0,
                "Target_Mean_Pct": 48.0,
                "Target_95CI": "[45, 51]",
                "HL_Diff_Pct": 7.5,
                "HL_95CI_Low_Pct": 2.5,
                "HL_95CI_High_Pct": 12.0,
                "Rank_Biserial_Effect": 0.4,
            }
        ],
    )
    _write_csv(
        performance / "time_bin_detailed_stats.csv",
        [
            {
                "time_bin": "night",
                "improvement_pct": -3.0,
                "p_value": 0.2,
            }
        ],
    )

    cp_dir = performance / "change_points" / "daily_signature_cp_20260603"
    _write_csv(
        cp_dir / "change_point_results.csv",
        [
            {
                "series_name": "daily_auc",
                "detected_change_date": "2026-01-14",
                "adjustment": "traffic_adjusted",
                "score": 4.2,
                "n_obs": 30,
                "series_type": "daily",
                "detection_method": "binary_segmentation",
                "split_index": 10,
                "notes": "candidate",
            }
        ],
    )
    result_row = {
        "series_name": "daily_auc",
        "detected_change_date": "2026-01-14",
        "hl_or_pim_shift": 0.08,
        "ci_low": 0.02,
        "ci_high": 0.14,
        "p_value": 0.03,
        "n_before": 12,
        "n_after": 18,
        "comparison_method": "mwu_hl",
        "effect_size": 0.3,
        "notes": "corroborates phase boundary",
    }
    _write_csv(cp_dir / "traffic_adjusted_results.csv", [result_row])
    _write_csv(cp_dir / "traffic_unadjusted_results.csv", [{**result_row, "series_name": "quantile_signature.q90"}])
    _write_csv(cp_dir / "traffic_test_results.csv", [{**result_row, "series_name": "traffic_count"}])

    rows = load_evidence_rows(out)
    by_name = {row.model_name for row in rows}
    assert {"bayesian_2group", "bayesian_2group_cuda", "distance_mwu_hl", "time_bin_test"} <= by_name
    assert "daily_signature_change_detector" in by_name
    assert "daily_signature_mwu_hl_traffic_test" in by_name
    assert "daily_signature_rank_probability_traffic_adjusted" in by_name
    assert {row.metric_family for row in rows} >= {"distance_bin", "time_bin", "traffic_proxy", "coverage_signature"}
