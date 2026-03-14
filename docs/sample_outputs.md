# Sample Outputs

Example output artifacts are provided in `output/sample/`.
These are generated from the public sample dataset and demonstrate ARENA's output structure.
They are not intended for statistical interpretation — the sample dataset is too small for that.

## Daily Summaries

| File | Description |
|---|---|
| `sample_adsb_daily_summary_v2.csv` | Primary daily aggregation: AUC, minutes covered, traffic proxies, hardware phase, unique hex counts per distance bin |
| `sample_adsb_daily_summary_raw.csv` | Raw daily summary before quality filtering |
| `sample_adsb_daily_summary.csv` | Legacy daily summary format |
| `sample_adsb_signal_daily_summary.csv` | Daily signal strength statistics (RSSI, noise floor) |
| `sample_adsb_signal_range_summary.csv` | Signal quality by distance range |

## Performance Evaluation

| File | Description |
|---|---|
| `sample_performance_phase_evaluator_report.txt` | Dual-baseline phase evaluation report (NumPyro NUTS): improvement estimates vs original and alt baselines with HDI and P(>0) |
| `sample_performance_phase_evaluator_results.csv` | Structured phase evaluation results: per-phase improvement percentages, HDI bounds, reliability ratings |
| `sample_performance_bayesian_phase_results_cuda.csv` | Bayesian pairwise phase comparison results (CUDA/CPU) |
| `sample_performance_bayesian_phase_results.csv` | Bayesian phase results (CPU-only variant) |
| `sample_performance_baseline_nb_summary.txt` | Negative Binomial GLM baseline model summary (statsmodels output) |
| `sample_performance_distance_binomial_summary.csv` | Distance-bin binomial test results |
| `sample_performance_distance_performance_summary.csv` | Per-distance-bin performance comparison across phases |
| `sample_performance_quality_binomial_summary.txt` | Quality metric binomial GLM summary |
| `sample_performance_time_bin_detailed_stats.csv` | Time-of-day bin statistics for diurnal pattern analysis |
| `sample_performance_pipeline_runs.jsonl` | Pipeline execution audit log (append-only JSONL) |

## Coverage and Spatial

| File | Description |
|---|---|
| `sample_coverage_coverage_trend.csv` | Daily coverage area (P95 distance, area estimate) |
| `sample_vertical_profile_los_efficiency_trend.csv` | Line-of-sight efficiency trend by altitude band |
| `sample_time_resolved_adsb_timebin_summary.csv` | AUC and traffic by time-of-day bin, used for diurnal correction |

## Fringe Decoding

| File | Description |
|---|---|
| `sample_fringe_decoding_fringe_decoding_stats.csv` | Fringe decoding quality statistics by distance band |
| `sample_fringe_decoding_fringe_decoding_trend.csv` | Fringe decoding quality trend over time |
| `sample_fringe_decoding_statistical_report.txt` | Statistical analysis of fringe decoding patterns |

## OpenSky Comparison

| File | Description |
|---|---|
| `sample_opensky_comparison_opensky_comparison_daily_summary.csv` | Daily comparison between local reception and OpenSky Network traffic |
| `sample_opensky_comparison_opensky_comparison_stats_report.txt` | Statistical summary of local vs OpenSky reception overlap |
| `sample_opensky_comparison_opensky_local_minutely_merged.csv` | Minute-level merged local and OpenSky data |
| `sample_opensky_comparison_opensky_skipped_days.csv` | Days excluded from OpenSky comparison (data quality or availability) |

## PLAO Distance AUC

| File | Description |
|---|---|
| `sample_plao_distance_auc_plao_daily_distance_auc_summary.csv` | Daily AUC from PLAO position data (independent data source) |
| `sample_plao_distance_auc_plao_daily_distance_auc_long.csv` | Long-format daily AUC for time-series analysis |
| `sample_plao_distance_auc_plao_distance_auc_stats_report.txt` | Statistical comparison of PLAO AUC across phases |
| `sample_plao_distance_auc_plao_skipped_days.csv` | Days excluded from PLAO analysis |

## Artifact Export

| File | Description |
|---|---|
| `sample_merged_for_ai_manifest.csv` | Manifest of artifacts selected for AI review export |

## Full Output Paths

When running against real data, `arena run` produces outputs at paths like:

- `output/adsb_daily_summary_v2.csv`
- `output/coverage/coverage_trend.csv`
- `output/performance/phase_evaluator_report.txt`
- `output/performance/bayesian_phase_results_cuda.csv`
- `output/performance/pipeline_runs.jsonl`
- `output/change_point/change_point_report.txt`
- `output/change_point/multi_change_points_report.txt`
- `output/time_resolved/adsb_timebin_summary.csv`
- `output/opensky_comparison/opensky_comparison_daily_summary.csv`
- `output/heatmaps/pos_YYYYMMDD_heatmap.html`

Generated `output/performance/` artifacts are local runtime outputs and are not versioned in the public repository.
