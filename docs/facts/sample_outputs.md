# Sample Outputs

This document describes the current public smoke-output contract.

`output/sample/` and `sample_*` prefixed committed output files are no longer part of the current public tree.
Smoke reproducibility is now verified through normalized expected artifacts under `sample_data/smoke/expected/`.

## Versioned Expected Outputs (Smoke Contract)

`python scripts/tools/sample_data/freeze_expected_outputs.py --force`
writes deterministic expected files:

| File | Description |
|---|---|
| `manifest.normalized.csv` | Normalized inventory of exported files (path-stable, sorted) |
| `merged_for_ai.normalized.md` | Normalized merged markdown export (absolute paths normalized) |
| `merged_for_ai.zip.normalized.json` | Hash/size summary of `merged_for_ai.zip` entries after normalization |

`python scripts/tools/sample_data/verify_sample_outputs.py`
re-runs the same deterministic flow and compares against the three files above.

## Runtime Outputs Generated During Freeze/Verify

Freeze/verify internally runs:

```bash
python -m arena.artifact_cli run --legacy --deterministic --no-ai-export ...
```

That flow produces runtime files in a temporary work directory:

| Runtime file | Description |
|---|---|
| `manifest.csv` | Raw export manifest before normalization |
| `merged_for_ai.md` | Raw merged markdown export before normalization |
| `merged_for_ai.zip` | Zip archive containing export payloads |

These runtime files are not versioned as release fixtures.

## Representative Pipeline Outputs (Real Runs)

For real-data pipeline runs (`arena run`), representative outputs include:

- `output/<run>/adsb_daily_summary_v2.csv`
- `output/<run>/time_resolved/adsb_timebin_summary.csv`
- `output/<run>/performance/pipeline_runs.jsonl`
- `output/<run>/performance/phase_evaluator_report.txt`
- `output/<run>/change_point/change_point_report.txt`
- `output/<run>/coverage/coverage_trend.csv`
- `output/<run>/vertical_profile/los_efficiency_trend.csv`
- `output/<run>/fringe_decoding/fringe_decoding_stats.csv`
- `output/<run>/plao/distance_auc/plao_daily_distance_auc_summary.csv`
- `output/<run>/opensky_comparison/opensky_comparison_daily_summary.csv`

These are runtime artifacts and are intentionally not part of the versioned smoke fixture set.
