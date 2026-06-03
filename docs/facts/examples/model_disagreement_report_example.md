# Model Disagreement Report Example

Status: example snapshot
Snapshot date: 2026-06-03 JST
Generated artifact path: `output/performance/model_disagreement_report.md`

## Purpose

This example shows how ARENA preserves model disagreement instead of averaging it away.

`output/` is ignored by Git, so this tracked example documents how reviewers should read the generated artifact.

## Example

```text
# Model Evidence Disagreement Report

This report preserves metric-family disagreements as validation targets instead of averaging them away.

## Airspy Mini|Airspy Mini|Airspy+Cable|default_window

- reason: metric_family_direction_conflict
- needed_data: re-check proxy assumptions and distance-bin traffic normalization
- capture_ratio: negative
- coverage_auc: positive
- counter_evidence:
  - opensky_daily_capture_ratio capture_ratio negative effect=-0.04677164942380441
```

## Reading Contract

| field | meaning |
| --- | --- |
| heading | conflict-group key |
| `reason` | why this became a validation target |
| `needed_data` | next data or assumption to check |
| `capture_ratio` | OpenSky proxy metric direction |
| `coverage_auc` | local coverage metric direction |
| `counter_evidence` | evidence that qualifies or contradicts the claim |

## What To Do

Use this report to decide the next validation step:

```text
1. Read reason.
2. Identify the conflicting metric families.
3. Inspect counter_evidence source rows in model_evidence_matrix.csv.
4. Add needed_data to the review queue or follow-up plan.
```

Do not average positive and negative directions into a neutral score.

## Related Docs

- `docs/principles/why-not-weighted-ensemble.md`
- `docs/principles/evidence-synthesis-stage.md`
