# Why ARENA Does Not Use Weighted Ensemble

Status: accepted
Decision date: 2026-06-03 JST
Scope: Stage 9 evidence synthesis, ADS-B receiver performance claims
Source of truth: `src/arena/evidence/`, `scripts/adsb/analysis/meta/adsb_model_evidence_synthesizer.py`

## Purpose

This document fixes the design reason why ARENA does not average model outputs into one weighted score.

ARENA's job is not to hide disagreement behind a clean number. Its job is to preserve support, counter-evidence, caveats, validation targets, and next-data-needed fields so a reviewer can see what is supported and what still needs verification.

## Context

ARENA evaluates receiver changes with several metric families.

| metric family | What it measures | Typical failure mode |
| --- | --- | --- |
| `coverage_auc` | time-integrated local coverage | traffic volume can masquerade as receiver gain |
| `rank_probability` | non-parametric dominance probability | easy to confuse with effect magnitude |
| `location_shift` | robust distribution shift | CI can be wide under small N |
| `posterior_ratio` | Bayesian phase ratio | depends on priors, sampling, convergence |
| `capture_ratio` | OpenSky-normalized proxy capture | depends on OpenSky coverage and distance-bin normalization |

These metrics can all point toward "improvement", but they do not measure the same target. Weighted averaging would erase the different assumptions and failure modes.

## Decision

ARENA does not use weighted ensemble for cross-family evidence.

Instead:

```text
model output
  -> EvidenceRow
  -> Claim Router
  -> support / counter_evidence / caveat / validation_target / next_data_needed
```

`evidence_score` is not a final truth score. It is a routing and review-priority aid with component breakdowns.

## Why Averaging Fails Here

Example:

```text
coverage_auc: positive
capture_ratio: negative
```

A weighted ensemble might turn this into a weak positive or neutral score. That is the wrong behavior. The important fact is the disagreement itself:

```text
local coverage improved, but OpenSky proxy capture moved the other way
```

That should become a validation target:

```text
re-check proxy assumptions
re-check distance-bin traffic normalization
check phase boundaries
check local data freshness
```

## Non-Goals

ARENA intentionally does not:

```text
- average proxy evidence with direct local evidence
- count same-input models as independent proof
- treat small-N robust statistics as strong evidence
- discard contradictory rows
- let a single score replace caveats or next-data-needed fields
```

## EvidenceRow Contract

Every normalized row must preserve enough information for audit:

| field | contract |
| --- | --- |
| `source_file` | source artifact path |
| `source_metric` | original metric name |
| `model_family` | method family |
| `metric_family` | measurement target |
| `effect_scale` | unit or interpretation of effect |
| `effect_direction` | positive / negative / neutral |
| `n_effective` | sample-size basis for reliability |
| `reliability_tag` | `reference_only`, `trend_only`, `usable`, `likely`, `strong` |
| `role` | primary / corroborating / proxy / contradictory |
| `diagnostics` | model or convergence diagnostics |
| `caveats` | interpretive constraints |
| `next_data_needed` | follow-up data or validation |

## Consequences

Positive:

```text
- metric-family conflicts remain visible
- reviewer can inspect counter-evidence directly
- proxy-quality failures are not hidden
- small-N uncertainty stays attached to claims
- new model families can join by implementing EvidenceRow normalization
```

Tradeoffs:

```text
- reports are longer than a single leaderboard score
- readers need the disagreement report contract
- downstream review tooling must preserve counter-evidence
```

## Related Docs

- `docs/principles/evidence-synthesis-stage.md`
- `docs/facts/examples/model_disagreement_report_example.md`
- `docs/facts/performance/mcmc-parallelism-and-gpu-policy.md`
