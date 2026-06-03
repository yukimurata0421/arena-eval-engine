# Evidence Synthesis Stage Contract

Status: accepted
Decision date: 2026-06-03 JST
Scope: `arena run` Stage 9, evidence matrix, claim routing, disagreement report
Source of truth: `src/arena/pipeline/stages.py`, `src/arena/evidence/`

## Purpose

Stage 9 turns the output of multiple statistical stages into a reviewable evidence structure.

It does not replace the statistical models. It runs after the existing analysis stages and preserves their agreement, disagreement, caveats, and validation needs.

## Pipeline Position

Current public `arena run` has 9 stages:

```text
1 Aggregation
2 Spatial / visualization
3 Statistics (CPU)
4 Phase evaluation
5 Bayesian / change point
6 Final report
7 PLAO
8 OpenSky comparison
9 Evidence synthesis
```

Stage 9 is deliberately placed after Stage 8 so it can read local metrics, Bayesian results, change-point diagnostics, PLAO outputs, and OpenSky proxy outputs together.

## Outputs

Stage 9 writes:

| artifact | purpose |
| --- | --- |
| `output/performance/model_evidence_matrix.csv` | row-level `EvidenceRow` table |
| `output/performance/model_evidence_summary.json` | machine-readable claim routes and validation targets |
| `output/performance/model_disagreement_report.md` | reviewer-readable disagreement report |

## Ownership Boundary

Stage 9 owns:

```text
- reading existing analysis artifacts
- normalizing rows into EvidenceRow
- scoring evidence components
- assigning evidence roles
- grouping rows into conflict groups
- routing support and counter-evidence
- rendering validation targets
```

Stage 9 does not own:

```text
- raw ADS-B ingestion
- phase timeline definition
- individual model fitting
- OpenSky API availability
- final human acceptance of a claim
```

## Claim Router Contract

Claim Router must preserve:

```text
support
counter_evidence
caveats
validation_targets
next_data_needed
```

It must not:

```text
- drop contradictory rows
- average direct and proxy metrics
- remove small-N caveats
- treat evidence_score as truth
- hide metric_family_direction_conflict
```

## Validation Target Example

```text
reason: metric_family_direction_conflict
coverage_auc: positive
capture_ratio: negative
needed_data: re-check proxy assumptions and distance-bin traffic normalization
```

This is not a pipeline failure. It is an explicit follow-up target.

## Current Real-Data Evidence

On the Dell workstation, Stage 9 has been verified against real data with:

```text
Evidence rows: 130
Claim candidates: 46
Validation targets: 1
Invalid: 0
```

## Related Docs

- `docs/principles/why-not-weighted-ensemble.md`
- `docs/facts/examples/model_disagreement_report_example.md`
- `docs/facts/performance/mcmc-parallelism-and-gpu-policy.md`
