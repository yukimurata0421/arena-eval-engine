# AEME — Aerial Evaluation & Measurement Engine

AEME is the analytical core of ARENA.

Its role is to determine whether observed changes in ADS-B receiver performance are real,
quantitatively meaningful, robust across methods, and reproducible under uncertainty.

AEME uses multiple complementary statistical approaches rather than a single metric or model. No
single method is treated as authoritative — conclusions are drawn from convergence or divergence
across models.

AEME also does not collapse model families into a weighted ensemble score. Coverage AUC, rank
probability, location shift, posterior ratio, and OpenSky capture ratio all describe different
evidence targets. Averaging them would hide contradictions, proxy-quality issues, and small-N
uncertainty. ARENA preserves those outputs as `EvidenceRow` records and routes them into support,
counter-evidence, caveats, validation targets, and next-data-needed fields.

For detailed model-by-model assumptions, limitations, and evidence boundaries, see
[statistical-assumptions-and-limitations.md](./statistical-assumptions-and-limitations.md).

---

## Why Multiple Methods

A receiver upgrade that increases aircraft counts does not prove the receiver improved. Traffic
volume, time of day, seasonal patterns, and operational variation can produce the same effect.

A single statistical test cannot distinguish these causes. AEME addresses this by running several
methods in parallel, each with different assumptions and different failure modes, and checking
whether they agree.

| Method | What it detects | Where it fails |
|---|---|---|
| Negative Binomial GLM | Overall trend with traffic offset | Assumes linear log-link; misses phase-specific effects |
| Bayesian Phase Evaluation (NumPyro NUTS) | Phase-level improvement with posterior uncertainty | Requires sufficient days per phase; sensitive to prior specification |
| Mann-Whitney U + Hodges-Lehmann CI | Distribution shift without parametric assumptions | Low power with small samples; no confounder control |
| Distance-bin NB-GLM with OpenSky | Per-distance-bin improvement with external traffic control | Depends on OpenSky data quality and coverage overlap |
| Binomial GLM (Quality) | Proportion-based quality metric change | Only captures binary quality, not magnitude |
| Change-point detection | Structural break timing | Can detect spurious breaks from traffic shifts |
| Time-resolved evaluation | Time-of-day and seasonal patterns | Requires sufficient data density per time bin |

When most methods agree on the direction and approximate magnitude of an effect, the conclusion is
defensible. When they disagree, AEME reports the disagreement rather than choosing a preferred
result.

The Stage 9 evidence synthesis layer is where this becomes an artifact contract:
`model_evidence_matrix.csv`, `model_evidence_summary.json`, and `model_disagreement_report.md`
preserve support and counter-evidence side by side. See
[why-not-weighted-ensemble.md](./why-not-weighted-ensemble.md) and
[evidence-synthesis-stage.md](./evidence-synthesis-stage.md).

---

## Core Design Principles

### 1. Coverage AUC over maximum distance

Maximum reception distance is not a reliable performance metric. A single far-away aircraft at high
altitude inflates the number without reflecting the receiver's sustained performance across its
coverage area.

AEME uses time-integrated coverage AUC (area under the curve of distance over time) as the primary
metric. This concept is borrowed from pharmacokinetics, where AUC measures total drug exposure
rather than peak concentration. The same principle applies: sustained reception across the coverage
area matters more than a single peak measurement.

### 2. Dual-baseline system

A major hardware change (RTL-SDR → Airspy Mini) produces a dominant improvement that masks smaller
subsequent changes (cable upgrades, adapter changes). If all phases are compared only against the
original baseline, every post-Airspy phase shows a large positive effect — but the incremental gains
between cable and adapter changes are invisible.

AEME solves this with a dual-baseline evaluator: one comparison against the original baseline (Phase
0) and one against a stabilized post-transition baseline (Phase 1). This separates the dominant
hardware jump from incremental tuning effects.

### 3. Failure-first design

The pipeline assumes partial failure at every stage. Scripts continue on failure, record the failure
reason in `pipeline_runs.jsonl`, and validate expected output existence and size after execution. A
successful return code does not mean the output is valid — AEME checks for stale outputs, empty
files, and truncated results.

### 4. Uncertainty-aware reporting

Outputs report effect size and uncertainty bounds, not just binary significance. Bayesian results
include HDI (Highest Density Interval) and P(>0) posterior probability. Frequentist results include
confidence intervals, rank-based tests, and robust location-shift estimates where appropriate. The
goal is decision support — "likely real" versus "promising but not yet conclusive" — not binary
accept/reject.

### 5. Reproducible experimentation

Improvement claims must survive re-analysis under multiple assumptions. Artifact bundles include
SHA256 hashes, provenance tracking, and reproducibility stamps so that a reviewer can verify that
the outputs were produced from the stated inputs. Deterministic export mode ensures identical inputs
produce identical bundles.

---

## Error Accounting

ARENA does not silently drop bad records.

When parsers encounter invalid rows or exceptions, they track: `n_total`, `n_ok`, `n_skip`, `n_err`,
reason histograms, and the first error sample.

Typical skip reasons include malformed JSON, missing keys, stale position data, and invalid filename
date parsing. This makes parsing behavior inspectable and reproducible.

---

## Model Reuse

The current public tree does not expose a shared `src/arena/lib/nb2_models.py` module.
NegativeBinomial2 model code currently lives in stage payload scripts (for example under
`scripts/adsb/analysis/phase/`, `scripts/adsb/analysis/gpu/`, and
`scripts/adsb/analysis/change_points/`).

Consistency is maintained at the pipeline/output-contract level rather than by a single shared NB2
module.

---

## Scope

AEME was developed for ADS-B receiver evaluation, but the framework generalizes to
improvement-verification problems where:

- measurements are noisy
- demand or traffic changes over time
- single-metric comparisons are misleading
- reproducibility matters more than impressions
