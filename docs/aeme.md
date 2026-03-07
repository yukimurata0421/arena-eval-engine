# AEME — Aerial Evaluation & Measurement Engine

AEME is the analytical core of ARENA.

Its role is to evaluate whether observed changes are:

- real
- quantitatively meaningful
- robust across methods
- reproducible under uncertainty

AEME is designed to validate improvement claims using multiple complementary approaches rather than a single metric or a single model.

---

## Statistical Methods

AEME integrates methods such as:

- Mann–Whitney U test
- Bootstrap confidence intervals
- Negative Binomial GLM
- Bayesian posterior inference using NumPyro / NUTS
- Change-point detection
- Proxy bias / endogeneity diagnostics
- External dataset validation using OpenSky Network

Outputs prioritize:

- effect size
- confidence / credible intervals
- uncertainty bounds
- robustness across models
- reproducibility under re-analysis

---

## Core Philosophy

AEME is designed around four principles.

### 1. Failure-first design

Pipelines assume partial failure.
Validation precedes execution wherever possible.

### 2. Configuration as source of truth

Runtime behavior is controlled through configuration such as `settings.toml`.

### 3. Uncertainty-aware evaluation

Outputs emphasize magnitude and uncertainty, not just binary significance.

### 4. Reproducible experimentation

Improvement claims must survive re-analysis under multiple statistical assumptions.

---

## Error Accounting and Skips

ARENA does not silently drop bad records.

When parsers encounter invalid rows or exceptions:

- counts are tracked (`n_total`, `n_ok`, `n_skip`, `n_err`)
- reason histograms are logged
- the first error sample is retained for debugging

Typical skip reasons include:

- malformed JSON
- missing keys
- stale position data
- invalid filename date parsing

This design makes parsing behavior inspectable and reproducible.

---

## Model Reuse

Change-point and GPU-oriented scripts share core NegativeBinomial2 model definitions through:

```text
src/arena/lib/nb2_models.py
```

This keeps statistical assumptions consistent while preserving script-level entry points.

---

## Scope

AEME was originally developed for ADS-B receiver evaluation, but the framework generalizes to broader improvement-verification problems where:

- measurements are noisy
- traffic or demand changes over time
- single-metric comparisons are misleading
- reproducibility matters more than impressions