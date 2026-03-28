# Statistical Assumptions, Limitations, and Evidence Boundaries

> This document is not a user guide.
> It is a methodological reference describing how ARENA interprets
> statistical evidence, where each model is reliable, and where
> conclusions must be qualified.
>
> Last updated: 2026-03-08  
> Primary evaluation window: 2025-12-26 to 2026-03-08  
> (Some sections use 73 raw calendar days, 68 quality-filtered days, or 61 strictly filtered days as noted below.)
>
> All numerical results in this document are dataset-dependent and should
> be interpreted as versioned evidence summaries rather than permanent
> properties of the system.

---

## Table of Contents

1. [Overview](#1-overview)
   - 1.1 [Scope of This Document](#11-scope-of-this-document)
2. [Phase Definitions and Change Log](#2-phase-definitions-and-change-log)
3. [Model-by-Model Analysis](#3-model-by-model-analysis)
   - 3.1 [Negative Binomial GLM (Baseline)](#31-negative-binomial-glm-baseline)
   - 3.2 [Bayesian Phase Evaluation (NumPyro / NUTS)](#32-bayesian-phase-evaluation-numpyro--nuts)
   - 3.3 [Mann-Whitney U + Bootstrap CI](#33-mann-whitney-u--bootstrap-ci)
   - 3.4 [Distance-bin NB-GLM with OpenSky Offset](#34-distance-bin-nb-glm-with-opensky-offset)
   - 3.5 [Binomial GLM (Quality)](#35-binomial-glm-quality)
   - 3.6 [Change-Point Detection](#36-change-point-detection)
   - 3.7 [Time-Resolved Evaluation](#37-time-resolved-evaluation)
4. [Cross-Model Consistency](#4-cross-model-consistency)
5. [Proxy Variable Limitations](#5-proxy-variable-limitations)
6. [Evidence Summary](#6-evidence-summary)
   - 6.1 [What the Data Confirms](#61-what-the-data-confirms)
   - 6.2 [What the Data Rejects](#62-what-the-data-rejects)
   - 6.3 [What the Data Cannot Determine](#63-what-the-data-cannot-determine)
   - 6.4 [What Remains Open](#64-what-remains-open)
7. [Recommendations](#7-recommendations)
   - 7.1 [Operational Next Steps](#71-operational-next-steps)
   - 7.2 [Model / Method Improvements](#72-model--method-improvements)
   - 7.3 [Open Research Questions](#73-open-research-questions)
8. [Appendix: Decision Criteria Reference](#8-appendix-decision-criteria-reference)

---

## 1. Overview

ARENA evaluates receiver performance changes using multiple statistical
approaches in parallel. No single model is treated as authoritative.
Instead, conclusions are drawn from the **convergence or divergence**
across models.

This document is structured around a simple principle:

> Every statistical model makes assumptions.
> When assumptions hold, the model's conclusions are defensible.
> When they don't, the conclusions must be qualified or withdrawn.

The dataset covers 73 calendar days (68 after quality exclusions for the
baseline NB-GLM; 61 for the phase evaluator after stricter filtering).
Hardware changes span five phases: RTL-SDR baseline, Airspy Mini introduction,
5D-FB cable upgrade, indoor cable change (2.5DS-QFB), and adapter change
(NM-SM50+).

### 1.1 Scope of This Document

This document covers: (a) the statistical assumptions underlying each
model, (b) where those assumptions hold or fail against the observed
data, and (c) the boundaries of what can and cannot be concluded from
the current evidence.

It does **not** cover: pipeline architecture (see README), data
collection methodology (see PLAO documentation), or how to run ARENA
(see CLI usage guide).

Three dataset scopes appear throughout this document:

- **73 calendar days**: raw observation window (2025-12-26 to 2026-03-08)
- **68 days**: usable days for the baseline NB-GLM after quality exclusions
- **61 days**: usable days for the phase evaluator after stricter filtering (AUC<5000: 1 day excluded, minutes<1296: 2 days excluded)

---

## 2. Phase Definitions and Change Log

### Phase Timeline

| Phase | Name | Start Date | Hardware Change | N (days) | Reliability |
|-------|------|------------|-----------------|----------|-------------|
| 0 | Baseline (RTL-SDR) | 2025-12-26 | Vinnant 8-P antenna, RTL-SDR V4 | 8 | definitive |
| 1 | Airspy Mini + Gain 13–18 | 2026-01-14 | Airspy Mini SDR | 31 | definitive |
| 2 | 5D-FB Cable & N-P | 2026-02-14 | Cosmowave 5D-FB 2m + SMA⇔N-P adapter | 12 | definitive |
| 3 | Indoor cable (2.5DS-QFB) | 2026-02-26 | Shikoku Electric Wire 2.5DS-QFB 50cm | 3 | preliminary (low N) |
| 4 | Adapter change (NM-SM50+) | 2026-02-28 | Mini-Circuits NM-SM50+ | 7 | definitive |

### Confounding: Soft Parameter Changes Within Phases

The change log reveals extensive soft-parameter tuning within Phase 1:

- **Gain:** 13 → 15 → 18 → 19 → 20 → 19 → 18 → 19 (Jan 14–Feb 15)
- **Decoder flags:** -e (30–60), -f (1–2), -m (12–20), -w (3–4), -C 80 removed
- **Night gain schedule:** Gain auto → 21 → 19 → guard script on/off

These intra-phase changes inflate day-to-day variance within Phase 1,
which makes it harder to detect small inter-phase effects (Phases 2–4).
ARENA evaluates at the hardware-phase level and does not isolate
individual soft-parameter effects.

Additionally, Phase 2 includes a decoder parameter revert on 2026-02-27
(-m 12, -f 1, -e 60), meaning the last day of Phase 2 and all of Phase 3/4
run under different decoder settings than the rest of Phase 2.

---

## 3. Model-by-Model Analysis

### 3.1 Negative Binomial GLM (Baseline)

**General assumptions:**
The NB-GLM assumes count data (auc_n_used) follows a Negative Binomial
distribution with log link. Observations are assumed independent conditional
on covariates, and the mean-variance relationship is quadratic
(Var = μ + αμ²).

**Model specification:**
```
auc_n_used ~ post + log_traffic
Family: NegativeBinomial, Link: Log, Method: IRLS
```

**What the data showed:**

| Parameter | Estimate | 95% CI | p-value |
|-----------|----------|--------|---------|
| Intercept | 9.984 | [7.06, 12.91] | <0.001 |
| post | 0.545 | [-0.006, 1.097] | 0.053 |
| log_traffic | 0.023 | [-0.438, 0.484] | 0.921 |

Key findings:
- The post effect (+72.5%) narrowly misses significance at α=0.05 (p=0.053).
  The 95% CI for the improvement percentage is [-0.62%, +199.41%],
  spanning zero.
- The traffic elasticity is effectively zero (0.023, p=0.921).
  hnd_nrt_movements does not predict local reception counts.
- Deviance = 2.09 with 65 df, Pearson χ² = 2.42. The NB family
  adequately controls overdispersion.
- Pseudo R² = 0.049, meaning 95% of variance is unexplained.
  The model captures the phase shift, but explains little of the
  remaining day-to-day variance.

**Limitations specific to this data:**
- N=68 total, with only 8 days in the baseline. The short baseline
  inflates standard errors for the post coefficient.
- The model treats all post-change days as a single group, ignoring
  the five-phase structure. This dilutes the Airspy effect with
  potentially different cable/adapter effects.
- log_traffic (hnd_nrt_movements) proved to be a poor proxy.
  See [Section 5](#5-proxy-variable-limitations).

---

### 3.2 Bayesian Phase Evaluation (NumPyro / NUTS)

**General assumptions:**
Bayesian NB-GLM with weakly informative priors, estimated via NUTS
(No-U-Turn Sampler). The model assumes the same NB data-generating
process as Section 3.1, but estimates full posterior distributions
for phase effects. Conclusions are based on 94% Highest Density
Intervals (HDI) and P(effect > 0).

**Model specification (Phase Evaluator v3.1):**
```
auc_n_used ~ phase + traffic_control + offset(log(minutes_covered))
Family: NegBin, MCMC: warmup=1000, samples=2000, chains=12
Dual baseline: Phase 0 (RTL-SDR) and Phase 1 (Airspy Mini)
```

**What the data showed (vs Original Baseline, Phase 0):**

| Phase | N | Mean % | 94% HDI | P(>0) |
|-------|---|--------|---------|-------|
| 1: Airspy Mini | 31 | +45.0% | [+16.9, +84.0] | 99.9% |
| 2: 5D-FB Cable | 12 | +100.7% | [+17.6, +701.4] | 99.9% |
| 3: Indoor cable | 3 | +51.5% | [-10.5, +113.0] | 91.3% [prelim: low N] |
| 4: Adapter | 7 | +54.6% | [+5.8, +100.4] | 100% |

**What the data showed (vs Alt Baseline, Phase 1 = Airspy Mini):**

| Phase | N | Mean % | 94% HDI | P(>0) |
|-------|---|--------|---------|-------|
| 2: 5D-FB Cable | 12 | +31.1% | [-11.5, +335.7] | 67% |
| 3: Indoor cable | 3 | +6.1% | [-51.4, +46.2] | 68% [prelim: low N] |
| 4: Adapter | 7 | +8.0% | [-42.5, +36.0] | 79% |

**Adjacent-phase comparisons (vs_previous):**

| Comparison | Mean % | P(>0) |
|------------|--------|-------|
| Phase 1 vs Phase 0 | +45.0% | 99.9% |
| Phase 2 vs Phase 1 | +31.1% | 67% |
| Phase 3 vs Phase 2 | +0.0% | 60% |
| Phase 4 vs Phase 3 | +4.5% | 58% |

**Separate Bayesian 2-group comparison (CUDA, no traffic control):**

| Comparison | Mean % | 94% HDI | P(>0) |
|------------|--------|---------|-------|
| Airspy Mini vs RTL-SDR | +69.7% | [+44.7, +94.5] | 100% |
| Airspy+Cable vs RTL-SDR | +72.5% | [+41.7, +102.8] | 100% |
| Airspy+Cable vs Airspy Mini | +1.9% | [-14.5, +18.9] | 56% |
| airspy_cable_v2 vs RTL-SDR | +86.3% | [+31.5, +142.0] | 100% |
| airspy_cable_v2 vs Airspy+Cable | +8.5% | [-23.9, +42.3] | 66% |
| airspy_adapter vs RTL-SDR | +91.4% | [+48.3, +135.9] | 100% |
| airspy_adapter vs airspy_cable_v2 | +5.0% | [-29.6, +41.1] | 57% |

#### Why the estimated effect size changes across Bayesian models

The Phase Evaluator (with traffic + minutes control) estimates Airspy
improvement at +45.0%. The simple Bayesian 2-group model (no controls)
estimates +69.7%. This 25-point gap is not a contradiction — it shows
that controlling for operational time (minutes_covered) absorbs part of
what the uncontrolled model attributes to the hardware change. The
minutes elasticity (β=0.38, HDI [-1.77, +0.98]) is estimated but the HDI
spans zero in this dataset, meaning the offset adjustment is directionally
plausible but not individually significant.

The implication is that the estimated hardware effect is sensitive to
model specification. The controlled model is more conservative and more
interpretable for this dataset, while the uncontrolled model likely
absorbs part of the uptime variation into the hardware coefficient.

#### Phase 2 HDI anomaly

Phase 2 vs baseline shows +100.7% with an extremely wide HDI upper bound
of +701.4%. This asymmetric posterior reflects the combination of (a) a
genuine large improvement from baseline, (b) intra-phase variance driven
by the decoder parameter revert on 2026-02-27, and (c) the NB model's
log-link exponentiation amplifying right-tail uncertainty. The mean
estimate is meaningful but the upper HDI bound should not be
over-interpreted.

**Limitations specific to this data:**
- Phase 3 has N=3 days. The posterior is strongly influenced by the
  prior rather than the data. ARENA flags this as `[prelim: low N]`.
- Adjacent-phase comparisons (vs_previous) all have P(>0) < 67%,
  meaning the model cannot distinguish cable/adapter effects from noise.
- The 12-chain MCMC with 2000 samples shows good convergence for Phases
  0–2 and 4, but the Phase 3 posterior is wide due to minimal data.
- Traffic elasticity is β = -0.015 (HDI: [-0.185, +0.138]), consistent
  with the frequentist result: the traffic proxy has no predictive power.

---

### 3.3 Mann-Whitney U + Bootstrap CI

**General assumptions:**
MWU is a non-parametric rank test that assumes independent observations
and tests whether one distribution stochastically dominates another.
It makes no distributional assumptions. Bootstrap CI estimates the
sampling distribution of the mean difference via resampling.

**What the data showed (OpenSky minute-level capture ratio):**

| Comparison | N₁ | N₂ | p-value | Bootstrap Δ | 95% CI |
|------------|----|----|---------|-------------|--------|
| Airspy Mini vs Airspy+Cable | 7001 | 12674 | 9.7e-164 | -0.048 | [-0.052, -0.044] |
| Airspy Mini vs airspy_adapter | 7001 | 7029 | 1.0e-97 | -0.042 | [-0.047, -0.038] |
| Airspy Mini vs airspy_cable_v2 | 7001 | 4257 | 2.7e-46 | -0.036 | [-0.041, -0.030] |
| Airspy+Cable vs airspy_adapter | 12674 | 7029 | 5.5e-7 | +0.006 | [+0.002, +0.009] |
| Airspy+Cable vs airspy_cable_v2 | 12674 | 4257 | 3.4e-17 | +0.012 | [+0.008, +0.017] |
| airspy_adapter vs airspy_cable_v2 | 7029 | 4257 | 2.6e-5 | +0.007 | [+0.002, +0.012] |

**Limitations specific to this data:**
- **Large-sample inflation:** With n > 7000 per group, trivial differences
  (Δ ≈ 0.04 on a scale of ~1.2) produce extreme p-values (10⁻¹⁶⁴).
  Statistical significance ≠ practical significance.
- **Minute-level non-independence:** Consecutive minutes within the same
  day are autocorrelated (same weather, same aircraft tracks). MWU
  assumes independence, so p-values are likely biased downward.
- **Direction contradicts other models:** MWU shows capture ratio
  *decreasing* from Airspy Mini to Cable/Adapter phases. This conflicts
  with AUC-based models showing improvement. See
  [Section 4](#4-cross-model-consistency).

**Daily-level comparison (MWU on daily median_capture_ratio):**

| Comparison | N₁ | N₂ | p-value | Bootstrap Δ | 95% CI |
|------------|----|----|---------|-------------|--------|
| Airspy Mini vs Airspy+Cable | 5 | 9 | 0.019 | -0.047 | [-0.073, -0.019] |
| Airspy Mini vs airspy_adapter | 5 | 5 | 0.059 | -0.040 | [-0.066, -0.012] |
| Airspy+Cable vs airspy_adapter | 9 | 5 | 0.423 | +0.007 | [-0.004, +0.019] |
| Airspy+Cable vs airspy_cable_v2 | 9 | 3 | 0.194 | +0.016 | [+0.006, +0.025] |

At the daily level, only the Airspy Mini vs Airspy+Cable comparison
reaches p < 0.05. The remaining comparisons are non-significant with
very small effect sizes, consistent with the phase evaluator's finding
that post-Airspy incremental changes are not individually detectable.

---

### 3.4 Distance-bin NB-GLM with OpenSky Offset

**General assumptions:**
NB-GLM with `local_count ~ phase + offset(log(os_bin))`, estimating
phase effects on local reception at each distance band while controlling
for OpenSky traffic as an exposure variable.

**What the data showed:**

| Distance Bin | Airspy+Cable coef | p-value | airspy_adapter coef | p-value | airspy_cable_v2 coef | p-value |
|-------------|-------------------|---------|---------------------|---------|----------------------|---------|
| 0–50 km | +0.017 | 0.322 | -0.014 | 0.468 | -0.014 | 0.546 |
| 50–100 km | -0.003 | 0.856 | +0.004 | 0.831 | +0.023 | 0.290 |
| 100–150 km | -0.037 | 0.027 | -0.004 | 0.815 | +0.005 | 0.819 |
| 150–200 km | -0.067 | <0.001 | -0.082 | <0.001 | -0.092 | <0.001 |
| 200+ km | -0.094 | <0.001 | -0.133 | <0.001 | -0.152 | <0.001 |

**Key finding:** At 150+ km, phase coefficients are significantly
*negative* — later phases capture proportionally *fewer* aircraft
relative to OpenSky. This is counterintuitive given other metrics
showing improvement. The effect is monotonically stronger with
distance: the further from the station, the larger the relative
decline against the OpenSky denominator.

At 0–100 km, no phase shows significant effects (all p > 0.29),
confirming that near-field reception is structurally determined
and insensitive to the hardware changes evaluated.

**Limitations specific to this data:**
- Pseudo R² ranges from 0.00006 to 0.002 across bins. Phase explains
  almost nothing about minute-level variation, even when statistically
  significant.
- The 0–50 km bin shows capture ratios consistently < 1.0 (mean ≈ 0.56–0.58),
  indicating a structural near-field deficit unrelated to hardware changes.
- The offset assumes OpenSky's per-bin counts are proportional to true
  traffic, which may not hold at the extremes of distance.

---

### 3.5 Binomial GLM (Quality)

**General assumptions:**
Binomial GLM with logit link, modeling a success/failure ratio as a
function of traffic and post-change indicator.

**What the data showed:**

```
Intercept:    34.57  ± 3.59e+05   p ≈ 1.0
log_traffic: -8.8e-15 ± 5.18e+04  p ≈ 1.0
post:         5.2e-14 ± 3.42e+04  p ≈ 1.0
IRLS iterations: 33
Deviance: 3.83e-09
```

**This model failed completely.** The coefficients are astronomically
large with equally large standard errors, and the deviance is
effectively zero. This is a textbook case of **complete separation**:
the response variable is so extreme (nearly all successes or all failures)
that the logistic curve cannot find a finite maximum likelihood estimate.

**Conclusion:** Results from this model must not be cited or used.
The binomial specification is inappropriate for this data structure.

---

### 3.6 Change-Point Detection

**General assumptions:**
Bayesian change-point models (single and multi-point, K=3) estimate
the posterior probability of structural breaks in the time series.
They assume piecewise-constant or piecewise-linear regimes with
abrupt transitions.

**What the data showed:**
- Pipeline runs completed successfully (single: 317s, multi K=3: 1377s).
- CPU execution was forced after discovering that GPU (GTX 1060) was
  38–50× slower than CPU for small datasets — a counterintuitive but
  reproducible finding specific to JAX/NumPyro on this hardware.

#### GPU vs CPU Performance on Small Datasets

During development, all Bayesian/change-point scripts were initially
run on GPU (NVIDIA GTX 1060 6GB). Benchmarking revealed a dramatic
inversion:

| Script | GPU (GTX 1060) | CPU (i7-8700K) | Ratio |
|--------|---------------|----------------|-------|
| Bayesian phase comparison (4 chains × 3000 draws) | 658 s | 17 s | GPU 38× slower |
| Multi change-point K=3 (1 chain × 4500 draws) | 748 s | 15 s | GPU 50× slower |

The root cause is that `DiscreteHMCGibbs` (used for change-point
location sampling) performs inherently sequential discrete variable
updates. With N=59 data points, the GPU kernel launch overhead and
host-device data transfer cost dominate the computation. The GPU's
parallel execution units remain underutilized because the workload
is too small and too sequential to benefit from parallelism.

Based on these benchmarks, ARENA defaults to CPU execution for all
Bayesian and change-point scripts via `numpyro.set_platform("cpu")`
when the dataset is below a configurable threshold (currently N < 500).
This decision is recorded in the pipeline execution log.

**Limitations specific to this data:**
- The RTL-SDR → Airspy transition (Jan 10–14) involved simultaneous
  antenna + SDR + gain changes over multiple days, creating a gradual
  rather than abrupt transition. Change-point models assume sharp breaks,
  which may mislocate the transition.
- With only 61 days, K=3 change points means each segment averages ~15
  days, approaching the minimum for stable estimation.

---

### 3.7 Time-Resolved Evaluation

**General assumptions:**
Two-hour time bins are compared across old (Phase 0) and new (Phase 1+)
periods using t-tests or MWU per bin.

**What the data showed:**

| Time Bin | Old Mean | New Mean | Improvement | p-value |
|----------|----------|----------|-------------|---------|
| 00–02 | 728 | 840 | +15.4% | 0.018 |
| 02–04 | 713 | 1155 | +61.9% | 1.3e-6 |
| 04–06 | 389 | 564 | +45.0% | 8.7e-5 |
| 06–08 | 1761 | 2297 | +30.5% | 1.8e-6 |
| 08–10 | 3532 | 4584 | +29.8% | 2.8e-7 |
| 10–12 | 4785 | 6053 | +26.5% | 4.8e-6 |
| 12–14 | 4622 | 5538 | +19.8% | 0.001 |
| 14–16 | 5078 | 6405 | +26.1% | 1.3e-5 |
| 16–18 | 4758 | 6001 | +26.1% | 6.3e-6 |
| 18–20 | 4365 | 5461 | +25.1% | 3.1e-6 |
| 20–22 | 3400 | 4211 | +23.8% | 3.1e-4 |
| 22–24 | 1542 | 1703 | +10.5% | 0.054 |

11 of 12 time bins show significant improvement (p < 0.05). The 22–24
bin is borderline (p = 0.054). Night hours (02–06) show the largest
relative gains, likely because low-traffic periods amplify the hardware
sensitivity difference.

**Limitations:**
- "Old" vs "New" groups this as a binary split, collapsing all five
  phases into two groups. It cannot distinguish cable/adapter effects.
- Day-of-week confounding within time bins is not controlled.

---

## 4. Cross-Model Consistency

The critical test of any multi-model framework is whether models agree.
ARENA's models show **partial convergence with one notable contradiction**.

### Where models agree

All models agree that the RTL-SDR → Airspy transition produced a large,
real improvement:
- Phase evaluator: +45.0% (traffic + minutes controlled)
- Bayesian 2-group: +69.7% (uncontrolled)
- Time-resolved: +15–62% across all time bins
- Coverage P95: 188–191 km → 201–209 km
- LOS efficiency: 56.5% → 60.3%

The magnitude differs by model specification (controlled vs uncontrolled),
but the direction and significance are unanimous.

All models also agree that individual cable/adapter effects are not
individually distinguishable from noise at this sample size:
- Phase evaluator vs alt baseline: P(>0) = 67%, 68%, 79%
- Bayesian CUDA adjacent comparisons: P(>0) = 56–66%
- Daily MWU for post-Airspy phases: p > 0.05 except one borderline case

### Where models disagree

**OpenSky capture ratio decreases in later phases**, while AUC, coverage,
and LOS metrics increase. Specifically:

| Metric | Airspy Mini | Airspy+Cable | airspy_cable_v2 | Adapter |
|--------|-------------|--------------|-----------------|---------|
| Capture ratio (overall) | 1.210 | 1.162 | 1.175 | 1.168 |
| AUC mean (daily) | ~42,400 | ~43,100 | — | ~46,600 |
| P95 distance (km) | ~190 | ~195 | ~205 | ~208 |
| LOS efficiency (%) | ~57.8 | ~58.5 | — | ~60.0 |

Three independent metrics (AUC, P95, LOS) show improvement while the
capture ratio shows decline. The distance-bin NB-GLM (Section 3.4)
provides additional specificity: the decline is concentrated at 150+ km,
where later phases show significantly negative coefficients (p < 0.001),
while 0–100 km shows no significant phase effects.

This suggests the capture ratio is not a
reliable improvement indicator in this context. Possible explanations:

1. **Near-field bias:** The 0–50 km capture ratio is consistently < 1.0
   (mean ≈ 0.56–0.58 across all phases), pulling the overall ratio down.
   Hardware changes that improve far-field reception may not
   proportionally improve near-field.
2. **OpenSky coverage changes:** If OpenSky's own coverage improved
   during the study period, the denominator grows faster than local
   improvement, deflating the ratio. The monotonic decline of capture
   ratio with increasing distance (150–200 km: -0.067 to -0.092;
   200+ km: -0.094 to -0.152) is consistent with this hypothesis.
3. **Schema migration:** The transition to PLAO pos schema_ver=1 may
   have changed how local unique aircraft are counted.
4. **Distance-dependent effect direction:** NB-GLM shows non-significant
   positive coefficients at 50–100 km but increasingly negative at
   150+ km, suggesting the effect varies with distance in complex ways
   that a single overall capture ratio cannot represent.

**Working interpretation:** The capture ratio should be treated as a
supplementary diagnostic, not a primary improvement metric, until the
near-field bias mechanism and the far-field decline are better understood.

---

## 5. Proxy Variable Limitations

### Traffic Proxy: hnd_nrt_movements

ARENA uses Haneda/Narita airport movement counts as a proxy for overhead
traffic. This proxy has two documented failures:

**Failure 1: Zero predictive power.**
In both the frequentist NB-GLM (β=0.023, p=0.921) and the Bayesian
model (β=-0.015, HDI [-0.185, +0.138]), the traffic variable explains
no variance in local AUC. Airport departures/arrivals do not correlate
with the number of aircraft flying over a ground station located ~60 km
from the airports.

**Failure 2: Missing data for key periods.**
Several days show anomalous traffic values: 2026-01-02 (111), 2026-01-24
(167), 2026-03-05 (467). These appear to be data fetch failures rather
than true traffic dips, introducing noise. Traffic data is entirely
missing for 2026-03-06 and 2026-03-07.

**Implication:** The traffic covariate does not confound the phase
estimates (because it has no effect), but it also does not add any
explanatory power. Claims of "traffic-controlled analysis" remain
technically correct, but in practice the control adds little
explanatory power in this dataset.

### Local Traffic Proxy (unique_hex_50km)

Available only from 2026-01-24 onward (no data for RTL-SDR Phase 0
or early Phase 1). This limits any normalized comparison that requires
a consistent denominator across all phases.

### OpenSky n_used as Offset

Using `offset(log(os_n_used))` in the NB-GLM assumes OpenSky counts
are a good measure of true overhead traffic per minute. This assumption
is weaker at distance extremes (0–50 km where terrain/altitude limits
OpenSky coverage, and 200+ km where both systems approach their range
limits).

---

## 6. Evidence Summary

### 6.1 What the Data Confirms

**The Airspy Mini introduction was a decisive improvement.**
Phase evaluator: +45.0% [HDI +16.9, +84.0], P(>0)=99.9%.
Bayesian CUDA: +69.7% [HDI +44.7, +94.5], P(>0)=100%.
The difference between estimates (45.0% vs 69.7%) reflects the impact
of controlling for operational minutes, not a contradiction.
Every model — frequentist, Bayesian, time-resolved, coverage, LOS — agrees.

**Improvement spans all hours of the day.**
11 of 12 two-hour bins show p < 0.05, with night hours (02–06) showing
the largest relative gains (+45–62%). This rules out the hypothesis
that improvement is an artifact of time-of-day sampling bias.

**Coverage area expanded measurably.**
Average P95 distance grew from ~188–191 km (Phase 1 early) to ~201–209 km
(Phase 4). Area_p95 increased correspondingly from ~117,000 km² to
~142,000 km².

**LOS efficiency improved progressively.**
From 56.5% (Phase 0) to 60.3% (Phase 4), with a visible step at the
cable change (Phase 2). This is an independent physical metric
consistent with the AUC-based evaluation.

**Cumulative improvement from RTL-SDR baseline is confirmed for all
post-Airspy phases.**
vs Phase 0: Phase 2 +100.7% (P=99.9%), Phase 4 +54.6% (P=100%).
The cumulative hardware stack (Airspy + cable + adapter) produces a
clear and statistically significant improvement over the original system.

### 6.2 What the Data Rejects

**"Traffic increased, not receiver performance."**
The traffic proxy has zero explanatory power (p=0.921 in frequentist,
HDI spanning zero in Bayesian). Including or excluding it does not
change the phase estimates. While the proxy itself may be flawed,
the phase effect persists regardless of traffic specification.

**"Cable and adapter changes each produced individually significant
improvement."**
Against the Airspy Mini alt-baseline: 5D-FB +31.1% (P=67%),
Indoor cable +6.1% (P=68%), Adapter +8.0% (P=79%). All 94% HDIs include zero.
Adjacent-phase comparisons (vs_previous) are all below P=67%.

**"The capture ratio validates the improvement."**
OpenSky capture ratio *decreases* from 1.21 to 1.16–1.17, contradicting
AUC, coverage, and LOS metrics. The distance-bin NB-GLM further shows
that this decline is concentrated at 150+ km (p < 0.001) while near-field
(0–100 km) is unaffected. The capture ratio is not a reliable
improvement indicator in this dataset.

**"The binomial quality model supports the conclusions."**
The binomial GLM suffered complete separation (coefficients ±10⁵, p≈1.0).
Its results are invalid and must not be cited.

### 6.3 What the Data Cannot Determine

**Individual cable/adapter effects.**
The five phases were applied sequentially with no reversal.
A/B testing (temporarily reverting to the old cable) would be needed
to isolate individual effects, but is impractical.

**Indoor cable change (2.5DS-QFB) impact.**
N=3 days. ARENA's own reliability tag marks this as `[prelim: low N]`.
The +6.1% estimate (P=68%) is a directional hint, not evidence.

**Soft parameter contributions.**
Gain (13–21), decoder flags (-e, -f, -m, -w), and night-time schedules
were changed multiple times within Phase 1. ARENA evaluates at the
hardware-phase level and cannot attribute effects to individual parameters.

**The cause of near-field capture deficit.**
The 0–50 km capture ratio is consistently ~0.56–0.58 (local sees ~57% of
what OpenSky sees at close range) across all phases. The distance-bin
NB-GLM confirms no phase has a significant effect at 0–50 km (all p > 0.32).
Whether this is antenna directivity, readsb filtering, terrain masking,
or low-altitude aircraft exclusion is undetermined.

### 6.4 What Remains Open

**Long-term stability and seasonal effects.**
The 61-day window (late December to early March) covers only winter
conditions. Atmospheric propagation, humidity, and temperature inversions
may affect ADS-B reception differently in summer.

**Achieving P(>0) ≥ 95% for the adapter change.**
Currently at P=79% (vs Airspy Mini baseline) with N=7 days. As data
accumulates, this may cross the 80% or 95% thresholds, but the rate
of convergence depends on day-to-day variance (CV ≈ 15–20%)
and is unreliable to extrapolate. A single low-value day (e.g., the
Feb 25 outlier with AUC=33,229) can significantly delay convergence.

**Reconciling the capture ratio contradiction.**
Why does the OpenSky-relative metric decline while absolute and
coverage-based metrics improve? The distance-bin analysis narrows the
problem: the decline is at 150+ km. Until the mechanism (OpenSky
denominator growth, atmospheric effects, or schema differences) is
identified, the capture ratio cannot be used as a primary evaluation metric.

**Phase 0 normalization quality.**
RTL-SDR period data lacks local_traffic_proxy, pos_records, and
unique_hex counts. Comparisons involving Phase 0 rely solely on
auc_n_used and hnd_nrt_movements (which has zero predictive power).

---

## 7. Recommendations

### 7.1 Operational Next Steps

1. **Accumulate data for Phase 4.** The adapter change shows P(>0)=79%
   with 7 days. Continue monitoring without further changes until the
   estimate stabilizes (target: 21+ days to allow for weekly cycles).

2. **Stabilize the traffic data pipeline.** Address the fetch failures
   producing anomalous values (e.g., 2026-01-02: 111, 2026-03-05: 467)
   and fill the missing days (2026-03-06, 2026-03-07).

### 7.2 Model / Method Improvements

3. **Improve the traffic proxy.** Consider using OpenSky's per-minute
   n_used directly (already available for 22 usable days) rather than
   airport-level movement counts. Alternatively, use unique_hex_50km
   from PLAO data once the coverage gap for Phase 0 is addressed.

4. **Retire the binomial quality model.** Replace with a beta-regression
   or quasi-binomial approach that can handle extreme success rates
   without separation.

5. **Analyze soft-parameter sensitivity.** Use the change log timestamps
   to create sub-phases within Phase 1 (e.g., gain 13–15 vs 17–19,
   -e 30 vs -e 60) and test whether any soft parameter produced
   detectable effects.

### 7.3 Open Research Questions

6. **Investigate the capture ratio anomaly.** Decompose the 0–50 km
   deficit by analyzing aircraft altitude distributions and comparing
   local vs OpenSky detection by altitude band. Also investigate whether
   the monotonic far-field decline (150–200 km, 200+ km) reflects
   OpenSky denominator growth over the study period.

7. **Add seasonal monitoring.** Extend the dataset through at least one
   full season change (June–August) before drawing long-term conclusions.

8. **Characterize the GPU/CPU crossover point.** The current threshold
   (datasets below N < 500 default to CPU execution) is based on
   empirical benchmarking on small datasets, including the N=59
   development case. As data accumulates, re-benchmark to find the
   dataset size where GPU parallelism begins to outperform CPU
   sequential execution for DiscreteHMCGibbs and NUTS on this hardware.

---

## 8. Appendix: Decision Criteria Reference

These thresholds are ARENA's operational interpretation rules,
not universal statistical standards. They were chosen to balance
sensitivity with the practical constraint of small sample sizes
in a single-station observation study.

For phases with `definitive` reliability (N ≥ 7):

| Criterion | Interpretation |
|-----------|---------------|
| P(>0) ≥ 95% | Improvement is very likely |
| P(>0) ≥ 80% | Improvement likely |
| P(>0) < 80% | Effect unclear (more data needed) |
| 94% HDI excludes 0 | Statistically significant |

For phases with `[reference: N<3]`: no statistical judgment is made.
For phases with `[prelim: low N]` (3 ≤ N < 7): trend indication only.