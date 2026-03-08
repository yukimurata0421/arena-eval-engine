# ARENA — Aerial Radio Evaluation & Network Analytics

[![CI](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-43%25-yellow)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## Overview

ADS-B (Automatic Dependent Surveillance–Broadcast) receivers collect aircraft position broadcasts transmitted by aircraft transponders.
However, before/after charts often lie.

An ADS-B receiver upgrade may increase aircraft counts — but that does not necessarily mean the receiver actually improved.
Traffic volume, time of day, and operational patterns can easily produce the same effect.

**ARENA exists to answer a single question:**

> Is a claimed receiver improvement statistically defensible?

ARENA is a statistical evaluation engine for noisy ADS-B telemetry.
It determines whether observed receiver improvements are real or simply artifacts
of traffic variation and external conditions.

ARENA does **not** collect telemetry and does **not** decode ADS-B messages.
It evaluates telemetry produced by upstream systems.

---

## What ARENA Outputs

ARENA does not decode ADS-B signals and does not operate as a live receiver UI.
It consumes telemetry produced by upstream systems and generates evaluation artifacts such as:

- Coverage-based AUC summaries
- Structured statistical reports
- Traffic-normalized performance comparisons
- Phase-level Bayesian and frequentist evaluation outputs

In practical terms, ARENA answers questions like:

- Did a receiver change produce a real improvement?
- Was the observed gain larger than normal traffic variation?
- Does the evidence support the claim, or is more data needed?

---

## Why ARENA

For a detailed discussion of model assumptions, failed specifications,
proxy limitations, and interpretation boundaries, see
[docs/statistical-assumptions-and-limitations.md](./docs/statistical-assumptions-and-limitations.md).

Most ADS-B receiver improvements are reported using simple before/after aircraft counts.
However, these metrics are heavily confounded by traffic volume, seasonal patterns, and operational variation.

ARENA was created to provide a statistically defensible evaluation method for receiver performance experiments.

---

## Architecture

ARENA is the final layer in a telemetry stack that starts at the receiver.

```
                    Raspberry Pi

                 readsb runtime
                    ├─ /run/readsb/aircraft.json
                    └─ /run/readsb/stats.json


        ┌───────────────┐        ┌───────────────┐
        │      PLAO     │        │   adsb-eval   │
        │ position log  │        │ runtime stats │
        └───────┬───────┘        └───────┬───────┘
                │                        │
                │ pos_YYYYMMDD.jsonl     │ dist_1m.jsonl
                │                        │
                └──────────────┬─────────┘
                               │
                               ▼

                          ARENA (WSL2)
               statistical evaluation engine
```

[PLAO](https://github.com/yukimurata0421/plao-pos-collector) and [adsb-eval](https://github.com/yukimurata0421/adsb-eval) run on the Raspberry Pi and produce telemetry logs independently.
ARENA pulls those logs and performs all statistical evaluation on a separate machine.
This layered design isolates telemetry collection from statistical evaluation,
allowing receiver experiments to be analyzed independently from the data acquisition layer.

For the internal pipeline architecture (stages 1–8), see [docs/architecture.md](./docs/architecture.md).

---

## Input Contract

ARENA expects telemetry produced by PLAO and adsb-eval.

| File | Source | Description |
|---|---|---|
| `pos_YYYYMMDD.jsonl` | PLAO | Per-aircraft position logs with lat/lon, altitude, distance, and bearing |
| `dist_1m.jsonl` | adsb-eval | Minute-level aggregated distance and signal statistics |
| `data/flight_data/airport_movements.csv` | OpenSky Network | Nearby airport traffic counts used as a confounding control |

`pos_YYYYMMDD.jsonl` and `dist_1m.jsonl` are required.
`airport_movements.csv` is optional but recommended for traffic normalization.

See `data/sample/` for example file structures.

---

## Key Features

- Statistical validation of receiver performance claims
- Coverage-based AUC performance metrics (not peak range)
- Bayesian (NumPyro MCMC) and frequentist (NB-GLM, Mann-Whitney U) evaluation
- Dual-baseline system: original RTL-SDR baseline and stabilized Airspy baseline
- OpenSky Network integration for external traffic normalization
- Bootstrap confidence intervals and distribution comparison
- 8-stage reproducible pipeline with failure-resilient JSONL logging
- CI-validated statistical core functions

---

## Example Result

The excerpt below shows how ARENA evaluates a receiver change against two baselines.

### Bayesian Phase Evaluation (Dual Baseline)

```
vs Original Baseline (RTL-SDR)          vs Alt Baseline (Airspy Mini)
-----------------------------------------  -----------------------------------------
Airspy Mini     +41.6% P(>0)=100%         5D-FB Cable      +3.4% P(>0)=64%
5D-FB Cable     +46.0% P(>0)=100%         Adapter change  +12.0% P(>0)=83%
Adapter change  +58.3% P(>0)=100%
```

**Interpretation:**
The RTL-SDR → Airspy Mini transition shows a dominant improvement,
with a posterior probability that strongly supports a real gain over the original baseline.
Post-Airspy fine-tuning (cable and adapter changes) remains directionally positive,
but the evidence is not yet strong enough to treat those gains as statistically settled —
P(>0) has not reached the 95% threshold.

**Decision support:**
ARENA separates major hardware changes from incremental tuning effects.
This allows the operator to distinguish "likely real improvement" from
"promising but not yet conclusive" — and to decide whether more observation
is needed before committing to a hardware configuration.

---

## Project Structure

```
arena-eval-engine/
├── src/arena/           # Core library (statistical functions, metrics, evaluation)
├── scripts/             # Pipeline scripts (stages 1–8)
├── tests/               # pytest test suite
├── data/sample/         # Public sample dataset
├── output/sample/       # Example output artifacts
├── docker/              # Container configuration
├── docs/
│   ├── architecture.md          # Pipeline stage diagram (Mermaid)
│   ├── statistical-assumptions-and-limitations.md
│   └── aeme.md                  # Statistical engine design and philosophy
├── .github/workflows/   # CI configuration (GitHub Actions)
└── pyproject.toml       # Package metadata and dependencies
```

---

## What You Can Run from This Public Repository

This public repository is designed to let readers:

- Inspect the evaluation architecture
- Validate the local environment
- Run the test suite
- Explore sample input and output artifacts

It is **not** intended to reproduce a full real-world receiver experiment from the included sample dataset alone.
Full evaluation requires telemetry generated by PLAO and adsb-eval over a meaningful observation period.

---

## Quick Start

ARENA can be executed locally with the included public sample dataset.

### 1. Clone repository

```bash
git clone https://github.com/yukimurata0421/arena-eval-engine.git
cd arena-eval-engine
```

### 2. Install dependencies

Recommended (development install):

```bash
pip install -e ".[dev]"
```

Minimal install:

```bash
pip install -e .
```

### 3. Validate environment

```bash
arena validate
```

Expected output:

```
Configuration OK
Environment validated
```

### 4. Run tests

```bash
pytest -q --cov=arena --cov-report=term-missing
```

```
15 passed
Coverage: 43%
```

Coverage focuses on statistical functions, telemetry parsing, and evaluation logic.
Pipeline scripts and CLI wrappers are intentionally less covered.

---

## Sample Data

A public example dataset is included in `data/sample/`.

| File | Description |
|---|---|
| `sample_dist_1m.jsonl` | Minute-level distance telemetry |
| `sample_plao_pos_pos_YYYYMMDD.jsonl` | Raw aircraft position logs |
| `sample_decoder_metrics.jsonl` | Decoder performance telemetry |
| `sample_signal_stats.jsonl` | SDR signal statistics |
| `sample_flight_data_airport_movements.csv` | Traffic proxy data |

### What the sample dataset provides

- Understanding ARENA input contracts and telemetry structure
- Verifying test reproducibility
- Exploring expected output artifacts

### What the sample dataset does NOT provide

The included dataset is intentionally minimal and **not intended for full statistical evaluation**.
It does not represent long-term receiver performance, full traffic normalization, or hardware improvement validation.
Full evaluation requires real telemetry generated by PLAO and adsb-eval.

---

## Example Output

Example output artifacts are provided in `output/sample/`:

| File | Description |
|---|---|
| `auc_summary.csv` | Coverage AUC evaluation results |
| `report.json` | Structured evaluation report |
| `metrics_summary.csv` | Aggregated performance metrics |

---

## Statistical Philosophy

ARENA emphasizes statistical defensibility over simple before/after comparison.

- Maximum distance alone is not a reliable metric
- Receiver performance must be evaluated as a distribution, not a single number
- Coverage AUC captures receiver performance across distance distributions rather than relying on maximum range
- Traffic volume is a confounding factor that must be controlled
- A dual-baseline system separates the dominant hardware jump from incremental improvements

The statistical engine design (AEME) is documented in
[docs/aeme.md](./docs/aeme.md).

For model assumptions, failed specifications, proxy limitations,
and interpretation boundaries, see
[docs/statistical-assumptions-and-limitations.md](./docs/statistical-assumptions-and-limitations.md).

---

## CI / Testing

ARENA uses GitHub Actions for continuous integration.

```bash
pytest -q --cov=arena --cov-report=term-missing
```

| Metric | Value |
|---|---|
| Tests | 15 passed |
| Coverage | 43% |
| Python | 3.11+ |

---

## License

[MIT License](./LICENSE)

---

## Author

Yuki Murata\
Building systems that measure reality, not impressions.