# ARENA — Aerial Radio Evaluation & Network Analytics

[![CI](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-monitored-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Lint](https://img.shields.io/badge/lint-ruff-ccff99)
![Docker Smoke](https://img.shields.io/badge/docker-smoke%20tested-2496ED)
![Artifacts](https://img.shields.io/badge/artifacts-verify%20%2F%20replay-orange)

---

## Overview

ADS-B (Automatic Dependent Surveillance–Broadcast) receivers collect aircraft position broadcasts transmitted by aircraft transponders.
However, before/after charts often lie.

An ADS-B receiver upgrade may increase aircraft counts — but that does not necessarily mean the receiver actually improved.
Traffic volume, time of day, and operational patterns can easily produce the same effect.

ARENA also exports evaluation outputs as reproducible artifact bundles
that can be independently verified and replayed for auditability.

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
- Modular 8-stage pipeline with failure-resilient JSONL logging and output validation
- Artifact bundle export, verification, and replay for auditability
- Deterministic artifact bundles with manifest, provenance, and integrity metadata
- Docker smoke validation with public sample data and opt-in real-data checks
- CI-validated contracts for CLI, pipeline, and artifact compatibility

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
├── src/arena/           # Core library (stats, pipeline, artifacts)
│   ├── artifacts/       # Public artifact substrate
│   ├── pipeline/        # Modular pipeline orchestration
│   └── lib/             # Shared config, path resolution, stats utilities
├── scripts/             # Pipeline scripts + compatibility tools
│   └── dev/             # Smoke validation helpers
├── tests/               # pytest test suite
├── data/sample/         # Public sample dataset
├── output/sample/       # Example output artifacts
├── docker/              # Container configuration
├── docs/
│   ├── architecture.md                        # Pipeline stages and package structure
│   ├── aeme.md                                # Statistical engine design and philosophy
│   ├── statistical-assumptions-and-limitations.md
│   ├── failure-taxonomy.md                    # Failure classification and error codes
│   ├── sample_outputs.md                      # Output artifact descriptions
│   ├── real-data-smoke.md                     # Opt-in real-data validation
│   └── adr/ADR-artifact-subsystem.md          # ADR for artifact subsystem
├── .github/workflows/   # CI configuration (GitHub Actions)
└── pyproject.toml       # Package metadata and dependencies
```

---

## Public Repository Scope

This public repository is designed to let readers:

- Inspect the evaluation architecture
- Validate the local environment
- Run the test suite
- Run Docker/sample smoke validation
- Verify and replay exported artifact bundles
- Explore sample input and output artifacts

It is **not** intended to reproduce a full real-world receiver experiment from the included sample dataset alone.
Full evaluation requires telemetry generated by PLAO and adsb-eval over a meaningful observation period.


---

## Documentation

- [docs/architecture.md](./docs/architecture.md) — system architecture, pipeline stages, and package structure
- [docs/aeme.md](./docs/aeme.md) — AEME statistical evaluation framework and design philosophy
- [docs/statistical-assumptions-and-limitations.md](./docs/statistical-assumptions-and-limitations.md) — model assumptions, evidence boundaries, and interpretation limits
- [docs/failure-taxonomy.md](./docs/failure-taxonomy.md) — pipeline and artifact failure classification with error codes and recovery actions
- [docs/sample_outputs.md](./docs/sample_outputs.md) — explanation of all sample output artifacts
- [docs/real-data-smoke.md](./docs/real-data-smoke.md) — optional validation using local telemetry datasets
-  [docs/artifact-bundle-spec.md](./docs/artifact-bundle-spec.md) — artifact bundle structure, verify/replay contract, and trust model
- [docs/adr/ADR-artifact-subsystem.md](./docs/adr/ADR-artifact-subsystem.md) — design decision for the artifact subsystem

---

## Artifact Workflow

ARENA exports evaluation results as artifact bundles that can be archived, verified, and replayed
independently from the original pipeline run.

```bash
arena run                              # produce evaluation outputs
arena artifacts verify <bundle>        # check hashes, schema, provenance
arena artifacts replay <bundle>        # revalidation and audit replay
```

Each bundle contains manifest records, provenance metadata, SHA256 integrity hashes,
run metadata, and the evaluation outputs themselves.
Deterministic export mode ensures identical inputs produce identical bundles.

See [docs/sample_outputs.md](./docs/sample_outputs.md) for artifact structures, and
[docs/artifact-bundle-spec.md](./docs/artifact-bundle-spec.md) for bundle format
and verification guarantees, and
[docs/adr/ADR-artifact-subsystem.md](./docs/adr/ADR-artifact-subsystem.md)
for the design decision.

---

## Quick Start

The steps below run ARENA using the included public sample dataset.
No external telemetry sources are required.

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
pytest -q
coverage run -m pytest
coverage report
```

Coverage is enforced in CI for contract-critical paths including:

- `src/arena/pipeline/`
- `src/arena/artifacts/`
- `src/arena/cli.py`
- `scripts/tools/merge_output_for_ai/merge_output_for_ai.py`

## Docker Smoke

Run the public sample smoke validation in Docker:

```bash
docker build -f docker/Dockerfile.cpu -t arena-release-smoke .
docker run --rm arena-release-smoke python scripts/dev/run_sample_smoke.py
```

or with Compose:

```bash
docker compose -f docker/docker-compose.yml run --rm arena-sample-smoke
```

The CPU image is provisioned for the full `arena run` CLI, including the
Bayesian and NumPyro/JAX dependencies required by Stage 4 and Stage 5.
This keeps the smoke and full-run workflows on the same image, at the cost of
a larger Docker build than a smoke-only image.

## Docker Full Run

Run the full pipeline in Docker with the repository mounted into `/workspace`
so outputs persist on the host:

```powershell
docker build -f docker/Dockerfile.cpu -t arena-release .
docker run --rm -it `
  -v "E:\arena_release:/workspace" `
  -w /workspace `
  arena-release `
  arena run --backend native --data-dir /workspace/data --output-dir /workspace/output --stage 1 --no-gpu
```

Compose equivalent:

```bash
docker compose -f docker/docker-compose.yml run --rm arena-run
```

You can override the default `arena run` arguments when needed. For example:

```bash
docker compose -f docker/docker-compose.yml run --rm arena-run arena run --only 1 --dry-run --skip-plao
docker compose -f docker/docker-compose.yml run --rm arena-run arena validate --data-dir /workspace/data --output-dir /workspace/output --create-dirs
```

## Real-Data Smoke

Real-data validation is local and opt-in only.
Point the public layer at a mounted or local private dataset through environment variables.

PowerShell example:

```powershell
$env:ARENA_REAL_DATA_ROOT="E:\arena\data"
$env:ARENA_REAL_ARTIFACT_BASE="E:\arena\output"
python scripts/dev/run_real_data_smoke.py
```

Docker example:

```bash
export ARENA_DATA_ROOT=/host/path/to/private/data
export ARENA_ARTIFACT_BASE=/host/path/to/private/output
docker compose -f docker/docker-compose.yml run --rm arena-real-data-smoke
```

More detail: [docs/real-data-smoke.md](./docs/real-data-smoke.md)

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

Public-release note:

- The public sample layer keeps schema-relevant telemetry fields, but station-identifying metadata such as exact receiver coordinates must be generalized.
- City-level or region-level context is acceptable; exact private station location and credentials are not.
- Generated `output/performance/` artifacts are local runtime outputs and are not versioned in the public layer.

---

## Example Output

Example output artifacts are provided in `output/sample/`.
For descriptions of each file, see [docs/sample_outputs.md](./docs/sample_outputs.md).

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

| Metric | Value |
|---|---|
| Lint | Ruff |
| Tests | pytest (unit + integration + compatibility) |
| Coverage | `coverage run -m pytest` |
| Docker | sample smoke on CI |
| Python | 3.11+ |

---

## License

[MIT License](./LICENSE)

---

## Author

Yuki Murata\
Building systems that measure reality, not impressions.
