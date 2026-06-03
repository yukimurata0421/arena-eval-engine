[![ci](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/ci.yml?branch=main&label=ci)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
[![verify-smoke](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/verify-smoke.yml?branch=main&label=verify-smoke)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/verify-smoke.yml)
[![docker-smoke](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/docker-smoke.yml?branch=main&label=docker-smoke)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/docker-smoke.yml)
[![coverage-threshold](https://img.shields.io/badge/coverage-%E2%89%A586%25-brightgreen)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
[![version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/yukimurata0421/arena-eval-engine/blob/main/CHANGELOG.md)

# ARENA — ADS-B Receiver Evaluation Engine
 
ARENA is a statistical evaluation engine that determines whether ADS-B receiver hardware changes actually improved performance — or whether observed differences are just noise.
 
The core difficulty is that observed improvements are easily confounded: traffic varies by time-of-day and season, observation conditions are non-stationary, and metrics become unstable under sparse or bursty data. Simple before/after comparisons cannot separate real gains from these confounders. ARENA is designed to make that separation explicit through multiple complementary statistical methods (Bayesian NB-GLM with NumPyro/NUTS, frequentist NB-GLM, Mann-Whitney U, change-point detection, OpenSky-normalized capture ratios, distance-band analysis) and draws conclusions from convergence or divergence across models — no single method is treated as authoritative.
 
## System Overview
 
```
Raspberry Pi (edge)                WSL2 / Linux (analysis)
┌──────────────────┐               ┌────────────────────────────┐
│  readsb → PLAO   │  rsync/pull   │                            │
│       → adsb-eval│──────────────>│  pipeline (9 stages,       │
└──────────────────┘               │           wave-parallel)   │
                                   │      │                     │
                                   │      ├──> /output          │
                                   │      │    (PNGs, reports)  │
                                   │      │                     │
                                   │  artifact run              │
                                   │      │                     │
                                   │      └──> /output/payload  │
                                   │           (CSVs, bundles)  │
                                   └────────────┬───────────────┘
                                                │
                                   CSVs (~55 files) + prompt
                                                │
                                                ▼
                                   ┌────────────────────────────┐
                                   │  Multiple LLMs             │
                                   │  → structured JSON claims  │
                                   │  → raw/{ai_name}/          │
                                   │    YYYYMMDD.json           │
                                   └────────────┬───────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────────┐
                                   │  synthesis                 │
                                   │  ingest → triage →         │
                                   │  proposition review        │
                                   │  (SQLite, two-layer DB)    │
                                   └────────────────────────────┘
```
 
Three subsystems:
 
- **Pipeline** — 9-stage orchestration with wave-parallel scheduling, failure-resilient execution, append-only audit logging, and a final evidence synthesis stage. Outputs human-readable graphs, reports, and evidence matrices to `/output`.
- **Artifacts** — Converts pipeline outputs into verifiable, LLM-ready evidence bundles. Content identity (SHA-256), schema validation, and provenance/lineage ensure that downstream analysis operates on auditable evidence, not implicit assumptions. Integrity verification carries through to synthesis ingestion.
- **Synthesis** — Cross-model claim ingestion from multiple LLMs, enrichment, baseline clustering, proposition mapping, automated triage, and human review queue. Two-layer DB design (proposition + claim layers with convergence judgments). SQLite-backed, path-isolated.
 
## Design Philosophy
 
ARENA treats LLMs as hypothesis generators, not truth sources — claims are validated through structured evidence and cross-model convergence. It does not average model families into a weighted ensemble score; support, counter-evidence, caveats, and validation targets remain separate so disagreement stays reviewable.

The full catalogue of 33 engineering decisions is in [docs/principles/engineering-decisions.md](docs/principles/engineering-decisions.md).
 
## Quick Start
 
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .[dev]
python -m arena.cli validate
pytest
```
 
## Key Commands
 
```bash
arena validate                                    # check settings/paths
arena run --only 1 --dry-run --no-gpu --skip-plao # pipeline dry run
arena artifacts verify <bundle>                   # verify artifact bundle
arena synthesis run --path sample_data/synthesis/raw --db ./tmp/s.db \
  --enriched-dir ./tmp/enriched --review-dir ./tmp/review \
  --raw-original-dir ./tmp/orig --raw-repaired-dir ./tmp/repaired \
  --repair-log-dir ./tmp/logs                     # synthesis smoke run
```
 
## Docker
 
```bash
docker compose -f docker/docker-compose.yml run --rm arena-tests
docker compose -f docker/docker-compose.yml run --rm arena-validate
docker compose -f docker/docker-compose.yml run --rm arena-synthesis-smoke
```
 
## Documentation
 
Detailed docs live in `docs/`. Start at [docs/README.md](docs/README.md).
 
| Category | Key Documents |
|---|---|
| Operations | [Architecture](docs/facts/architecture.md) · [Reproducibility](docs/facts/reproducibility.md) · [Synthesis](docs/facts/synthesis.md) · [MCMC / GPU Policy](docs/facts/performance/mcmc-parallelism-and-gpu-policy.md) |
| Design | [Engineering Decisions](docs/principles/engineering-decisions.md) · [Why Not Weighted Ensemble](docs/principles/why-not-weighted-ensemble.md) · [Evidence Synthesis Stage](docs/principles/evidence-synthesis-stage.md) · [AEME](docs/principles/aeme.md) |
| Context | [System Context](docs/facts/system-context.md) · [Statistical Assumptions](docs/principles/statistical-assumptions-and-limitations.md) · [Model Disagreement Report Example](docs/facts/examples/model_disagreement_report_example.md) |
 
## Tech Stack
 
Python 3.11+, NumPyro/JAX, PyMC, statsmodels, pandas, SQLite, Docker, GitHub Actions CI (lint + test matrix + coverage + smoke + Docker).
 
## Notes
 
- Secrets, credentials, and private data are excluded by design.
- Use environment variables or CLI overrides for local paths.
- Sample data under `sample_data/` is synthetic and deterministic.
