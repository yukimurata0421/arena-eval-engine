[![ci](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/ci.yml?branch=main&label=ci)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
[![verify-smoke](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/verify-smoke.yml?branch=main&label=verify-smoke)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/verify-smoke.yml)
[![docker-smoke](https://img.shields.io/github/actions/workflow/status/yukimurata0421/arena-eval-engine/docker-smoke.yml?branch=main&label=docker-smoke)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/docker-smoke.yml)
[![coverage-threshold](https://img.shields.io/badge/coverage-%E2%89%A586%25-brightgreen)](https://github.com/yukimurata0421/arena-eval-engine/actions/workflows/ci.yml)
[![version](https://img.shields.io/badge/version-0.3.0-blue)](https://github.com/yukimurata0421/arena-eval-engine/blob/main/CHANGELOG.md)

# ARENA — ADS-B Receiver Evaluation Engine

ARENA is a statistical evaluation engine that determines whether ADS-B receiver hardware changes actually improved performance — or whether observed differences are just traffic variation and noise.

It uses multiple complementary methods (Bayesian NB-GLM with NumPyro/NUTS, frequentist NB-GLM, Mann-Whitney U, change-point detection, OpenSky-normalized capture ratios, distance-band analysis) and draws conclusions from convergence or divergence across models. No single method is treated as authoritative.

## System Overview

```
Raspberry Pi (edge)                WSL2 / Linux (analysis)
┌──────────────────┐               ┌────────────────────────────┐
│  readsb → PLAO   │  rsync/pull   │                            │
│       → adsb-eval│──────────────>│  pipeline (8 stages,       │
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

- **Pipeline** — 8-stage orchestration with wave-parallel scheduling, failure-resilient execution, and append-only audit logging. Outputs human-readable graphs and reports to `/output`.
- **Artifacts** — SHA256-verified, schema-validated bundles with provenance, lineage, and deterministic replay. Integrity verification carries through to synthesis ingestion. Generates CSV-centric payloads to `/output/payload` for multi-LLM analysis handoff.
- **Synthesis** — Cross-model claim ingestion from multiple LLMs, enrichment, baseline clustering, proposition mapping, automated triage, and human review queue. Two-layer DB design (proposition + claim layers with convergence judgments). SQLite-backed, path-isolated.

Design decisions (why rsync --append, why CSVs over graphs for LLM input, why edge/analysis separation) are documented in [`docs/principles/`](docs/principles/).

## Design Philosophy

ARENA treats LLMs as hypothesis generators, not truth sources — claims are
validated through structured evidence and cross-model convergence.
The full catalogue of 31 engineering decisions is in
[docs/principles/engineering-decisions.md](docs/principles/engineering-decisions.md).

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
| Operations | [Architecture](docs/facts/architecture.md) · [Reproducibility](docs/facts/reproducibility.md) · [Synthesis](docs/facts/synthesis.md) |
| Design | [Engineering Decisions](docs/principles/engineering-decisions.md) · [Artifact Design](docs/principles/artifact-design.md) · [AI-Assisted Analysis](docs/principles/ai-assisted-analysis.md) · [AEME](docs/principles/aeme.md) |
| Context | [System Context](docs/facts/system-context.md) · [Statistical Assumptions](docs/principles/statistical-assumptions-and-limitations.md) |

## Tech Stack

Python 3.11+, NumPyro/JAX, PyMC, statsmodels, pandas, SQLite, Docker, GitHub Actions CI (lint + test matrix + coverage + smoke + Docker).

## Notes

- Secrets, credentials, and private data are excluded by design.
- Use environment variables or CLI overrides for local paths.
- Sample data under `sample_data/` is synthetic and deterministic.