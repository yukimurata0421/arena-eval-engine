# Real-Data Docker Validation

This repository keeps real-data execution opt-in. Use Docker with local mounts so private data and
credentials never need to be copied into this repo.

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Local private data/settings/phase/credentials files prepared outside this repository
- Writable temporary output path dedicated for this validation run

## Required Local Inputs

- Real telemetry root (example: `<private_data_root>`)
- Real settings file with your receiver location (example: `<private_settings.toml>`)
- Phase config file (example: `<private_phases.txt>`)
- OpenSky credentials JSON (example: `<private_opensky_credentials.json>`)

## Setup

1. Copy `docker/.env.example` to `docker/.env`.
2. Edit `docker/.env` with your local absolute paths.
3. Optionally use repository templates in `real-settings/` as a starting point:
   - `real-settings/settings.toml`
   - `real-settings/phases.txt`
Then copy them to a private path and point `docker/.env` to that private copy.

`docker/.env` is ignored by Git.

## Run Stage 1 Against Real Data

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml --profile real run --rm arena-real-stage1
```

## Run Full Stage 1-9 Pipeline Against Real Data

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml --profile real run --rm arena-real-full
```

Stage 9 writes the evidence synthesis artifacts after the statistical, PLAO, and OpenSky comparison
stages complete:

- `performance/model_evidence_matrix.csv`
- `performance/model_evidence_summary.json`
- `performance/model_disagreement_report.md`

## Native Real-Data Validation

Docker is the preferred public validation surface, but the same path-isolated contract applies to
native execution:

```bash
arena run \
  --scripts-root <repo_root>/scripts \
  --data-dir <private_data_root> \
  --output-dir <temporary_output_root> \
  --settings <private_settings.toml> \
  --phase-config <private_phases.txt> \
  --workers <logical_cpu_count>
```

Keep `<private_data_root>`, settings, phase definitions, and credentials outside this repository.
Runtime outputs should go to a scratch directory that is ignored by Git.

## Cleanup

- Remove `docker/.env` after validation.
- Remove temporary output directory you mapped with `ARENA_REAL_OUTPUT_ROOT`.
- Keep credentials outside this repository.
