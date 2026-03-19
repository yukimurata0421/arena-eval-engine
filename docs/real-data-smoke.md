# Real-Data Docker Validation

This repository keeps real-data execution opt-in.
Use Docker with local mounts so private data and credentials never need to be copied into this repo.

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

`docker/.env` is ignored by Git.

## Run Stage 1 Against Real Data

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml --profile real run --rm arena-real-stage1
```

## Run Full Stage 1-8 Pipeline Against Real Data

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml --profile real run --rm arena-real-full
```

## Cleanup

- Remove `docker/.env` after validation.
- Remove temporary output directory you mapped with `ARENA_REAL_OUTPUT_ROOT`.
- Keep credentials outside this repository.
