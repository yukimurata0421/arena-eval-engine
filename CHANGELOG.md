# Changelog

All notable changes to this project will be documented in this file.

---

## [0.3.0] - 2026-03-28

### Added
- Public `synthesis` subsystem in the release layer:
  - `arena synthesis` CLI entrypoint and `synthesis` console script.
  - Core modules for ingest/repair/validate, enrichment, baseline clustering, proposition mapping, triage, and review-state updates.
  - Externalized proposition definitions in `src/arena/synthesis/config/propositions.json`.
- Public synthesis sample dataset:
  - `sample_data/synthesis/raw/{claude,gemini,gpt,grok}/20260327.json`
  - `sample_data/synthesis/README.md` with isolated validation commands.
- New synthesis documentation:
  - `docs/facts/synthesis.md` for path isolation, real-data run pattern, and verification checklist.
- Synthesis test suite migrated into release tree (`tests/test_synthesis_*.py`).

### Changed
- Version bump:
  - `pyproject.toml` -> `0.3.0`
  - `arena.__version__` -> `0.3.0`
- CLI integration:
  - `arena cli` now exposes `synthesis` (`distill` alias) and daily-operation shorthand forwarding.
- English unification:
  - Converted synthesis module/config/test strings to English-only wording.
- README refreshed for `v0.3.0` with synthesis quickstart, sample-data flow, and CI/Docker guidance.

### CI / Quality
- Coverage scope continues to include `src/arena` and `scripts`, now with synthesis paths included in normal test runs.
- Added synthesis smoke validation path for containerized runs (`arena-synthesis-smoke` compose service).

### Notes
- Synthesis execution is path-isolated by design:
  - defaults to `workspace/synthesis/`
  - supports full override via CLI arguments and `ARENA_SYNTHESIS_DIR`.
- Raw input files are never overwritten; original and repaired snapshots are persisted separately.

---

## [0.2.9] - 2026-03-20

### Changed
- Strengthened public-release definition and boundary messaging in `README.md`:
  - explicitly separates research/statistical reproducibility from release-layer smoke reproducibility
  - emphasizes deterministic public smoke verification as a release contract

### Added
- Release-layer import isolation guard:
  - added repository-local `arena` shim package so `import arena` resolves to this checkout (`src/arena`) even when other editable clones exist
- Public smoke reproducibility subsystem and fixture:
  - deterministic sample build/freeze/verify commands under `scripts/tools/sample_data/`
  - frozen expected outputs under `sample_data/smoke/expected/`
  - public reproducibility guide in `docs/facts/reproducibility.md`
- Coverage boundary hardening:
  - added `.coveragerc` to fix measurement scope to this release tree (`src/arena`, `scripts`)
  - integrated coverage flags into `pytest.ini`
  - added import-boundary runtime tests to detect path contamination
- Public packaging verification:
  - added tests for `scripts/tools/build_public_zip.py` exclusion rules and deterministic zip behavior

### Tests
- Added/expanded tests for previously untested utility modules:
  - `arena.lib.data_loader`
  - `arena.lib.input_utils`
  - `arena.lib.geo`
  - `arena.lib.platform_setup`
- Full suite passes with coverage-enabled pytest execution in release layer.

---

## [0.2.8] - 2026-03-19

### Changed
- Aligned package metadata with public branding:
  - `project.name`: `adsb-scripts` -> `arena-eval-engine`
  - updated project description to ARENA-focused wording
- Aligned release/version metadata across package and changelog:
  - `pyproject.toml` version -> `0.2.8`
  - `arena.__version__` -> `0.2.8`
- Aligned Python version baseline with documented release policy:
  - `requires-python` -> `>=3.11`
  - tooling targets updated to Python 3.11 (`ruff`, `mypy`)
- Removed ineffective `tool.setuptools.package-data.scripts` entry to avoid install-time ambiguity between editable/non-editable environments.

### Fixed
- Corrected cross-stage parallel scheduling so Stage 3 no longer races Stage 2 outputs (`fringe_decoding_stats.csv` dependency).
- Updated orchestration contract test expectations for the new stage-group behavior.
- Replaced broad exception handlers in `src` critical paths with `except Exception as exc` + debug logging:
  - `arena.lib.input_utils`
  - `arena.lib.platform_setup`
  - `arena.artifacts.repro_stamp`

### Validation
- Re-ran Docker real-data full pipeline (`arena-real-full`) through Stage 1-8.
- Confirmed artifact packaging flow works on real-data outputs:
  - `python -m arena.artifact_cli run`
  - `python -m arena.cli artifacts verify`
  - `python -m arena.cli artifacts replay`
- Re-ran test suite and coverage in public release tree:
  - `coverage run -m pytest -q`
  - `coverage report -m` (TOTAL 86%)
- Verified non-editable install flow:
  - `pip install .`
  - `python -m arena.cli --help`
  - `python -m arena.cli validate`
  - `arena --help`
  - `artifact --help`

### Docs
- Updated environment setup guidance for:
  - minimum public setup (`.[dev]`)
  - full Stage 1-8 runtime dependencies
  - Docker real-data prerequisites and cleanup steps
- Added reproducible public zip packaging guidance using `scripts/tools/build_public_zip.py`.

### Security
- Removed runtime-injected private inputs after verification (`docker/.env`, temporary real-output directories).
- Added explicit clean-distribution packaging flow to exclude VCS/build/cache residues from release zips.

### Tech Debt
- Broad `except Exception` blocks remain in multiple `scripts/` modules by design for now.
- These are tracked as future refactor targets to improve error granularity and diagnostics.

---

## [0.2.7] - 2026-03-19

### Changed
- Updated Docker runtime defaults for public release:
  - removed dependency on deleted `scripts/dev/*` smoke scripts
  - switched container default command to CLI help
  - made container data/output directories runtime-created
- Added real-data Docker execution path with host-mounted settings/phase/credentials via `docker/.env`.
- Added documentation for opt-in real-data validation without persisting private data in repository.

### Security
- Added `.env` / `docker/.env` ignore rules to prevent accidental credential/path commits.

---

## [0.2.6] - 2026-03-19

### Changed
- Split pipeline builder responsibilities into modular stage constructors and shared orchestration flow.
- Standardized pipeline build-time environment separation with `PipelineBuildOptions`.
- Hardened public release defaults to avoid local absolute paths and private host/user settings.
- Updated OpenSky credential handling to prefer environment variables (`OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET`).

### Tests
- Increased CLI-oriented coverage (`--help`, `validate`, staged run entrypoints and env resolution paths).
- Strengthened pipeline contract tests around stage ordering, options resolution, and entrypoint orchestration.

---

## [0.2.5] - 2026-03-15

### Added
- Added runtime optional dependency groups in `pyproject.toml`:
  - `run_cpu` for Stage 4 execution (`jax`, `jaxlib`, `numpyro`)
  - `run_full` for full pipeline execution including Stage 5 (`jax`, `jaxlib`, `numpyro`, `pymc`, `arviz`)

### Changed
- Updated dependency installation guidance in `README.md` to explicitly distinguish:
  - validate/tests/smoke setup (`.[dev]`)
  - Stage 4 runtime setup (`.[dev,gpu]` or `.[run_cpu]`)
  - Stage 5/full runtime setup (`.[dev,bayes,gpu]` or `.[run_full]`)
- Updated `all` extra to follow the full runtime dependency path.
- Improved pipeline missing-module guidance in `arena.pipeline.entrypoint`:
  - Stage 4 missing modules now suggest `.[dev,gpu]`
  - Stage 5 missing modules now suggest `.[dev,bayes,gpu]`
  - Duplicate missing module names are de-duplicated in error output

### Validation
- Verified with real data root `<local_data_root>`:
  - Stage 1 executes successfully with `--only 1 --no-gpu --skip-plao`
  - Stage 4 executes successfully after installing `jax`/`numpyro`
  - Stage 5 requires and executes successfully after adding `pymc`/`arviz`
- Verified artifact export/verify/replay workflow succeeds in the updated environment.

---

## [0.2.4] - 2026-03-14

### Fixed
- Aligned the packaged project version with the published Docker full-run release.
- Added the missing release metadata so the changelog and package version match the latest public tag.

---

## [0.2.3] - 2026-03-14

### Fixed
- Corrected changelog alignment after the Docker full-run release.

---

## [0.2.2] - 2026-03-14

### Added
- Docker full-run support for `arena run` in the public CPU image.
- `arena-run` service in `docker/docker-compose.yml` for running the pipeline through Docker Compose.

### Changed
- Updated `docker/Dockerfile.cpu` to install the full pipeline dependency set required by Stage 4 and Stage 5.
- Expanded README Docker documentation to distinguish smoke validation from full pipeline execution.

### Validation
- Verified Docker build succeeds with the full dependency set.
- Verified `arena validate` and `arena run --only 1 --dry-run --skip-plao` through the new Compose service.
- Verified the full Stage 1-8 pipeline against real telemetry in a scratch output tree using Docker.

---

## [0.2.1] - 2026-03-14

### Fixed
- Fixed Ruff lint failures in artifact tool entrypoints caused by intentional early `sys.path` initialization.
- Added file-level `E402` exceptions for artifact entrypoint modules that must modify `sys.path` before local imports.
- Normalized import ordering across affected modules with Ruff autofix.
- Fixed an OS-dependent runner test that expected Windows path separators and failed on Linux CI.

### Tests
- Confirmed `ruff check src tests scripts/tools/artifacts scripts/dev` passes.
- Confirmed `python -m pytest -q -m "not real_data"` passes.
- Confirmed the non-real-data test suite passes on WSL Ubuntu 22.04 with Python 3.11.
- Confirmed `coverage run -m pytest -q -m "not real_data" && coverage report` passes in the same Linux environment.

### Notes
- Docker smoke was not re-run in this fix environment.

---

## [0.2.0] - 2026-03-14

### Summary

This release upgrades ARENA from a public statistical evaluation engine
to a more reproducible, failure-visible, and compatibility-preserving research codebase.

Major themes:
- Public artifact subsystem with verify/replay support
- Modular pipeline package replacing the monolithic orchestrator
- Stronger CI, smoke validation, and compatibility guarantees
- Public sample data sanitized for privacy-safe reproducibility

### Added

**Artifact subsystem** (`src/arena/artifacts/`)
- Public artifact substrate covering discovery, selection, manifest generation, provenance, integrity verification, and replay.
- SHA256 hash verification, deterministic export mode, and bundle-level identity (`bundle_sha256`).
- JSON Schema validation for all bundle outputs (manifest, candidate status, provenance, integrity, run metadata, artifact index).
- CLI: `arena artifacts verify <bundle>` and `arena artifacts replay <bundle>`.
- Design decision documented in `docs/adr/ADR-artifact-subsystem.md`.

**Modular pipeline** (`src/arena/pipeline/`)
- Pipeline orchestration split into focused modules: entrypoint, stages, runner, decision, backend, record_io, error_policy.
- Dependency-aware skip logic: `--skip-existing` now detects stale outputs when upstream inputs have been updated.
- Structured error codes and recommended recovery actions.

**Smoke validation**
- `run_sample_smoke.py`: validate → dry-run → artifact export → verify → replay.
- `run_real_data_smoke.py` with PowerShell/Bash wrappers for opt-in local validation.
- Docker smoke job in CI.

**Documentation**
- `docs/facts/architecture.md`: Full architecture document with system context, package structure, and execution backend description.
- `docs/principles/aeme.md`: Design philosophy with method rationale table and dual-baseline explanation.
- `docs/facts/failure-taxonomy.md`: Structured failure reference with error codes, conditions, and recovery actions.
- `docs/facts/sample_outputs.md`: Per-file descriptions for all sample outputs.
- `docs/facts/real-data-smoke.md`: Opt-in real-data validation guide.

### Changed

- Refactored pipeline orchestration from single file into package with separated concerns.
- Replaced module-level global path variables with injectable functions (`resolve_scripts_root()`, `resolve_output_dir()`, `resolve_data_dir()`).
- Added `settings_loader.py` for TOML settings discovery and `_toml_compat.py` for fallback support.
- Expanded CI from basic lint/test to 5 jobs: lint, tests (ubuntu + windows matrix), coverage (`fail_under = 55`), compatibility, docker-smoke.
- Expanded test coverage from statistical core to pipeline failure branches, artifact integrity/replay, compatibility shims, and smoke workflows.
- Coarsened receiver and aircraft coordinates in public sample fixtures to reduce disclosure risk while preserving interface validation value.
- Excluded operational `output/performance/` content from the public layer.
- Package name: `adsb-scripts` → `arena-eval-engine`.
- `pyproject.toml`: added `jsonschema>=4.23`, pytest markers (`slow`, `gpu`, `real_data`), `requires-python >= 3.11`.
- README updated for public-release operation.

### Compatibility Notes

- Legacy merge entrypoints remain available (`scripts/tools/merge_output_for_ai/merge_output_for_ai.py`).
- Legacy artifact tooling paths preserved through compatibility re-exports and migration shims.
- Artifact state names (`missing_required`, `excluded_by_rule`, etc.), output filenames, and selection semantics are unchanged.
- Migration regression test asserts alias integrity between legacy and core modules.
- This release is structurally significant but not a breaking public CLI change.

### Removed

- Monolithic `src/arena/pipeline.py` (replaced by `src/arena/pipeline/` package).

---

## [0.1.9] - 2026-03-08

### Fixed
- Corrected logging behavior for `pipeline_runs.jsonl`.
- Introduced configurable logging mode: `--log-jsonl-mode {append, overwrite}`.
  - `append`: preserves append-only audit log behavior.
  - `overwrite`: clears the log at run start to keep only the current execution.

- Fixed data contamination in `merge_output_for_ai.py`.
  - Excluded `performance/pipeline_runs*.jsonl` from merged outputs.
  - Added wildcard exclusion using `fnmatch`.
  - Implemented tail-priority reading for large `.jsonl` and `.log` files.

- Corrected numerical values in the **Statistical assumptions and limitations** section.
  - Previous values were derived from merged outputs that included audit log data.
  - After fixing the merge logic and regenerating results, the statistical values were recalculated.
  - Statistical assumptions themselves remain unchanged.

### Added
- Regression test for JSONL logging mode.
- Added `test_log_jsonl_mode_overwrite` to verify overwrite-mode behavior.

### Validation
- Verified using real pipeline data with repeated `dry-run` executions.
- Confirmed correct behavior for both logging modes.
- Confirmed `performance/pipeline_runs*.jsonl` appears as `excluded_file` in the merge manifest.

---

## [0.1.8] - 2026-03-08
### Fixed
- Corrected the project version in `pyproject.toml` for the release.
- Added a follow-up release because the previous release was published with an incorrect package version.


## [0.1.7] - 2026-03-08

### Added
- Added `docs/principles/statistical-assumptions-and-limitations.md` to document model assumptions, failed specifications, proxy limitations, and evidence boundaries.
- Added README links to detailed methodology and limitation documents.

### Changed
- Reworked `README.md` for public release readiness.
- Improved architecture documentation to reflect the actual runtime data flow.
- Added input contract documentation for required telemetry files.
- Clarified sample data scope, example outputs, and statistical philosophy.

---

## [0.1.6] - 2026-03-08

### Added
- Added test coverage measurement with `pytest-cov`.
- Added GitHub Actions coverage job for Python 3.11.

### Changed
- Reworked README structure for public release readiness.
- Updated `.gitignore` to exclude generated coverage artifacts.

### Notes
- Test suite: 15 passed, total coverage: 43%.

---

## [0.1.5] - 2026-03-07
### Added
- Added `docs/principles/aeme.md` for AEME analytical framework documentation.
- Added documentation links from README to architecture and AEME details.

### Changed
- Refactored README for readability.
- Clarified the relationship between `PLAO`, `adsb-eval`, and `ARENA`.

---

## [0.1.4] - 2026-03-07
### Added
- Added lightweight public test suite.
- Added `scripts/tools/create_public_samples.py` for reproducible sample dataset generation.

### Changed
- Updated README with minimal local and Docker reproducibility steps.
- Clarified public data policy to track sample datasets only under `data/sample/` and `output/sample/`.
- Strengthened ignore rules to prevent committing raw data and local caches.

### Removed
- Removed large experimental datasets and full processing outputs from the public tree.

---

## [0.1.2] - 2026-03-01
### Changed
- Pinned dev tool versions for reproducibility.
- Stabilized CI workflow (Python 3.11 / 3.12).
- Enforced ruff lint and format checks.

---

## [0.1.1] - 2026-03-01
### Changed
- Established stable baseline for reproducible ADS-B coverage evaluation.

---

## [0.1.0] - 2026-02-28
### Added
- Initial public release.
- Core evaluation engine (AEME statistical framework).
- CLI orchestration layer.

