# Changelog

All notable changes to this project will be documented in this file.

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
- `docs/architecture.md`: Full architecture document with system context, package structure, and execution backend description.
- `docs/aeme.md`: Design philosophy with method rationale table and dual-baseline explanation.
- `docs/failure-taxonomy.md`: Structured failure reference with error codes, conditions, and recovery actions.
- `docs/sample_outputs.md`: Per-file descriptions for all sample outputs.
- `docs/real-data-smoke.md`: Opt-in real-data validation guide.

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
- Added `docs/statistical-assumptions-and-limitations.md` to document model assumptions, failed specifications, proxy limitations, and evidence boundaries.
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
- Added `docs/aeme.md` for AEME analytical framework documentation.
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
