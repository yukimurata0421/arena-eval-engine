# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.3] - 2026-03-07
### Added
- Added lightweight public test suite (`test_cli`, `test_schema`, `test_settings`, `test_pipeline_minimal`, `test_jsonl_parser`)
- Added `scripts/tools/create_public_samples.py` to generate reproducible sample datasets and clean original `data/` and `output/` files

### Changed
- Updated README with minimal local and Docker reproducibility steps for public users
- Clarified public data policy to track sample datasets only under `data/sample/` and `output/sample/`
- Strengthened ignore rules (`.gitignore`, `.dockerignore`) to prevent committing raw data, output artifacts, and local caches
- Aligned package metadata version with this release (`pyproject.toml` -> `0.1.3`)
- Switched Docker CPU image command from `bash -lc` to `sh -c` for better portability on slim images

### Removed
- Removed large experimental raw datasets and full processing outputs from the public tree (sample-only retained)
- Removed local build/cache artifacts from repository contents (`__pycache__`, `*.pyc`, `*.egg-info`)

---

## [0.1.2] - 2026-03-01
### Changed
- Pinned dev tool versions for reproducibility
- Stabilized CI workflow (Python 3.11 / 3.12)
- Enforced ruff lint and format checks
- Added pre-commit integration
- CI passing consistently

---

## [0.1.1] - 2026-03-01
### Changed
- Established stable baseline for reproducible ADS-B coverage evaluation
- Minor formatting and lint cleanup

---

## [0.1.0] - 2026-02-28
### Added
- Initial public release
- Core evaluation engine (AEME statistical framework)
- CLI orchestration layer
