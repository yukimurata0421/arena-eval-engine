# Artifact Subsystem

This directory implements the research artifact subsystem used by `merge_output_for_ai.py`.

It exists to turn research outputs into reviewable, reproducible artifact bundles without hiding failure states.

## Layering (Incremental Migration)

The subsystem is split into two layers:

- artifact substrate (public core): `src/arena/artifacts/`
- tool layer (orchestration + compatibility): `scripts/tools/artifacts/`

Core modules are promoted to `src/arena/artifacts/` and remain import-compatible through legacy shim modules under `scripts/tools/artifacts/`.

## Responsibilities

The subsystem is organized around artifact lifecycle stages:

- artifact discovery
- artifact selection
- artifact manifest
- artifact packaging
- artifact documentation
- artifact integrity

## Artifact Lifecycle

The subsystem is now treated as research artifact infrastructure, not only as an export helper.

The lifecycle is:

1. discover candidate outputs
2. select required, recommended, and candidate artifacts
3. write audit manifests with failure states preserved
4. package AI-facing review bundles
5. write documentation and reproducibility metadata
6. verify exported bundle integrity for later replay

## Artifact Schema

Schema validation is defined in `schema.py` and is applied before writing or accepting:

- manifest records
- candidate status records
- pack manifest payloads
- integrity summaries

Validation failures raise immediately. The subsystem does not silently coerce malformed records.

## Reproducibility Guarantees

The subsystem keeps failure visibility for:

- `missing_required`
- `missing_recommended`
- `excluded_by_rule`
- `duplicate_source_skipped`
- `copy_failed`

Selected exported artifacts receive SHA256 hashes. The manifest records those hashes and `artifact_hashes.txt` provides a revalidation source for later integrity checks.

`reproducibility_stamp.json` records:

- timestamp
- python version
- platform
- git commit when available
- artifact subsystem version
- export mode
- deterministic flag
- policy version

Additional lifecycle metadata is emitted as JSON:

- `artifact_provenance.json`
- `run_metadata.json`
- `artifact_lineage.json`
- `integrity_summary.json`
- `artifact_index.json`

## Deterministic Export

The CLI supports:

```bash
python scripts/tools/merge_output_for_ai/merge_output_for_ai.py --deterministic ...
```

Deterministic mode normalizes generated timestamps and stabilizes ordering where the subsystem controls it:

- generated documentation timestamps
- summary metadata
- pack manifest timestamps
- packaging traversal order
- normal mode `mtime` sort tie-breaks

The intent is that identical input data produces identical artifact contents for the core exported outputs.

## Hash Verification

`integrity.py` verifies:

- duplicate output paths
- missing exported artifacts
- SHA256 mismatches against recorded hashes

`artifact_hashes.txt` uses the format:

```text
<sha256>  <relative_export_path>
```

This allows later revalidation of the exported artifact set without re-running discovery or selection.

`artifact_index.json` records a deterministic `bundle_sha256` computed from:

- sorted artifact hashes
- manifest rows
- schema catalog

This provides a stable bundle identity for CI comparison and long-term audit.

## Artifact Provenance

`artifact_provenance.json` records how each exported artifact was produced.

Each entry contains:

- `artifact_path`
- `artifact_sha256`
- `source_paths`
- `generation_stage`
- `generation_timestamp`
- `policy_version`

The file is intended for audit and later provenance consistency checks. Verification fails if provenance disagrees with the manifest or hash inventory.

## Experiment Lineage

`artifact_lineage.json` defines artifact relationships using:

- `nodes`
- `edges`
- `artifact_types`

Current lineage edges use `derived_from` from source paths to exported artifact paths. This keeps the bundle extensible toward evaluation artifacts and reproduction artifacts without changing the existing export structure.

## Verification CLI

Installed CLI:

```bash
arena artifacts verify
arena artifacts verify <artifact_bundle>
```

Verification checks:

- manifest and candidate-status schema validity
- pack manifest structure
- integrity summary consistency
- SHA256 hashes
- duplicate output path detection
- provenance consistency
- bundle hash consistency

Exit code is `0` only when the bundle is valid.

## Replay Capability

Installed CLI:

```bash
arena artifacts replay <artifact_bundle>
```

Replay performs verification again, prints reproducibility metadata, prints run metadata, and lists missing artifacts or failure-visible states that still matter for later review.

This is intentionally not a pipeline rerun. It is a bundle-level revalidation and audit replay.

## CI Validation

The GitHub Actions workflow runs:

- pytest
- flake8
- legacy CLI help smoke test
- deterministic export smoke test
- artifact verification smoke test
- replay smoke test
- deterministic bundle hash comparison

The deterministic smoke test uses a minimal temporary fixture and checks that the subsystem can produce a deterministic export without relying on repository-local output data.

## Revalidation Later

Artifact bundles can be revalidated later without rerunning discovery or selection:

1. validate schema-bearing outputs
2. validate recorded SHA256 hashes
3. validate provenance against the manifest and hash inventory
4. compare the stored `bundle_sha256`

This supports reproducibility review, CI drift detection, and later audit of research bundles.

## Validation Commands

```bash
python -m pytest tests -q
arena artifacts verify
arena artifacts replay <artifact_bundle>
python scripts/tools/merge_output_for_ai/merge_output_for_ai.py --help
```
