# ADR: Artifact Subsystem Boundary for Research Review Bundles

## Status

Accepted

## Context

`merge_output_for_ai.py` originally started as a small utility script for merging text outputs.
Over time, its responsibilities expanded significantly and it effectively became the entry point
for producing research artifact bundles used in evaluation and review workflows.

The workflow now needs to support:

- discovering candidate artifacts across multiple output-like roots
- selecting required, recommended, and future-statistics artifacts
- preserving failure visibility in manifests
- packaging review bundles for different AI consumers
- generating documentation and settings snapshots
- running integrity checks over exported outputs

Keeping all of this functionality inside a single script created several problems:

- responsibilities were tightly coupled
- regression testing boundaries were unclear
- artifact lifecycle stages were difficult to reason about
- extending the system toward evaluation artifacts or reproducibility bundles became increasingly complex

To support long-term maintainability and extensibility, the artifact workflow needs explicit lifecycle boundaries.

## Decision

The implementation is reorganized as an artifact subsystem located under:

`scripts/tools/artifacts/`

The subsystem is structured around the lifecycle stages of research artifacts rather than around a single CLI script.

The following lifecycle stages define the subsystem boundaries:

- artifact discovery
- artifact selection
- artifact manifest
- artifact packaging
- artifact documentation
- artifact integrity

Each stage is implemented as a dedicated module to make responsibilities explicit and independently testable.

The original CLI path is preserved through a thin compatibility wrapper located at:

`scripts/tools/merge_output_for_ai/merge_output_for_ai.py`

This wrapper forwards execution to the new subsystem while preserving existing CLI behavior.

## Module Boundaries

The subsystem modules reflect the artifact lifecycle stages.

- `policies.py`
  Defines required and recommended targets, fallback rules, exclusion rules, category mappings, and AI pack policies.

- `models.py`
  Defines dataclasses used to transport audited state between lifecycle stages.

- `discovery.py`
  Resolves source paths, search roots, glob fallbacks, and priority-based discovery candidates.

- `selection.py`
  Produces the ordered artifact list from required, recommended, candidate, and discovered inputs.

- `manifest.py`
  Builds manifest records, copies selected files into the export bundle, and writes manifest-related CSV outputs.

- `packaging.py`
  Builds Gemini, GPT, and Grok packs, preserves relative layout where required, and emits pack manifests.

- `documentation.py`
  Generates summary reports, settings snapshots, methodology notes, hardware/date notes, and change-point notes.

- `integrity.py`
  Evaluates exported output consistency while preserving failure signals for auditability.

- `app.py`
  Orchestrates the lifecycle workflow and preserves the existing "normal merge" execution path.

- `cli.py`
  Owns argument parsing (`argparse`) and delegates execution to the application layer.

## Compatibility Policy

External behavior is preserved unless there is a strong reproducibility reason to change it.

The compatibility wrapper remains in place because:

- existing shell scripts and operational workflows depend on the legacy CLI path
- migration cost for users should remain minimal
- internal restructuring should not require immediate CLI migration

The following external behaviors are intentionally preserved:

- legacy CLI entry path
- option names
- output filenames
- manifest generation
- AI pack generation
- summary and documentation outputs
- integrity reporting behavior

## Failure-First and Reproducibility Philosophy

The subsystem is intentionally designed to preserve failure visibility rather than hide it.

Important artifact states include:

- `missing_required`
- `missing_recommended`
- `excluded_by_rule`
- `duplicate_source_skipped`
- `copy_failed`

These states form part of the audit trail for research artifact bundles.

They must remain observable through:

- manifest outputs
- candidate status reports
- documentation summaries
- integrity checks

The goal is not merely successful export.
The goal is producing reproducible review bundles with enough trace information to re-evaluate missing,
excluded, duplicated, or failed artifacts at a later time.

## Testing Strategy

Regression tests prioritize filesystem-realistic end-to-end behavior over isolated mocking.

This reflects the nature of the artifact subsystem, where correctness depends on file layout,
export behavior, and lifecycle ordering.

Current tests cover:

- AI export end-to-end generation of manifests, documentation, packs, and integrity outputs
- state visibility for missing, excluded, duplicate, and copy-failed artifacts
- stability of artifact selection order and fallback resolution
- duplicate handling behavior across runs
- legacy wrapper compatibility via dry-run execution
- documentation outputs containing required sections

Minimum validation commands:

```bash
python -m pytest tests/test_artifacts_subsystem.py -q
python -m pytest tests/test_artifacts_e2e.py -q
python scripts/tools/merge_output_for_ai/merge_output_for_ai.py --help
```

## Alternatives Considered

### Keep the single monolithic script

Rejected because:

- responsibilities were tightly coupled
- testing boundaries were unclear
- artifact lifecycle stages were difficult to isolate
- extending the system toward provenance, lineage, or replay workflows would further increase complexity

### Convert the script into a monolithic CLI tool

Rejected because:

- artifact lifecycle stages would remain tightly coupled
- packaging, documentation, and integrity checks would share hidden dependencies
- independent testing of lifecycle stages would remain difficult

### Lifecycle-based subsystem (chosen)

The chosen design organizes modules around artifact lifecycle stages, enabling:

- clearer separation of responsibilities
- improved testability
- easier extension toward reproducibility infrastructure features

## Trade-offs

Introducing a subsystem increases the number of modules and may initially increase navigation complexity.

However, this trade-off was accepted because:

- artifact lifecycle stages are expected to expand (provenance, replay, lineage, reproducibility)
- explicit lifecycle boundaries improve maintainability
- lifecycle modules allow independent testing and evolution

In practice, the subsystem reduces long-term complexity despite increasing short-term structural overhead.

## Deferred Changes

The following changes were intentionally postponed:

- replacing the flat `files/` export layout with a tree-preserving structure
- externalizing policy definitions into TOML or JSON
- redesigning print/log output formatting
- optimizing discovery and selection ordering

These changes were deferred because they carry higher compatibility risk
than the current subsystem hardening step.

## Future Extensions

The subsystem boundary is designed to support more than AI review bundles.

Likely future artifact bundle types include:

- AI review artifact bundles
- evaluation artifact bundles
- reproducibility bundles

The subsystem is therefore organized around artifact lifecycle stages
rather than around a single consumer or export script.
