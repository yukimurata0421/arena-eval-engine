# ARENA Architecture (Public Release Layer)

## What ARENA Is

ARENA is an **evaluation runtime and reproducibility surface** for the public release layer.
It is not only a collection of research scripts.

This document is the canonical architecture description for the current public repository state.
Detailed release evolution from `v0.2.9` to `v0.3.0` is documented in `docs/evolution/v0.2.9-to-v0.3.0.md`.

This document does not replace:

- `docs/principles/artifact-design.md` (why artifact control exists and what responsibilities it carries)
- `docs/principles/ai-assisted-analysis.md` (how AI is used and what is not trusted automatically)
- `docs/facts/reproducibility.md` (public smoke reproducibility contract)
- `docs/facts/synthesis.md` (synthesis path-isolated ingest/triage/review workflow)

## System Position

In the broader stack, ARENA is positioned as the evaluation/reproducibility layer.

- PLAO: raw position logging
- adsb-eval: edge-side metric/telemetry generation
- ARENA: orchestration, statistical evaluation runtime, artifact control, and public reproducibility surface

This architecture document focuses on ARENA's public-layer responsibilities, not upstream system internals.

## Design Goals

The public architecture prioritizes:

1. Clear responsibility boundaries between orchestration, payload execution, runtime support, and artifact control.
2. Explicit execution auditability for each run.
3. Explicit config resolution and validation before execution.
4. Reproducibility contracts that can be checked in CI and smoke flows.
5. AI-assisted analysis as hypothesis-generation support under human judgment.
6. Clear separation between public-release guarantees and development-side extensions.

## Responsibility Boundaries

### CLI / entrypoint

`src/arena/cli.py` is responsible for:

- public runtime command surface
- path/config override intake
- config resolution/validation handoff before execution
- delegation to pipeline runtime and artifact verify/replay operations

`src/arena/artifact_cli.py` is responsible for:

- artifact export command surface (`artifact run`)
- compatibility delegation to tool-layer artifact CLI

Boundary:

- CLI defines supported public entrypoints.
- CLI does not implement payload analytics or artifact packaging internals.

### Pipeline / orchestration

`src/arena/pipeline/` is responsible for:

- step/stage contracts
- orchestration flow control
- runtime execution behavior and reporting
- backend/environment selection
- decision/error policies
- run-record serialization

Boundary:

- The pipeline is responsible for execution control and expected-output enforcement.
- Payload scripts remain responsible for domain computations.

### Shared runtime support

`src/arena/lib/` is responsible for:

- path/root resolution
- settings/phase resolution metadata
- settings and runtime snapshot loading
- shared runtime/config helpers used by orchestration and payload layers

Boundary:

- This layer provides reusable runtime substrate.
- It does not own orchestration policy or artifact policy.

### Scripts as payload layer

`scripts/` is responsible for:

- stage payloads used by `arena run` (`adsb/`, `signals/`, `plao/`)
- release/runtime support tools (`tools/sample_data/`, `tools/artifacts/`, `tools/merge_output_for_ai/`)
- runtime configuration assets (`config/`)
- compatibility entrypoints (`master.py`, `phase_config.py`)

Current public-tree structure (summary):

```text
scripts/
├─ adsb/      # aggregation, stats, bayesian, change-point, reports, heatmap, ops
├─ signals/   # signal aggregators/evaluators
├─ plao/      # PLAO distance-AUC evaluation
├─ tools/     # smoke sample, artifact compatibility, packaging/ops helpers
├─ config/    # default settings + phase config templates
├─ master.py
└─ phase_config.py
```

Boundary:

- Scripts implement payload behavior.
- Orchestration and public execution contracts stay in CLI/pipeline layers.

### Observability / execution audit

Execution auditability is provided by:

- append-only run records (`output/performance/pipeline_runs.jsonl`)
- per-step status/timing/command/output-check logging
- config snapshot logging at run start
- structured error-code reporting

Boundary:

- This is run/audit visibility for execution behavior.
- It is not equivalent to scientific validity of analysis conclusions.

### Configuration resolution / validation

Configuration responsibilities are split across:

- path resolution metadata (`src/arena/lib/config_resolution.py`)
- settings/runtime snapshot construction (`src/arena/lib/runtime_config.py`)
- pre-run validation (`arena validate`)
- run-path config checks before orchestration starts

Boundary:

- Resolution and validation are first-class runtime inputs.
- Payload scripts consume resolved state.

### Artifact / reproducibility layer

In this public architecture, artifact/reproducibility is not a peripheral export helper.
It is part of the control surface that carries execution-side consistency into AI-assisted interpretation, auditability, and re-validation.

Confirmed public implementation surfaces:

- core artifact substrate: `src/arena/artifacts/`
- tool/compatibility layer: `scripts/tools/artifacts/`
- bundle verification/replay via CLI (`arena artifacts verify`, `arena artifacts replay`)
- artifact export entrypoint (`artifact run` or `python -m arena.artifact_cli run`)
- artifact subsystem tests under `tests/`

Boundary:

- This document defines placement/responsibility of artifact control in the public architecture.
- `docs/principles/artifact-design.md` defines detailed rationale and design necessity.

### Tests / CI / smoke / verification surface

Public verification surface includes:

- test suites under `tests/`
- workflow surface under `.github/workflows/`
- smoke reproducibility fixtures under `sample_data/smoke/expected/`

Detailed workflow-level release evolution belongs in `docs/evolution/v0.2.9-to-v0.3.0.md`.

Boundary:

- These checks validate release-layer execution/reproducibility contracts.
- They do not by themselves establish research/statistical validity.

## End-to-End Execution Flow

Public release-layer flow:

1. Resolve runtime config paths and metadata.
2. Validate runtime prerequisites.
3. Build pipeline step plan from stage contracts and options.
4. Execute payload steps according to orchestration policy.
5. Validate expected outputs and apply fail/soft-fail policy.
6. Persist run/config records for audit.
7. Build artifact bundles for structured review/revalidation.
8. Verify/replay artifact bundles when required.
9. Feed AI-assisted outputs into human-managed validation loops.

Detailed version-to-version execution changes belong in `docs/evolution/v0.2.9-to-v0.3.0.md`.

## Reproducibility and Auditability Model

The public reproducibility/auditability model combines:

- deterministic smoke tooling and fixed expected outputs
- explicit config resolution + validation
- append-only execution logging and config snapshots
- output contract checks in runtime execution
- artifact integrity/provenance/lineage/hash/index checks
- verify/replay execution paths

Bottleneck principle:

- overall analysis quality is capped by the weakest stage
- this includes AI input/control quality, not only upstream data/statistical stages

## Position of AI-Assisted Analysis

AI-assisted analysis is positioned as:

- hypothesis-generation support, not a truth engine
- a validation-loop input, not an automatic conclusion
- cross-model validation where agreement is baseline and disagreement is a validation target

Boundary:

- This document defines system position.
- `docs/principles/ai-assisted-analysis.md` defines detailed operating/trust rules.
- `docs/principles/artifact-design.md` defines why artifact control is required for that model.

## Public Release Boundary vs Development Boundary

Public release layer centers on:

- CLI + pipeline control plane
- shared runtime/config support
- payload scripts
- artifact verify/replay-capable subsystem
- public verification surface

Development boundary:

- database-oriented extensions are documented as development-side and not part of the current public reproducibility path

## Reading Guide

Recommended reading order:

1. `docs/facts/system-context.md` (optional stack/context primer)
2. `src/arena/cli.py`, `src/arena/artifact_cli.py`
3. `src/arena/pipeline/`
4. `src/arena/lib/`
5. `src/arena/artifacts/` and `scripts/tools/artifacts/`
6. `tests/` and `.github/workflows/`
7. `docs/facts/reproducibility.md`
8. `docs/facts/synthesis.md`
9. `docs/principles/artifact-design.md` and `docs/principles/ai-assisted-analysis.md`
10. `docs/evolution/v0.2.9-to-v0.3.0.md` for release-evolution details

