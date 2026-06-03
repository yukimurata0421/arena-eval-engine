# Engineering Decisions

Every technical choice in ARENA exists for a reason.  This document catalogues 33 design decisions
organised by data-flow stage — from the Raspberry Pi edge through the analysis pipeline, artifact
packaging, multi-LLM handoff, and synthesis database.  Cross-cutting patterns (append-only logging,
SHA-256 integrity chains, idempotency, visible failure) are summarised at the end.

---

## 1. Edge (Raspberry Pi)

### 1-1. No heavy processing on the edge

Locations: [backend.py](../../src/arena/pipeline/backend.py),
[architecture.md](../facts/architecture.md)

Rationale: The Raspberry Pi runs 24/7 as an observation instrument. Heavy processing is kept off the
edge not because of resource constraints, but because there is no necessity to run it there —
analysis tasks that do not require real-time execution belong on the workstation.  The same
principle applies to airband-ai's use of a GCP API instead of a local LLM: consuming local resources
would compromise immediacy, interconnection, and maintainability of the observation platform.

### 1-2. Safe sync options (SSH / rsync --append)

Locations: [rpi_log_sync.py](../../scripts/adsb/ops/rpi_log_sync.py)

Rationale: `rsync --append` minimises bandwidth consumption by transferring only newly appended
data.  If corruption occurs, it is confined to the affected file segment — the rest of the dataset
remains intact.  Additional flags suppress interactive prompts, prevent inconsistent partial copies,
and exclude transient files.

---

## 2. Pipeline (9 stages, wave-parallel)

### 2-1. Wave-parallel scheduling

Locations: [entrypoint.py](../../src/arena/pipeline/entrypoint.py),
[stages.py](../../src/arena/pipeline/stages.py)

Rationale: Stages are subdivided into waves with explicit dependency metadata. Steps within the same
wave run in parallel; dependent waves execute sequentially. On the current Xeon E5-2695 v4
workstation, pipeline workers use the 36 logical CPU threads by default while dependency boundaries
keep Stage 3 and Stage 9 from consuming incomplete upstream outputs.

### 2-2. Runtime environment abstraction (native / WSL)

Locations: [backend.py](../../src/arena/pipeline/backend.py)

Rationale: Path translation and execution routing between Windows native and WSL are abstracted
behind a `Backend` class, preventing OS-specific failures from halting the pipeline.

### 2-3. Encoding and headless rendering hardening

Locations: [backend.py](../../src/arena/pipeline/backend.py)

Rationale: `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` prevent CP932 encoding crashes on Windows.
`MPLBACKEND=Agg` forces matplotlib into a non-interactive backend for headless/batch operation.

### 2-4. GPU detection failure → CPU fallback

Locations: [backend.py](../../src/arena/pipeline/backend.py)

Rationale: Earlier small-N benchmarks showed that `DiscreteHMCGibbs` with n = 59 was 38–50× slower
on a GTX 1060 than on CPU. The current workstation has a GTX 1070, but the active JAX runtime
exposes CPU only. The pipeline now reports physical NVIDIA hardware separately from JAX CUDA
availability and falls back to CPU when CUDA is unavailable or the dataset is below the GPU
threshold.

### 2-5. Configuration path resolution and pre-flight validation

Locations: [config_resolution.py](../../src/arena/lib/config_resolution.py),
[cli.py](../../src/arena/cli.py), [entrypoint.py](../../src/arena/pipeline/entrypoint.py)

Rationale: Configuration mismatches are caught before any stage executes, avoiding wasted compute
from a long-running pipeline that fails mid-way due to a bad path.

### 2-6. Safe default date on phase-config load failure

Locations: [entrypoint.py](../../src/arena/pipeline/entrypoint.py)

Rationale: If the phase configuration file is unreadable, the pipeline continues with an explicit
warning and a safe default date rather than aborting entirely.  Graceful degradation is preferred
over a full stop.

### 2-7. Append-only execution audit log (JSONL)

Locations: [record_io.py](../../src/arena/pipeline/record_io.py),
[runner.py](../../src/arena/pipeline/runner.py)

Rationale: Every pipeline execution appends a structured JSON record to a log file.  Append-only
semantics make tampering and accidental truncation structurally impossible, preserving a complete
audit trail.

### 2-8. Output contract checks (existence / size / freshness)

Locations: [runner.py](../../src/arena/pipeline/runner.py),
[stages.py](../../src/arena/pipeline/stages.py)

Rationale: A zero exit code does not guarantee correct output.  Each step declares expected outputs
with minimum byte sizes; the runner verifies them after execution.  Empty, undersized, or stale
files trigger a failure even when the process itself exited cleanly.

### 2-9. Dependency-mtime skip control (stale-output guard)

Locations: [decision.py](../../src/arena/pipeline/decision.py),
[stages.py](../../src/arena/pipeline/stages.py)

Rationale: Prevents accidental reuse of outdated artefacts by comparing output modification times
against their input dependencies.

### 2-10. Two-tier failure policy (soft-fail / fail-fast)

Locations: [stages.py](../../src/arena/pipeline/stages.py),
[decision.py](../../src/arena/pipeline/decision.py),
[entrypoint.py](../../src/arena/pipeline/entrypoint.py)

Rationale: Critical steps (e.g., daily AUC aggregation) abort the pipeline immediately on failure.
Non-critical steps (e.g., OpenSky data fetch) are allowed to fail softly so that downstream analysis
can proceed with stale but still usable data.

### 2-11. Explicit timeout and exception handling

Locations: [runner.py](../../src/arena/pipeline/runner.py)

Rationale: Every step has a `timeout_s`.  Hung processes are killed and the step is recorded as
timed-out rather than left running indefinitely.  Uncaught exceptions are captured, logged, and
converted to structured failure records.

### 2-12. Error code taxonomy + recommended recovery actions

Locations: [error_policy.py](../../src/arena/pipeline/error_policy.py)

Rationale: Structured error codes map to predefined recovery suggestions, standardising incident
response and making failures reproducible and actionable.

### 2-13. Lock-based record and print isolation

Locations: [runner.py](../../src/arena/pipeline/runner.py)

Rationale: Parallel wave execution requires two separate locks (`_record_lock`, `_print_lock`) to
prevent JSONL corruption and interleaved console output.  Lock scopes are kept minimal to avoid
serialising unrelated work.

### 2-14. Pipeline workers separated from MCMC chains

Locations: [stages.py](../../src/arena/pipeline/stages.py),
[platform_setup.py](../../src/arena/lib/platform_setup.py),
[mcmc-parallelism-and-gpu-policy.md](../facts/performance/mcmc-parallelism-and-gpu-policy.md)

Rationale: `--workers` controls outer pipeline concurrency, not posterior sampling chains. Allowing
a 36-worker pipeline to expand every probabilistic stage into 36 MCMC chains made small-N models
slower without changing the decision. Stage 4/5 chain counts now default to 4 with a separate
host-device cap, and future convergence diagnostics can selectively raise chains only when R-hat,
ESS, or divergence checks require it.

### 2-15. Evidence synthesis as Stage 9, not weighted ensemble

Locations: [stages.py](../../src/arena/pipeline/stages.py),
[schema.py](../../src/arena/evidence/schema.py),
[claim_router.py](../../src/arena/evidence/claim_router.py),
[why-not-weighted-ensemble.md](why-not-weighted-ensemble.md)

Rationale: ARENA does not reduce coverage AUC, rank probability, location shift, posterior ratio,
and OpenSky capture ratio into one weighted score. Those metric families measure different targets
and carry different failure modes. Stage 9 normalizes model outputs into `EvidenceRow` records and
routes them into support, counter-evidence, caveats, validation targets, and next-data-needed fields
so contradictory evidence remains inspectable.

---

## 3. Artifacts (/output/payload)

### 3-1. SHA-256 hashing (per-file and per-bundle)

Locations: [hash_utils.py](../../src/arena/artifacts/hash_utils.py)

Rationale: Per-file SHA-256 detects corruption introduced during copy or transfer.  A deterministic
bundle-level hash provides a single identity for the entire artefact set.  This integrity chain
carries through to synthesis ingestion (see 5-2).

### 3-2. Schema validation (jsonschema) on all metadata

Locations: [schema.py](../../src/arena/artifacts/schema.py),
[integrity.py](../../src/arena/artifacts/integrity.py)

Rationale: Every metadata file (manifest, provenance, lineage, run metadata, integrity summary,
artefact index) is validated against a JSON schema at write time and again at verification time,
catching structural corruption before it silently propagates.

### 3-3. Integrity re-computation and cross-check

Locations: [integrity.py](../../src/arena/artifacts/integrity.py)

Rationale: `verify_artifact_bundle` re-computes the integrity summary from scratch, compares it
against the stored summary, and cross-checks provenance entries against manifest records and on-disk
hashes.  Discrepancies between any layer are surfaced as explicit errors.

### 3-4. Failure states are recorded, never hidden

Locations: [manifest.py](../../src/arena/artifacts/manifest.py),
[failure-taxonomy.md](../facts/failure-taxonomy.md)

Rationale: Artefacts that are missing, undersized, or failed to copy are written into the manifest
with statuses like `missing_required` or `copy_failed`.  Downstream consumers can inspect these
states rather than guessing why a file is absent.

### 3-5. Deterministic mode (fixed timestamps, fixed zip times)

Locations: [repro_stamp.py](../../src/arena/artifacts/repro_stamp.py),
[packaging.py](../../scripts/tools/artifacts/packaging.py)

Rationale: When deterministic mode is enabled, timestamps and zip entry times are pinned so that
identical inputs produce bit-identical outputs.  This supports CI diffing and reproducibility
experiments.

### 3-6. Candidate discovery guards (extension / size / exclusion patterns)

Locations: [discovery.py](../../src/arena/artifacts/discovery.py),
[policies.py](../../src/arena/artifacts/policies.py)

Rationale: Prevents oversized raw data files, transient outputs, and noise files from being swept
into artefact bundles.

### 3-7. Duplicate prevention (same-source dedup / name-collision guard)

Locations: [manifest.py](../../src/arena/artifacts/manifest.py)

Rationale: Duplicate source files are skipped and logged.  Destination name collisions are detected
before copy, preventing silent overwrites.

### 3-8. Verify and replay separated from production runs

Locations: [integrity.py](../../src/arena/artifacts/integrity.py), [cli.py](../../src/arena/cli.py)

Rationale: `arena artifacts verify` re-validates a bundle without re-running the pipeline, enabling
lightweight post-hoc audits at any time.

---

## 4. LLM Handoff (human + multiple LLMs)

### 4-1. CSVs are sent to LLMs — graphs are not

Locations: [policies.py](../../src/arena/artifacts/policies.py),
[ai-assisted-analysis.md](ai-assisted-analysis.md)

Rationale: Three factors drove this decision.

1. **Context efficiency** — roughly 55 files are submitted per session.
Images consume far more tokens than equivalent CSV text.
2. **Cross-model bias elimination** — each LLM interprets visual charts
differently.  Feeding raw numerical data removes graph-reading variance from the cross-model
verification process.
3. **Multi-dimensional interpretation** — humans can only perceive data
through two-dimensional projections (graphs).  LLMs can ingest the raw CSV and reason over
higher-dimensional relationships directly.  Rendering a graph compresses information; omitting it
preserves the full signal for the model.

---

## 5. Synthesis (ingest → triage → proposition review)

### 5-1. Path isolation + canonical DB enforcement

Locations: [paths.py](../../src/arena/synthesis/paths.py), [db.py](../../src/arena/synthesis/db.py),
[synthesis.md](../facts/synthesis.md)

Rationale: The synthesis database path is resolved from `ARENA_SYNTHESIS_DIR` and enforced as
canonical.  This prevents accidental cross-environment contamination (e.g., a dev DB leaking into
production analysis).

### 5-2. Idempotent ingestion (source_sha256 dedup + upsert)

Locations: [ingest.py](../../src/arena/synthesis/ingest.py)

Rationale: Each raw JSON file's SHA-256 is computed at ingest time and compared against previously
ingested hashes.  Duplicate files are skipped, preventing double-counting and preserving the
integrity chain that originates at artefact packaging (see 3-1).

### 5-3. Immutable originals (write-once + hash-tagged filenames)

Locations: [ingest_artifacts.py](../../src/arena/synthesis/ingest_artifacts.py),
[synthesis.md](../facts/synthesis.md)

Rationale: The original JSON and any repaired version are saved under filenames that include the
first 12 characters of their SHA-256.  Files are never overwritten.  This makes it possible to trace
any claim back to the exact input that produced it.

### 5-4. Repair policy safety (semantic repair requires explicit opt-in)

Locations: [ingest.py](../../src/arena/synthesis/ingest.py)

Rationale: Syntax repair (malformed JSON) and structural repair (missing fields) run automatically.
Semantic repair (value normalisation) is gated behind `--repair-semantic` because automated meaning
changes carry a higher risk of silent data corruption.

### 5-5. Append-only repair log (ingest_repair.jsonl)

Locations: [ingest_artifacts.py](../../src/arena/synthesis/ingest_artifacts.py)

Rationale: Every repair action — what was changed, what layer (syntax / structure / semantic), what
the before and after hashes were — is appended to an immutable log.  If a repaired record is later
questioned, the full repair history is available for re-audit.

### 5-6. Backward-compatible migration layer (view + INSTEAD OF triggers)

Locations: [db.py](../../src/arena/synthesis/db.py)

Rationale: When the `hypotheses` table was renamed to `claims`, a backward-compatible view with
`INSTEAD OF` triggers for insert, update, and delete was retained.  Existing queries and scripts
continue to work without modification during the migration period.

---

## Cross-cutting patterns

The decisions above are not isolated tactics; several design principles recur across subsystem
boundaries.

**Append-only logging** — Both the pipeline execution log (2-7) and the synthesis repair log (5-5)
are append-only.  Tampering and accidental truncation are structurally impossible.

**SHA-256 integrity chain** — Hashes originate at artefact packaging (3-1), are re-verified during
bundle verification (3-3), and are checked again at synthesis ingestion (5-2).  The chain is
unbroken from payload generation to database insertion.

**Visible failure** — Output contract checks (2-8), manifest failure statuses (3-4), and repair logs
(5-5) ensure that failures are recorded, never hidden. Downstream consumers always know the
provenance and health of their inputs.

**Idempotency** — Skip control (2-9), source-hash dedup (5-2), and write-once storage (5-3)
guarantee that re-running any stage produces no unintended side effects.
