# Failure Taxonomy

ARENA classifies failures into two categories: pipeline execution failures and artifact integrity failures.
Both categories are designed to be visible, structured, and non-fatal.

---

## Pipeline Failure States

Every pipeline step produces a status recorded in `pipeline_runs.jsonl`.

| Status | Meaning | When it occurs |
|---|---|---|
| `OK` | Step completed and all expected outputs are valid | Normal execution |
| `DRY` | Step was not executed (dry-run mode) | `--dry-run` flag |
| `SKIP` | Step was skipped because valid outputs already exist | `--skip-existing` with fresh outputs |
| `WARN` | Step completed but with degraded behavior | Rate limits, log rotation, stale secondary outputs |
| `FAIL` | Step execution returned a non-zero exit code | Script error, import failure, data issue |
| `FAIL_OUTPUT` | Step returned success but expected outputs are missing, empty, or stale | Silent failure — return code 0 but broken output |
| `TIMEOUT` | Step exceeded its time limit | Hung process or unexpectedly large dataset |
| `NOT_FOUND` | Script file does not exist at the expected path | Misconfigured scripts root or missing file |
| `ERROR` | Unexpected exception during step orchestration | Internal pipeline error |

### Error Codes

The error policy module (`src/arena/pipeline/error_policy.py`) assigns structured error codes
using the pattern `S{stage}-{step}-{code}`:

| Code pattern | Category | Examples |
|---|---|---|
| `*-W01` | Log rotation activity detected | Non-critical, informational |
| `*-W02` | API rate limit reached | OpenSky fetch throttled |
| `*-W03` | Stale secondary output | Output exists but is outdated |
| `*-W99` | Unclassified warning | Catch-all for new warning types |
| `*-E10` | General execution failure | Script crashed |
| `*-E11` | Encoding error | UnicodeEncodeError in output |
| `*-E12` | Rate limit failure | API rate limit caused hard failure |
| `*-E20` | Output validation failure (general) | Expected output missing or invalid |
| `*-E21` | Stale output detected | Output exists but older than inputs |
| `*-E22` | Output too small | File exists but below minimum size |
| `*-E23` | Empty directory | Expected output directory is empty |
| `*-E31` | Timeout | Step exceeded time limit |
| `*-E32` | Script not found | File missing at expected path |
| `*-E33` | Orchestration error | Internal exception |
| `*-E99` | Unclassified error | Catch-all for new error types |

### Dependency-Aware Skip Logic

`--skip-existing` does not just check if output files exist.
`decision.py` compares `latest_dependency_mtime` (the newest modification time
among a step's declared `depends_on_outputs`) against the step's own output timestamps.
If any dependency is newer than the existing output, the step re-runs.

---

## Artifact Integrity Failures

The artifact verification system (`arena artifacts verify`) checks exported bundles
without re-running the pipeline. Failures are reported as structured results, never exceptions.

| Failure type | What it means | What `verify` checks |
|---|---|---|
| `hash_mismatch` | An exported file's SHA256 does not match the recorded hash | File content changed after export |
| `schema_invalid` | A JSON output does not pass its JSON Schema | Manifest, provenance, run metadata, or index is malformed |
| `missing_artifact` | An expected file is absent from the bundle | File was deleted or not copied during export |
| `provenance_inconsistency` | Provenance source paths disagree with the manifest | Source tracking is broken |
| `malformed_manifest` | CSV manifest or pack manifest cannot be parsed | Structural corruption |
| `bundle_hash_mismatch` | Recomputed bundle-level SHA256 differs from `artifact_index.json` | Any bundle content changed since export |

### Verification Guarantees

- `verify` and `replay` never crash on malformed input. They return structured error results.
- Schema validation failures are reported per-file, not as a single pass/fail.
- Missing files are enumerated individually.
- Audit logs (`pipeline_runs.jsonl`) are never modified by verification.

### Failure Visibility States in Manifests

The artifact subsystem preserves the following states in manifests and candidate status outputs:

- `missing_required` — a required artifact was not found during discovery
- `missing_recommended` — a recommended artifact was not found
- `excluded_by_rule` — an artifact matched an exclusion glob pattern
- `duplicate_source_skipped` — the same source file was already selected under a different target
- `copy_failed` — file copy to the export directory failed

These states are part of the audit trail. They are never smoothed away or hidden.
A bundle that contains `missing_required` entries still exports successfully —
the failure is visible in the manifest rather than causing a silent incomplete export.

---

## Recommended Actions

The error policy module generates actionable recovery suggestions for known failure patterns.

Example: when OpenSky API rate limits are hit (`S1-06-W02` or `S1-06-E12`),
the recommended action includes the specific environment variables and commands
to retry the fetch and refresh the comparison stage.

```
OpenSky API rate limit reached. Wait 10-15 minutes and rerun only the OpenSky fetch:
  $env:OPENSKY_REFRESH_DAYS='1'; arena fetch-opensky
  After the fetch completes, refresh the comparison with: arena run --only 8
```
