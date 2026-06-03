# Synthesis Workflow

## Purpose

`arena synthesis` is the claim-ingest and proposition-review layer for AI-assisted analysis outputs.
It is designed to run in a repository-local or fully overridden path configuration without depending
on sibling clones.

## Path Isolation

Default workspace is `workspace/synthesis/`, but every operational path can be overridden:

- `--db`
- `--path` (raw input root)
- `--enriched-dir`
- `--review-dir`
- `--raw-original-dir`
- `--raw-repaired-dir`
- `--repair-log-dir`

You can also set `ARENA_SYNTHESIS_DIR` to relocate the default workspace root.

## End-to-End Sample Run

```bash
python -m arena.cli synthesis run \
  --path sample_data/synthesis/raw \
  --db ./tmp/synthesis-smoke/synthesis.sqlite3 \
  --enriched-dir ./tmp/synthesis-smoke/enriched \
  --review-dir ./tmp/synthesis-smoke/review \
  --raw-original-dir ./tmp/synthesis-smoke/raw_original \
  --raw-repaired-dir ./tmp/synthesis-smoke/raw_repaired \
  --repair-log-dir ./tmp/synthesis-smoke/repair_logs
```

## Real Data Validation Flow

For real data, keep private raw files outside this repository and mount or point to them explicitly.

Example:

```bash
python -m arena.cli synthesis run \
  --path /secure/llm-claims/raw \
  --db /secure/arena-synthesis/synthesis.sqlite3 \
  --enriched-dir /secure/arena-synthesis/enriched \
  --review-dir /secure/arena-synthesis/review \
  --raw-original-dir /secure/arena-synthesis/raw_original \
  --raw-repaired-dir /secure/arena-synthesis/raw_repaired \
  --repair-log-dir /secure/arena-synthesis/repair_logs
```

## Verification Checklist

- `status=SUCCESS outcome=pipeline_completed`
- triage output JSONL exists under `review/triage/`
- queue export exists under `review/queue/`
- DB file contains `propositions`, `claim_propositions`, and triage tables

## Prompt Templates (Optional Manual Flow)

For manual AI-assisted claim drafting, use:

- `docs/prompt-templates/phase1_claim_extraction.md`
- `docs/prompt-templates/phase2_json_transform.md`

These templates are operator-facing guidance and are not loaded by runtime code.

## Operational Notes

- `--repair` is enabled by default in `synthesis run`.
- `--repair-semantic` requires `--repair`.
- Raw input files are never overwritten; repaired and original snapshots are persisted separately.
