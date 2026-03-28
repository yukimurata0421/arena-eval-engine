# Synthesis Sample Data

This folder provides a public, non-sensitive sample dataset for `arena synthesis` flows.

## Layout

- `raw/claude/20260327.json`
- `raw/gemini/20260327.json`
- `raw/gpt/20260327.json`
- `raw/grok/20260327.json`

Each file contains at least one schema-valid claim record suitable for:

- ingest (`arena synthesis ingest`)
- enrichment (`arena synthesis enrich`)
- baseline clustering/report/suggest
- proposition-layer/triage/review queue export

## Quick Validation

Run from repo root:

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

This keeps generated artifacts in `./tmp/synthesis-smoke/` instead of `workspace/synthesis/`.
