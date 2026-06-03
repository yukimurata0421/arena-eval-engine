# Synthesis Prompt Templates

These templates are optional operator-facing prompts for the AI-assisted synthesis workflow. They
are not loaded by `arena synthesis` runtime code directly.

Use them when you want a reproducible manual path from raw analysis text to schema-ready claim JSON.

## Templates

- [Phase 1: Claim Extraction](phase1_claim_extraction.md)
- [Phase 2: JSON Transform](phase2_json_transform.md)

## Intended Flow

1. Run AI analysis with the Phase 1 template to produce structured claim drafts.
2. Run the Phase 2 template to convert drafts into strict JSON records.
3. Validate and ingest with `arena synthesis ingest` or `arena synthesis run`.

