# ARENA Synthesis Prompt: Phase 1 Claim Extraction

Analyze ARENA statistical outputs and extract **re-validatable claims**.

## Goal

Produce claims that can later be stored in a DB and reused for:

- re-validation and audit
- search and discovery
- proposition-layer integration

Preserve complete metadata. Do not drop supporting context.

## Role

Read the provided analysis artifacts and extract claims. Do not produce a loose summary. Produce
structured, traceable claims.

## Claim Types (`claim_type`)

- `supported`
- `negative`
- `unknown`
- `future`

## Critical Rules

- One claim = one proposition.
- Avoid vague wording.
- Include numeric values whenever possible.
- Evidence files are required.
- Metadata must not be missing.
- Do not invent missing information.

## Priority Hint (`priority_hint`)

- `high`
- `medium`
- `low`

## Exploration Axes (`exploration_axes`)

- Up to 3 axes.
- 1 primary axis + 0-2 secondary axes.
- Include only evidence-backed axes.
- Use `null` if unknown.

## Baseline Candidate

```yaml
baseline_candidate:
  label: short identifier
  type: parameter | hardware | temporal | metric | null
  confidence: high | medium | low | null
  reason: short evidence-based reason
```

## Output Format (per claim)

```text
[claim_type]
claim:
basis_summary:
evidence_files:
evidence_refs:
required_files:
metrics_used:
evidence_level:
limitation_or_counterpoint:
next_data_needed:
priority_hint:
exploration_axes:
baseline_candidate:
raw_text:
created_at:
```

