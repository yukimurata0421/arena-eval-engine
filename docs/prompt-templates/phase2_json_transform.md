# ARENA Synthesis Prompt: Phase 2 JSON Transform

Convert the Phase 1 structured claim output into a DB-ingestable JSON array.

## Goal

Transform claims without losing information. Keep exploration metadata (`exploration_axes`,
`baseline_candidate`) intact.

## Critical Rules

- No information reduction.
- Do not delete metadata.
- Output JSON only.

## Required Schema Keys

```json
{
  "id": "CLAIM-001",
  "claim": "...",
  "claim_type": "supported | negative | unknown | future",
  "basis_summary": "...",
  "evidence_files": ["file1.csv", "file2.txt"],
  "evidence_refs": ["ref string 1", "ref string 2"],
  "required_files": [
    {
      "file_name": "...",
      "priority": "A | B | C",
      "reason": "...",
      "required_for": "..."
    }
  ],
  "metrics_used": [
    {
      "file": "target file",
      "metric": "metric name",
      "value": 123.45,
      "context": {
        "series_name": "series name or null",
        "condition": "condition or null"
      }
    }
  ],
  "evidence_level": "direct | inferred",
  "limitation_or_counterpoint": ["...", "..."],
  "next_data_needed": ["..."] or null,
  "priority_hint": "high | medium | low",
  "exploration_axes": ["axis1", "axis2"] or null,
  "baseline_candidate": {
    "label": "string or null",
    "type": "parameter | hardware | temporal | metric | null",
    "confidence": "high | medium | low | null",
    "reason": "string or null"
  } or null,
  "raw_text": "..." or null,
  "created_at": "ISO string" or null
}
```

## Value Constraints (Strict)

- `claim_type`: `supported`, `negative`, `unknown`, `future`
- `priority_hint`: `high`, `medium`, `low`
- `exploration_axes`: array (max 3 elements) or `null`
- `baseline_candidate.confidence`: `high`, `medium`, `low`, `null`
- `baseline_candidate.type`: `parameter`, `hardware`, `temporal`, `metric`, `null`
- `required_files[].priority`: `A`, `B`, `C`
- `evidence_level`: `direct`, `inferred`
- `limitation_or_counterpoint`: array of strings, never scalar string (empty array allowed)

## Prohibitions

- Metadata deletion
- Priority tampering
- Guessed field completion without evidence
- Forced baseline generation
- Over-generation of exploration axes
- Any non-JSON output

## Output

Return a JSON array only, starting with `[` and ending with `]`. Do not add explanations.

