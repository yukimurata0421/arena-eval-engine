# Documentation Index

This index is the entry point for detailed ARENA documentation.

## Core Buckets

- [Facts](facts/README.md): operational structure, runtime behavior, and reproducibility contracts
- [Principles](principles/README.md): design intent, trust boundaries, and methodological rationale

## Evidence Synthesis

- [Why ARENA Does Not Use Weighted Ensemble](principles/why-not-weighted-ensemble.md): rationale for
  preserving metric-family disagreement instead of averaging it away
- [Evidence Synthesis Stage Contract](principles/evidence-synthesis-stage.md): Stage 9
  responsibilities, outputs, and claim-router contract
- [Model Disagreement Report Example](facts/examples/model_disagreement_report_example.md):
  reviewer-readable example artifact

## Reproducibility and Release Validation

- [Reproducibility](facts/reproducibility.md): public smoke reproducibility contract
- [Real-Data Docker Validation](facts/real-data-smoke.md): opt-in private real-data checks
- [MCMC Parallelism and GPU Policy](facts/performance/mcmc-parallelism-and-gpu-policy.md): current
  Xeon/GTX1070 hardware note, worker-chain separation, and CPU/GPU policy
- [Artifact Design](principles/artifact-design.md): why artifact controls exist
- [Evolution v0.3.1 to v0.4.0](evolution/v0.3.1-to-v0.4.0.md): release-layer rationale for Stage 9
  evidence synthesis

## Design Decisions

- [Engineering Decisions](principles/engineering-decisions.md): 33 design decisions organised by
  data-flow stage

## Synthesis

- [Synthesis Workflow](facts/synthesis.md): ingest/enrich/cluster/proposition/triage/review flow
- [Synthesis Prompt Templates](prompt-templates/README.md): optional manual prompt set for claim
  extraction and JSON transform
- [AI-Assisted Analysis](principles/ai-assisted-analysis.md): trust boundaries and human review
  model

## Analysis Method Docs

- [AEME](principles/aeme.md)
- [Failure Taxonomy](facts/failure-taxonomy.md)
- [Sample Outputs](facts/sample_outputs.md)
- [Statistical Assumptions and Limitations](principles/statistical-assumptions-and-limitations.md)

## Release History

- [Changelog](../CHANGELOG.md)
- [Evolution v0.2.9 to v0.3.0](evolution/v0.2.9-to-v0.3.0.md)
