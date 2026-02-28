# Architecture (High-Level)

This is a minimal, CLI-first evaluation pipeline. All stages are orchestrated by `arena run`
and emit artifacts into `output/` (ignored in VCS by default).

```mermaid
graph TD
  A[Stage 1: Aggregation] --> B[Stage 2: Spatial / Visual]
  B --> C[Stage 3: Statistics (CPU)]
  C --> D[Stage 4: Phase Evaluation]
  D --> E[Stage 5: Bayesian & Change Points]
  E --> F[Stage 6: Final Reports]
  F --> G[Stage 7: PLAO]
  G --> H[Stage 8: OpenSky Comparison]
```
