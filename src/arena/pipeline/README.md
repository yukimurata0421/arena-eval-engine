# Pipeline module map

## Read order
1. entrypoint.py
2. stages.py
3. runner.py
4. decision.py
5. backend.py
6. record_io.py
7. error_policy.py

## Responsibilities
- entrypoint: collect runtime config and start pipeline execution
- stages: define Step objects and stage ordering
- runner: orchestrate step execution and state transitions
- decision: pure-ish execution/skip/soft-fail decisions
- backend: execution environment abstraction
- record_io: append-only run logging
- error_policy: error classification and recommended actions

### Standard Artifact Note
- Change-point analysis is part of the standard pipeline Stage 5.
- Required artifacts: `output/change_point/change_point_report.txt`, `output/change_point/multi_change_points_report.txt`
