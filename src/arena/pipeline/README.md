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

## <JP>
- entrypoint: 実行開始点。設定を集めて runner に渡す
- stages: 実行対象 Step の定義
- runner: Step を順番に制御して実行する司令塔
- decision: skip / stale / soft-fail の判定ロジック
- backend: native / WSL など実行環境差分の吸収
- record_io: 実行記録の保存
- error_policy: エラーコード化と推奨アクション生成

### Standard Artifact Note
- Change-point 解析は標準パイプライン Stage 5 に含まれます。
- 必須成果物: `output/change_point/change_point_report.txt`, `output/change_point/multi_change_points_report.txt`
