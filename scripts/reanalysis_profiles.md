# Reanalysis Profiles

このメモは、再評価時に混同しやすい `baseline_date` / `alt_baseline_date` / `analysis_start_date` の違いを整理するためのものです。

## 1) `baseline_date`

- 役割:
  - 主基準（メイン比較の基準日）を指す概念。
- このリポジトリでの実態:
  - `phases.txt` に `baseline_date` というキーはありません。
  - 実運用では「主解析の基準」は `post_change_date`（現在は `2026-01-14`）として扱います。
- 使いどころ:
  - RTL-SDR -> Airspy の主解析境界を固定した比較。

## 2) `alt_baseline_date`

- 役割:
  - 第二基準（代替ベースライン）の開始日。
- 定義場所:
  - `scripts/config/phases.txt` の `[settings]`
- 現在値:
  - `2026-01-29`
- 使いどころ:
  - Airspy 導入直後の遷移期間を避け、Airspy 安定運用期を基準に cable/adapter の微差を比較する。
  - `adsb_phase_evaluator_v3.py` の Section 2 で参照される。

## 3) `analysis_start_date`

- 役割:
  - 解析対象データの「読み込み開始日」フィルタ。
- 指定方法:
  - CLI: `--analysis-start-date YYYY-MM-DD`
  - 環境変数: `ARENA_ANALYSIS_START_DATE`
- 使いどころ:
  - 例: `2026-01-14` を指定して Airspy 後限定の再評価を実行する。
- 注意:
  - これは基準日そのものではなく、対象データ範囲を切るための条件。

## 使い分けの要点

- `baseline_date`（概念）:
  - 「何と比較するか」の主基準。
- `alt_baseline_date`:
  - 「別の基準でも比較する」ための第二基準。
- `analysis_start_date`:
  - 「どこから先のデータを使うか」を決めるフィルタ。

## 例（Airspy後限定 + 代替基準あり）

```powershell
python -m arena.cli run ^
  --analysis-start-date 2026-01-14 ^
  --phase-config scripts/config/phases.txt
```

