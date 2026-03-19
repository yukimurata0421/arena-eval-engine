# Reanalysis Profiles

This memo is to clarify the differences between `baseline_date` / `alt_baseline_date` / `analysis_start_date`, which can be easily confused during re-evaluation.

## 1) `baseline_date`

- role:
  - A concept that refers to the main reference (base date of the main comparison).
- Actual situation in this repository:
  - There is no key called `baseline_date` in `phases.txt`.
  - In actual operation, the "main analysis standard" is treated as `post_change_date` (currently `2026-01-14`).
- When to use:
  - RTL-SDR -> Comparison with fixed main analysis boundary of Airspy.

## 2) `alt_baseline_date`

- role:
  - Start date of the second standard (alternative baseline).
- Definition location:
  - `[settings]` in `scripts/config/phases.txt`
- Current value:
  - `2026-01-29`
- When to use:
  - Avoid the transition period immediately after Airspy introduction, and compare the slight differences in cable/adapter based on the stable operation period of Airspy.
  - Referenced in Section 2 of `adsb_phase_evaluator_v3.py`.

## 3) `analysis_start_date`

- role:
  - "Loading start date" filter for analysis target data.
- How to specify:
  - CLI: `--analysis-start-date YYYY-MM-DD`
  - Environment variable: `ARENA_ANALYSIS_START_DATE`
- When to use:
  - Example: Specify `2026-01-14` to perform re-evaluation only after Airspy.
- Note:
  - This is not the base date itself, but a condition for cutting the target data range.

## Key points for proper usage

- `baseline_date` (concept):
  - The main criterion of "what to compare with".
- `alt_baseline_date`:
  - A second standard for ``comparing with other standards.''
- `analysis_start_date`:
  - A filter that determines "from where to use the data."

## Example (only after Airspy + with alternative criteria)

```powershell
python -m arena.cli run ^
  --analysis-start-date 2026-01-14 ^
  --phase-config scripts/config/phases.txt
```

