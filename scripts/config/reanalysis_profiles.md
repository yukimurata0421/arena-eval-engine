# Reanalysis Profiles

This note documents safe reanalysis patterns without replacing production defaults.

## 1) Main phase definition (adopted)

- File: `scripts/config/phases.txt`
- Includes RTL split:
  - `2026-01-06` RTL-SDR Default
  - `2026-01-10` RTL-SDR Gain Tuned
  - `2026-01-14` Airspy Baseline

## 2) Airspy-post-only reanalysis (adopted helper)

Use date-window override to isolate post-Airspy period:

```powershell
python -m arena.cli run ^
  --analysis-start-date 2026-01-14 ^
  --phase-config scripts/config/phases_v3_airspy_baseline.txt
```

Optional end date:

```powershell
python -m arena.cli run ^
  --analysis-start-date 2026-01-14 ^
  --analysis-end-date 2026-03-12 ^
  --phase-config scripts/config/phases_v3_airspy_baseline.txt
```

## 3) Distance bins fine-grain comparison (experiment only)

- Production keeps: `scripts/config/settings.toml`
- Experimental bins file:
  - `scripts/config/settings_experimental_distance_bins.toml`
  - bins: `[0, 50, 100, 150, 200, 250, 300, 9999]`

Run with experimental bins:

```powershell
python -m arena.cli run ^
  --settings scripts/config/settings_experimental_distance_bins.toml
```

Compare against production settings results before any promotion.
