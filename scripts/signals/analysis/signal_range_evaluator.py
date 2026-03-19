import os
import sys
from pathlib import Path
import pandas as pd


from arena.lib.paths import ADSB_SIGNAL_RANGE_SUMMARY
from arena.lib.phase_config import get_config

from arena.log import get_script_logger



log = get_script_logger(__name__)
def run_range_analysis():
    input_file = str(ADSB_SIGNAL_RANGE_SUMMARY)
    if not os.path.exists(input_file): return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    if df.empty:
        log.info("Signal strength data is empty. Skipping.")
        return
    df = df[df['date'] < df['date'].max()].sort_values('date')

    cfg = get_config()
    phases = cfg.signal_phases

    ranges = {
        '0-25km':   'sig_0_25',
        '25-50km':  'sig_25_50',
        '50-75km':  'sig_50_75',
        '75-100km': 'sig_75_100',
        '100-125km':'sig_100_125',
        '125-150km':'sig_125_150',
        '150-175km':'sig_150_175',
        '175-200km':'sig_175_200',
        '200-250km':'sig_200_250',
        '250-300km':'sig_250_300',
        '300-400km':'sig_300_400',
        '400km+':   'sig_400_9999',
    }

    phase_names = [p['name'] for p in phases]
    COL_W = 20

    header_parts = [f"{'Range':<12}"]
    for j, name in enumerate(phase_names):
        short = name[:COL_W - 1]
        if j == 0:
            header_parts.append(f"{short:<{COL_W}}")
        else:
            header_parts.append(f"{short:<{COL_W}}")
    table_w = 12 + 3 + (COL_W + 3) * len(phase_names)

    log.info("\n" + "=" * table_w)
    log.info(" | ".join(header_parts))
    log.info("-" * table_w)

    for label, col in ranges.items():
        if col not in df.columns:
            log.info(f"{label:<12} | ---")
            continue
        results = []
        for i, p in enumerate(phases):
            mask = (df['date'] >= pd.Timestamp(p['date']))
            if i + 1 < len(phases):
                mask &= (df['date'] < pd.Timestamp(phases[i+1]['date']))
            val = df[mask][col].mean()
            results.append(val)

        parts = [f"{label:<12}"]
        for j, val in enumerate(results):
            if pd.isna(val):
                parts.append(f"{'---':>{COL_W}}")
            elif j == 0:
                parts.append(f"{val:>8.2f} dB{'':>{COL_W - 11}}")
            else:
                prev = results[j - 1]
                if pd.notna(prev):
                    diff = val - prev
                    parts.append(f"{val:>7.2f} ({diff:>+5.1f}dB)")
                else:
                    parts.append(f"{val:>8.2f} dB{'':>{COL_W - 11}}")
        log.info(" | ".join(parts))

    log.info("=" * table_w)

if __name__ == "__main__":
    run_range_analysis()
