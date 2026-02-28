import os
import sys
from pathlib import Path
import pandas as pd


from arena.lib.paths import ADSB_SIGNAL_RANGE_SUMMARY
from arena.lib.phase_config import get_config


def run_range_analysis():
    input_file = str(ADSB_SIGNAL_RANGE_SUMMARY)
    if not os.path.exists(input_file): return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    if df.empty:
        print("Signal strength data is empty. Skipping.")
        return
    df = df[df['date'] < df['date'].max()].sort_values('date')

    cfg = get_config()
    phases = cfg.signal_phases

    ranges = {
        '0-50km': 'sig_0_50',
        '100-150km': 'sig_100_150',
        '150-175km': 'sig_150_175',
        '175-200km': 'sig_175_200',
        '200km+': 'sig_200_plus'
    }

    print("\n" + "="*95)
    phase_names = [p['name'] for p in phases]
    header_parts = [f"{'Range':<12}"]
    for j, name in enumerate(phase_names):
        if j == 0:
            header_parts.append(f"{name:<12}")
        else:
            header_parts.append(f"{name} (diff){'':>4}")
    print(" | ".join(header_parts))
    print("-" * 95)

    for label, col in ranges.items():
        if col not in df.columns:
            print(f"{label:<12} | --- (missing column: {col})")
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
                parts.append("---")
            elif j == 0:
                parts.append(f"{val:>8.2f} dB")
            else:
                prev = results[j-1]
                if pd.notna(prev):
                    diff = f"{val-prev:>+6.2f} dB"
                    parts.append(f"{val:>7.2f} ({diff})")
                else:
                    parts.append(f"{val:>7.2f}")
        print(" | ".join(parts))

    print("="*95)

if __name__ == "__main__":
    run_range_analysis()
