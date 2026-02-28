import os
import sys
from pathlib import Path
import pandas as pd


from arena.lib.paths import ADSB_SIGNAL_DAILY_SUMMARY
from arena.lib.phase_config import get_config
import numpy as np


def run_signal_analysis():
    input_file = str(ADSB_SIGNAL_DAILY_SUMMARY)
    if not os.path.exists(input_file):
        print("CSV not found. Run the aggregator first.")
        return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df = df[df['date'] < df['date'].max()].copy()

    cfg = get_config()
    phases = cfg.signal_phases  # [{"date": "...", "name": "..."}, ...]

    print("\n" + "="*75)
    print(f"{'Phase':<20} | {'150-175km Avg Sig':<20} | {'Delta'}")
    print("-" * 75)

    prev_avg = None
    for i, p in enumerate(phases):
        mask = (df['date'] >= pd.Timestamp(p['date']))
        if i + 1 < len(phases):
            mask &= (df['date'] < pd.Timestamp(phases[i+1]['date']))
        
        sig_data = df[mask]['sig_150_175'].dropna()
        if not sig_data.empty:
            avg_db = sig_data.mean()
            diff_str = "---" if prev_avg is None else f"{avg_db - prev_avg:>+6.2f} dB"
            
            print(f"{p['name']:<20} | {avg_db:>15.2f} dBFS | {diff_str}")
            prev_avg = avg_db

    print("="*75)

if __name__ == "__main__":
    run_signal_analysis()
