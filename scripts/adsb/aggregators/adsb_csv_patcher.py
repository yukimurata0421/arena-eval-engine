import os
import sys
from pathlib import Path
import pandas as pd


from arena.lib.paths import ADSB_DAILY_SUMMARY
from arena.lib.phase_config import get_config

DEFAULT_CSV_PATH = os.getenv("ADSB_DAILY_SUMMARY_PATH", str(ADSB_DAILY_SUMMARY))

def patch_adsb_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    cfg = get_config()

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    df['is_post_change'] = (df['date'] >= pd.Timestamp(cfg.post_change_date)).astype(int)

    df['hardware'] = cfg.default_hardware
    for date_str, hw_name in cfg.hardware_transitions:
        df.loc[df['date'] >= pd.Timestamp(date_str), 'hardware'] = hw_name

    if 'hnd_nrt_movements' in df.columns:
        df.loc[df['hnd_nrt_movements'] == 0, 'hnd_nrt_movements'] = 1000

    df.to_csv(csv_path, index=False)

    print("✅ CSV patching complete.")
    print(f"  post_change_date: {cfg.post_change_date}")
    print(f"  hardware transitions: {cfg.hardware_transitions}")

    if cfg.hardware_transitions:
        first_date = cfg.hardware_transitions[0][0]
        ts = pd.Timestamp(first_date)
        check = df[(df['date'] >= ts - pd.Timedelta(days=2)) & (df['date'] <= ts + pd.Timedelta(days=2))]
        print(f"\n--- Hardware transition check (around {first_date}) ---")
        print(check[['date', 'auc_n_used', 'is_post_change', 'hardware']])

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    patch_adsb_data(target_path)
