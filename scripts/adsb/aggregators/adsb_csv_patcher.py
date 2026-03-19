import os
import sys
from pathlib import Path
import pandas as pd


from arena.lib.paths import ADSB_DAILY_SUMMARY
from arena.lib.phase_config import get_config

from arena.log import get_script_logger


log = get_script_logger(__name__)
DEFAULT_CSV_PATH = os.getenv("ADSB_DAILY_SUMMARY_PATH", str(ADSB_DAILY_SUMMARY))
TRAFFIC_ZERO_POLICY = os.getenv("ADSB_TRAFFIC_ZERO_POLICY", "impute").strip().lower()


def _repair_traffic_values(df: pd.DataFrame, policy: str) -> tuple[int, int]:
    """
    Repair invalid traffic values.
    policy:
      - impute: replace <=0 by median of same weekday (fallback: global median)
      - skip: set <=0 to NaN
    Returns (repaired_count, skipped_count).
    """
    if "hnd_nrt_movements" not in df.columns:
        return 0, 0

    s = pd.to_numeric(df["hnd_nrt_movements"], errors="coerce")
    bad = s.notna() & (s <= 0)
    if not bad.any():
        return 0, 0

    repaired = 0
    skipped = 0

    if policy == "skip":
        s.loc[bad] = pd.NA
        skipped = int(bad.sum())
    else:
        # Weekday-aware imputation keeps seasonality better than a constant.
        dow = df["date"].dt.dayofweek
        valid = s.where(s > 0)
        global_med = valid.median()
        for idx in s.index[bad]:
            wd = dow.loc[idx]
            wd_med = valid[dow == wd].median()
            fill = wd_med if pd.notna(wd_med) else global_med
            if pd.notna(fill):
                s.loc[idx] = round(float(fill), 1)
                repaired += 1
            else:
                s.loc[idx] = pd.NA
                skipped += 1

    df["hnd_nrt_movements"] = s
    if "traffic_missing" in df.columns:
        df["traffic_missing"] = df["hnd_nrt_movements"].isna().astype(int)
    return repaired, skipped

def patch_adsb_data(csv_path):
    if not os.path.exists(csv_path):
        log.info(f"エラー: {csv_path} が見つかりません。")
        return

    cfg = get_config()

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    df['is_post_change'] = (df['date'] >= pd.Timestamp(cfg.post_change_date)).astype(int)

    df['hardware'] = cfg.default_hardware
    for date_str, hw_name in cfg.hardware_transitions:
        df.loc[df['date'] >= pd.Timestamp(date_str), 'hardware'] = hw_name

    repaired, skipped = _repair_traffic_values(df, TRAFFIC_ZERO_POLICY)

    df.to_csv(csv_path, index=False)

    log.info("CSV patching complete.")
    log.info(f"  post_change_date: {cfg.post_change_date}")
    log.info(f"  hardware transitions: {cfg.hardware_transitions}")
    log.info(f"  traffic zero policy: {TRAFFIC_ZERO_POLICY} (repaired={repaired}, skipped={skipped})")

    if cfg.hardware_transitions:
        first_date = cfg.hardware_transitions[0][0]
        ts = pd.Timestamp(first_date)
        check = df[(df['date'] >= ts - pd.Timedelta(days=2)) & (df['date'] <= ts + pd.Timedelta(days=2))]
        log.info(f"\n--- Hardware transition check (around {first_date}) ---")
        log.info(check[['date', 'auc_n_used', 'is_post_change', 'hardware']])

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    patch_adsb_data(target_path)
