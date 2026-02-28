from pathlib import Path

import numpy as np
import pandas as pd

from arena.lib.config import get_quality_thresholds
from arena.lib.paths import ADSB_DAILY_SUMMARY_V2
from arena.lib.phase_config import get_config


def load_summary(
    path: str | None = None,
    min_auc: int | None = None,
    min_minutes: int | None = None,
    require_proxy: bool = True,
    post_date: str | pd.Timestamp | None = None,
    default_post_date: str | None = None,
):
    if default_post_date is None:
        default_post_date = get_config().intervention_date
    """
    Load adsb_daily_summary_v2.csv and apply common preprocessing:
      - local_traffic_proxy numeric + median imputation
      - log_traffic column
      - post flag (from post/is_post_change or by date)
      - auc filter and minutes_covered filter
    """
    csv_path = Path(path) if path else Path(ADSB_DAILY_SUMMARY_V2)
    if not csv_path.exists():
        print(f"  Error: {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if min_auc is None or min_minutes is None:
        cfg_min_auc, cfg_min_minutes = get_quality_thresholds()
        if min_auc is None:
            min_auc = cfg_min_auc
        if min_minutes is None:
            min_minutes = cfg_min_minutes

    # Filters
    if min_auc is not None:
        df = df[df["auc_n_used"] > min_auc].copy()
    if min_minutes is not None and "minutes_covered" in df.columns:
        df = df[df["minutes_covered"] >= min_minutes].copy()

    # local_traffic_proxy
    if require_proxy and "local_traffic_proxy" not in df.columns:
        print("  local_traffic_proxy not found. Check adsb_daily_summary_v2.csv.")
        return None
    if "local_traffic_proxy" in df.columns:
        df["local_traffic_proxy"] = pd.to_numeric(df["local_traffic_proxy"], errors="coerce")
        med_proxy = df["local_traffic_proxy"].median()
        fill_val = med_proxy if pd.notna(med_proxy) else 1
        df["local_traffic_proxy"] = df["local_traffic_proxy"].fillna(fill_val)
        df["local_traffic_proxy"] = df["local_traffic_proxy"].replace(0, fill_val)
        df["log_traffic"] = np.log(df["local_traffic_proxy"])

    # post flag
    if post_date is not None:
        cutoff = pd.Timestamp(post_date)
        df["post"] = (df["date"] >= cutoff).astype(int)
    elif "post" in df.columns:
        df["post"] = df["post"].astype(int)
    elif "is_post_change" in df.columns:
        df["post"] = df["is_post_change"].astype(int)
    else:
        cutoff = pd.Timestamp(default_post_date)
        df["post"] = (df["date"] >= cutoff).astype(int)

    return df


def check_proxy_endogeneity(df: pd.DataFrame):
    """
    Quick endogeneity check: correlation between post and local_traffic_proxy.
    """
    if df is None:
        return None
    if "post" not in df.columns or "local_traffic_proxy" not in df.columns:
        print("  check_proxy_endogeneity: required columns are missing.")
        return None
    corr = np.corrcoef(df["post"].astype(float), df["local_traffic_proxy"].astype(float))[0, 1]
    print(f"  proxy endogeneity (corr post vs local_traffic_proxy): {corr:.4f}")
    return corr
