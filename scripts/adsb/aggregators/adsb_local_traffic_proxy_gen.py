"""
adsb_local_traffic_proxy_gen.py (improved)

Generate daily local traffic proxy from receiver pos data and merge it into
adsb_daily_summary.csv to produce v2.csv.

Improvements:
  1. Parallelization via ProcessPoolExecutor (i7-8700K 6-core)
  2. Compute 25km/50km/100km counts for endogeneity checks
  3. Run a simple endogeneity check during merge
  4. Integrates with paths.py
"""

import os
import sys
import json
import glob
import gzip
import math
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None


from arena.lib.config import get_site_latlon
from arena.lib.paths import ADSB_DAILY_SUMMARY, OUTPUT_DIR, DATA_DIR, ensure_dir

POS_DIR = str(DATA_DIR / "plao_pos")
FINAL_OUTPUT = str(OUTPUT_DIR / "adsb_daily_summary_v2.csv")
SITE_LAT, SITE_LON = get_site_latlon()

RADII_KM = [25.0, 50.0, 100.0]
PRIMARY_RADIUS = 50.0

MAX_WORKERS = min(6, os.cpu_count() or 4)


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance (km)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def open_file(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def process_one_file(pos_path: str) -> dict | None:
    """Compute unique aircraft count from a single-day pos file (worker)."""
    base = os.path.basename(pos_path)
    date_str = "".join(filter(str.isdigit, base))[:8]
    try:
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
    except Exception:
        return None

    hex_sets = {r: set() for r in RADII_KM}
    total_lines = 0

    try:
        with open_file(pos_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    lat, lon, hex_id = d.get('lat'), d.get('lon'), d.get('hex')
                    if lat is None or lon is None or hex_id is None:
                        continue
                    dist = haversine_km(SITE_LAT, SITE_LON, lat, lon)
                    for r in RADII_KM:
                        if dist <= r:
                            hex_sets[r].add(hex_id)
                    total_lines += 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except Exception:
        return None

    if total_lines < 100:
        return None

    result = {'date': str(target_date), 'pos_records': total_lines}
    for r in RADII_KM:
        col = f"unique_hex_{int(r)}km"
        result[col] = len(hex_sets[r])

    result['local_traffic_proxy'] = result[f"unique_hex_{int(PRIMARY_RADIUS)}km"]

    return result


def run_endogeneity_check(df: pd.DataFrame) -> dict:
    """Simple endogeneity check for local_traffic_proxy."""
    if sp_stats is None:
        return {'error': 'scipy not available'}
    if 'local_traffic_proxy' not in df.columns:
        return {'error': 'no proxy column'}

    if 'is_post_change' in df.columns:
        post_col = 'is_post_change'
    elif 'post' in df.columns:
        post_col = 'post'
    else:
        return {'error': 'no post flag'}

    pre  = df.loc[df[post_col] == 0, 'local_traffic_proxy'].dropna().values
    post = df.loc[df[post_col] == 1, 'local_traffic_proxy'].dropna().values

    if len(pre) < 3 or len(post) < 3:
        return {'error': f'insufficient data (pre={len(pre)}, post={len(post)})'}

    u, p = sp_stats.mannwhitneyu(pre, post, alternative='two-sided')
    pre_mean  = float(np.mean(pre))
    post_mean = float(np.mean(post))

    return {
        'pre_mean':  round(pre_mean, 1),
        'post_mean': round(post_mean, 1),
        'change_pct': round(((post_mean / pre_mean) - 1) * 100, 1),
        'p_value':   round(p, 6),
        'is_endogenous': p < 0.05,
    }


def generate_traffic_proxy():
    ensure_dir(OUTPUT_DIR)

    pos_files = sorted(glob.glob(os.path.join(POS_DIR, "pos_*.jsonl*")))
    if not pos_files:
        print(f"  Error: pos file not found: {POS_DIR}")
        return

    print(f">>> Computing local traffic proxy")
    print(f"    Target: {len(pos_files)} files")
    print(f"    Radius: {RADII_KM} km")
    print(f"    Parallel: {MAX_WORKERS} workers")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, pf): pf for pf in pos_files}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                continue
            print(f"  [{i + 1}/{len(pos_files)}] done", end="\r")

    print(f"\n  Valid data: {len(results)} days")

    if not results:
        print("  Error: No valid data.")
        return

    df_proxy = pd.DataFrame(results).sort_values('date').reset_index(drop=True)

    for r in RADII_KM:
        col = f"unique_hex_{int(r)}km"
        print(f"    {col}: mean={df_proxy[col].mean():.1f}, "
              f"range=[{df_proxy[col].min()}, {df_proxy[col].max()}]")

    summary_path = str(ADSB_DAILY_SUMMARY)
    if not os.path.exists(summary_path):
        proxy_csv = str(OUTPUT_DIR / "local_traffic_proxy.csv")
        df_proxy.to_csv(proxy_csv, index=False)
        print(f"  Saved proxy only: {proxy_csv}")
        df_proxy.to_csv(FINAL_OUTPUT, index=False)
        print(f"  Saved v2 with proxy only: {FINAL_OUTPUT}")
        return

    df_summary = pd.read_csv(summary_path)
    df_summary['date'] = df_summary['date'].astype(str).str.strip()
    df_proxy['date']   = df_proxy['date'].astype(str).str.strip()

    n_before = len(df_summary)

    proxy_cols = [c for c in df_summary.columns
                  if c.startswith('unique_hex_') or c in ('local_traffic_proxy', 'pos_records')]
    if proxy_cols:
        df_summary = df_summary.drop(columns=proxy_cols)

    df_merged = pd.merge(df_summary, df_proxy, on='date', how='left')

    n_matched = df_merged['local_traffic_proxy'].notna().sum()
    n_missing = df_merged['local_traffic_proxy'].isna().sum()
    print(f"\n  Merge result: {n_matched} days matched, {n_missing} days missing")

    if n_missing > 0:
        missing_dates = df_merged.loc[
            df_merged['local_traffic_proxy'].isna(), 'date'
        ].tolist()
        print(f"  Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}")

    df_merged.to_csv(FINAL_OUTPUT, index=False)
    print(f"\n  Saved: {FINAL_OUTPUT}")

    endo = run_endogeneity_check(df_merged)
    print(f"\n  --- Endogeneity check ---")
    if 'error' in endo:
        print(f"  Skip: {endo['error']}")
    else:
        print(f"  Pre  mean: {endo['pre_mean']}")
        print(f"  Post mean: {endo['post_mean']}")
        print(f"  Change: {endo['change_pct']:+.1f}%  (P = {endo['p_value']:.6f})")
        if endo['is_endogenous']:
            print(f"  ⚠ Significant difference: unique aircraft within 50km increased after hardware change.")
            print(f"    Consider using 25km values as an alternative proxy.")
        else:
            print(f"  ✓ Exogeneity confirmed: proxy can be used as a confounder control.")


if __name__ == "__main__":
    generate_traffic_proxy()
