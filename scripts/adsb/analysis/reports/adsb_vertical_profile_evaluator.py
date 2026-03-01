import os
import sys
from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


from arena.lib.config import get_site_latlon
from arena.lib.paths import DATA_DIR, OUTPUT_DIR as OUT_ROOT

INPUT_DIR = str(DATA_DIR / "plao_pos")
OUTPUT_DIR = str(OUT_ROOT / "vertical_profile")
TREND_CSV = os.path.join(OUTPUT_DIR, "los_efficiency_trend.csv")
TREND_IMG = os.path.join(OUTPUT_DIR, "los_efficiency_trend_report.png")

SITE_LAT, SITE_LON = get_site_latlon()
RECEIVER_ALT_M = 4.87
MAX_WORKERS = min(6, os.cpu_count() or 6)

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_los_distance(alt_ft, rx_alt_m):
    """
adsb_vertical_profile_evaluator.py module.
"""
    alt_m = alt_ft * 0.3048
    alt_m = np.where(alt_m > 0, alt_m, 0)
    return 4.12 * (np.sqrt(alt_m) + np.sqrt(rx_alt_m))

def process_one_file(f_path: str):
    filename = os.path.basename(f_path)
    base_name = filename.split('.')[0]

    try:
        date_str = base_name.split('_')[1]
        record_date = datetime.strptime(date_str, "%Y%m%d").date()
    except:
        return None

    data = []
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d and 'alt' in d:
                    lat, lon, alt = d['lat'], d['lon'], d['alt']
                    if 20 < lat < 50 and 120 < lon < 150 and 10000 <= alt <= 42000:
                        dist = calculate_distance(SITE_LAT, SITE_LON, lat, lon)
                        data.append({'dist': dist, 'alt': alt})
            except:
                continue

    sample_count = len(data)
    if sample_count < 10000:
        return None

    df = pd.DataFrame(data)
    df['alt_bin'] = (df['alt'] // 4000) * 4000 + 2000

    stats = df.groupby('alt_bin')['dist'].apply(lambda x: np.percentile(x, 95)).reset_index()
    stats.rename(columns={'dist': 'p95_dist'}, inplace=True)
    stats['los_dist'] = calculate_los_distance(stats['alt_bin'], RECEIVER_ALT_M)
    stats['efficiency_pct'] = (stats['p95_dist'] / stats['los_dist']) * 100

    daily_efficiency = stats['efficiency_pct'].mean()
    return {
        'date': record_date,
        'sample_count': sample_count,
        'los_efficiency': daily_efficiency
    }


def process_los_efficiency_trend():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*"))
    if not files:
        print(f" File not found: {INPUT_DIR}")
        return

    print(f">>> {len(files)}  scanning days of data and computing LOS achievement (%)... (workers={MAX_WORKERS})\n")

    trend_data = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    trend_data.append(result)
            except Exception:
                continue

    if not trend_data:
        print("No valid data.")
        return

    df_trend = pd.DataFrame(trend_data).sort_values('date').reset_index(drop=True)

    if len(df_trend) > 2:
        df_trend = df_trend.iloc[1:-1].copy()

    df_trend.to_csv(TREND_CSV, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(len(df_trend))
    x_labels = df_trend['date'].apply(lambda d: d.strftime('%Y-%m-%d')).tolist()

    ax.plot(x_positions, df_trend['los_efficiency'], marker='o', color='magenta', lw=2, markersize=8, label='LOS Achievement Rate (%)')
    ax.fill_between(x_positions, df_trend['los_efficiency'], color='magenta', alpha=0.1)

    from arena.lib.phase_config import get_config
    phases = get_config().vertical_phases

    for p in phases:
        d_val = pd.Timestamp(p['date']).date()
        if d_val in df_trend['date'].values:
            idx = df_trend.index[df_trend['date'] == d_val].tolist()[0]
            pos = x_positions[list(df_trend['date']).index(d_val)]
            ax.axvline(pos, color=p['color'], linestyle='--', lw=1.5, alpha=0.8)
            ax.text(pos, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                     f" {p['name']}", color=p['color'], fontweight='bold', va='bottom', ha='left', rotation=30, fontsize=10)

    ax.set_title("Theoretical Limit (Line of Sight) vs Actual P95 Achievement Rate (%)", fontsize=15, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("LOS Achievement Rate (%)", fontsize=12, fontweight='bold')

    min_y = np.floor(df_trend['los_efficiency'].min() / 5) * 5
    max_y = np.ceil(df_trend['los_efficiency'].max() / 5) * 5
    ax.set_ylim(min_y, max(100, max_y))

    ax.axhline(90, color='gray', linestyle=':', lw=1.5, label='90% Excellence Line')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(TREND_IMG, dpi=150)
    plt.close(fig)

    print(f"\n� Saved trend CSV: {TREND_CSV}")
    print(f" Saved trend image: {TREND_IMG}")

if __name__ == "__main__":
    process_los_efficiency_trend()
