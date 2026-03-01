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
OUTPUT_DIR = str(OUT_ROOT / "coverage")
TREND_CSV = os.path.join(OUTPUT_DIR, "coverage_trend.csv")
TREND_IMG = os.path.join(OUTPUT_DIR, "coverage_trend_report.png")

SITE_LAT, SITE_LON = get_site_latlon()
MAX_WORKERS = min(6, os.cpu_count() or 6)

def calculate_distance_bearing(lat1, lon1, lat2, lon2):
    """
adsb_polar_coverage_evaluator.py module.
"""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    bearing = (bearing + 360) % 360

    return dist, bearing

def process_one_file(f_path: str):
    filename = os.path.basename(f_path)
    base_name = filename.split('.')[0]
    out_img = os.path.join(OUTPUT_DIR, f"{base_name}_polar_p95.png")

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
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d:
                    lat, lon = d['lat'], d['lon']
                    if 20 < lat < 50 and 120 < lon < 150:
                        dist, bearing = calculate_distance_bearing(SITE_LAT, SITE_LON, lat, lon)
                        data.append({'bearing': bearing, 'dist': dist})
            except:
                continue

    sample_count = len(data)
    if sample_count == 0:
        return None

    df = pd.DataFrame(data)
    df['bearing_bin'] = df['bearing'].round().astype(int) % 360

    polar_stats = df.groupby('bearing_bin')['dist'].agg(
        dist_max='max',
        dist_p95=lambda x: np.percentile(x, 95)
    ).reindex(range(360)).fillna(0).reset_index()

    avg_p95_dist = polar_stats['dist_p95'].mean()
    area_score_p95 = np.sum(np.pi * (polar_stats['dist_p95'] ** 2) / 360)

    theta = np.radians(polar_stats['bearing_bin'])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Polar P95 Coverage: {base_name}", fontsize=12)
    ax.plot(theta, polar_stats['dist_p95'], color='blue', linewidth=1.5, label='P95 Dist')
    ax.fill(theta, polar_stats['dist_p95'], color='blue', alpha=0.1)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.1))
    fig.tight_layout()
    fig.savefig(out_img, dpi=150)
    plt.close(fig)

    return {
        'date': record_date,
        'sample_count': sample_count,
        'avg_p95_dist': avg_p95_dist,
        'area_p95': area_score_p95
    }


def process_polar_coverage():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*"))
    if not files:
        print(f" File not found: {INPUT_DIR}")
        return

    print(f">>> {len(files)}  scanning days of data and computing P95 coverage and trend... (workers={MAX_WORKERS})\n")

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
        print(f"\n Dropped first/last day data. Valid days: {len(df_trend)}")
    else:
        print("\n Too few days; skipping first/last day trimming.")

    df_trend.to_csv(TREND_CSV, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_positions = np.arange(len(df_trend))
    x_labels = df_trend['date'].apply(lambda d: d.strftime('%Y-%m-%d')).tolist()

    ax.plot(x_positions, df_trend['avg_p95_dist'], marker='o', label='Avg P95 Dist (km)')
    ax2 = ax.twinx()
    ax2.plot(x_positions, df_trend['area_p95'], color='orange', marker='s', label='Area P95')

    ax.set_title("P95 Coverage Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg P95 Distance (km)")
    ax2.set_ylabel("P95 Area Score")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    fig.savefig(TREND_IMG, dpi=150)
    plt.close(fig)

    print(f"� Saved trend CSV: {TREND_CSV}")
    print(f" Saved trend image: {TREND_IMG}")

if __name__ == "__main__":
    process_polar_coverage()
