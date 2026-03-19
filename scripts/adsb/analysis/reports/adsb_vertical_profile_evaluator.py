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

# Ensure local src package is preferred when this script is executed directly.
_ROOT = Path(__file__).resolve().parents[4]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from arena.lib.config import get_site_latlon
from arena.lib.geo import haversine_km
from arena.lib.paths import DATA_DIR, OUTPUT_DIR as OUT_ROOT
from arena.lib.platform_setup import resolve_workers
from arena.log import get_script_logger

log = get_script_logger(__name__)

INPUT_DIR = str(DATA_DIR / "plao_pos")
OUTPUT_DIR = str(OUT_ROOT / "vertical_profile")
TREND_CSV = os.path.join(OUTPUT_DIR, "los_efficiency_trend.csv")
TREND_IMG = os.path.join(OUTPUT_DIR, "los_efficiency_trend_report.png")

SITE_LAT, SITE_LON = get_site_latlon()
RECEIVER_ALT_M = 4.87



MAX_WORKERS = resolve_workers(default_cap=12)


def _site_is_valid(lat: float, lon: float) -> bool:
    return abs(lat) > 0.001 and abs(lon) > 0.001

def calculate_los_distance(alt_ft, rx_alt_m):
    """
adsb_vertical_profile_evaluator.py module.
"""
    alt_m = alt_ft * 0.3048
    alt_m = np.where(alt_m > 0, alt_m, 0)
    return 4.12 * (np.sqrt(alt_m) + np.sqrt(rx_alt_m))

def save_daily_vertical_profile_plot(record_date, stats_df):
    date_token = record_date.strftime("%Y%m%d")
    out_png = os.path.join(OUTPUT_DIR, f"pos_{date_token}_vertical_profile.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stats_df["alt_bin"], stats_df["p95_dist"], marker="o", lw=2, label="Actual P95 Distance")
    ax.plot(stats_df["alt_bin"], stats_df["los_dist"], marker="s", lw=2, linestyle="--", label="Theoretical LOS")
    ax.set_title(f"Vertical Profile: {record_date.isoformat()}")
    ax.set_xlabel("Altitude Bin (ft)")
    ax.set_ylabel("Distance (km)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def process_one_file(f_path: str):
    filename = os.path.basename(f_path)
    base_name = filename.split('.')[0]

    try:
        date_str = base_name.split('_')[1]
        record_date = datetime.strptime(date_str, "%Y%m%d").date()
    except Exception:
        return None

    data = []
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d and 'alt' in d:
                    lat, lon, alt = d['lat'], d['lon'], d['alt']
                    if 20 < lat < 50 and 120 < lon < 150 and 10000 <= alt <= 42000:
                        dist = haversine_km(SITE_LAT, SITE_LON, lat, lon)
                        data.append({'dist': dist, 'alt': alt})
            except Exception:
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
    save_daily_vertical_profile_plot(record_date, stats)

    daily_efficiency = stats['efficiency_pct'].mean()
    return {
        'date': record_date,
        'sample_count': sample_count,
        'los_efficiency': daily_efficiency
    }


def process_los_efficiency_trend():
    if not _site_is_valid(SITE_LAT, SITE_LON):
        log.info(" 観測点座標が不正です (lat/lon=0)。settings.toml の [site] を確認してください。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*"))
    if not files:
        log.info(f" ファイルが見つかりません: {INPUT_DIR}")
        return

    log.info(f">>> {len(files)} 日分のデータを走査し、LOS達成率(%)を算出中... (workers={MAX_WORKERS})\n")

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
        log.info("有効データがありません。")
        return

    df_trend = pd.DataFrame(trend_data).sort_values('date').reset_index(drop=True)

    df_trend.to_csv(TREND_CSV, index=False)

    # Use a taller canvas for readability on long date ranges.
    fig_height = max(9.0, min(16.0, 7.0 + (len(df_trend) / 10.0)))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
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

    tick_step = max(1, len(x_positions) // 20)
    tick_positions = x_positions[::tick_step]
    tick_labels = [x_labels[i] for i in range(0, len(x_labels), tick_step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    fig.tight_layout(pad=1.4)
    fig.savefig(TREND_IMG, dpi=150)
    plt.close(fig)

    log.info(f"\n 保存しました: {TREND_CSV}")
    log.info(f" 保存しました: {TREND_IMG}")

if __name__ == "__main__":
    process_los_efficiency_trend()
