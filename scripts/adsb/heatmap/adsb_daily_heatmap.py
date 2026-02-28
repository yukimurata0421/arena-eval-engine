import os
import sys
from pathlib import Path
import json
import glob
import folium
from folium.plugins import HeatMap
from concurrent.futures import ProcessPoolExecutor, as_completed


from arena.lib.config import get_site_latlon
from arena.lib.paths import DATA_DIR, OUTPUT_DIR as OUT_ROOT

INPUT_DIR = str(DATA_DIR / "plao_pos")
OUTPUT_DIR = str(OUT_ROOT / "heatmaps")
SITE_LAT, SITE_LON = get_site_latlon()
MAX_WORKERS = min(6, os.cpu_count() or 6)

def process_one_file(f_path: str):
    filename = os.path.basename(f_path)
    base_name = filename.split('.')[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_heatmap.html")

    coords = []
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d:
                    lat, lon = d['lat'], d['lon']
                    if 20 < lat < 50 and 120 < lon < 150:
                        coords.append([lat, lon])
            except:
                continue

    if not coords:
        return f"   ⚠️ {filename}: No valid data. Skipping."

    sampled_coords = coords[::50]

    m = folium.Map(location=[SITE_LAT, SITE_LON], zoom_start=7, tiles='CartoDB dark_matter')
    folium.Marker(
        [SITE_LAT, SITE_LON],
        popup="Tsuchiura ADS-B (00001090)",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    HeatMap(
        sampled_coords,
        radius=6,
        blur=4,
        max_zoom=9,
        gradient={0.1: 'navy', 0.3: 'blue', 0.5: 'cyan', 0.7: 'lime', 0.9: 'yellow', 1.0: 'red'}
    ).add_to(m)

    m.save(out_path)
    return f"   ✅ {filename}: Saved {out_path} (raw: {len(coords):,} -> plotted: {len(sampled_coords):,})"


def process_daily_heatmaps():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*"))
    if not files:
        print(f"❌ File not found: {INPUT_DIR}")
        return

    print(f">>> {len(files)}  days of data detected. Starting parallel processing... (workers={MAX_WORKERS})\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")

if __name__ == "__main__":
    process_daily_heatmaps()
