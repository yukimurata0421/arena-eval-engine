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
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_alt_heatmap.html")

    coords_low = []
    coords_mid = []
    coords_high = []

    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d and 'alt' in d:
                    lat, lon, alt = d['lat'], d['lon'], d['alt']
                    if 20 < lat < 50 and 120 < lon < 150:
                        if alt < 10000:
                            coords_low.append([lat, lon])
                        elif 10000 <= alt < 25000:
                            coords_mid.append([lat, lon])
                        else:
                            coords_high.append([lat, lon])
            except:
                continue

    sample_rate = 50
    coords_low = coords_low[::sample_rate]
    coords_mid = coords_mid[::sample_rate]
    coords_high = coords_high[::sample_rate]

    m = folium.Map(location=[SITE_LAT, SITE_LON], zoom_start=7, tiles='CartoDB dark_matter')
    folium.Marker(
        [SITE_LAT, SITE_LON],
        popup="Tsuchiura ADS-B (00001090)",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    fg_low = folium.FeatureGroup(name="Low Altitude (< 10,000 ft)", show=False)
    fg_mid = folium.FeatureGroup(name="Mid Altitude (10,000 - 25,000 ft)", show=False)
    fg_high = folium.FeatureGroup(name="High Altitude (> 25,000 ft)", show=True)

    if coords_low:
        HeatMap(coords_low, radius=6, blur=4,
                gradient={0.4: 'navy', 0.65: 'cyan', 1.0: 'lime'}).add_to(fg_low)
    if coords_mid:
        HeatMap(coords_mid, radius=6, blur=4,
                gradient={0.4: 'navy', 0.65: 'yellow', 1.0: 'orange'}).add_to(fg_mid)
    if coords_high:
        HeatMap(coords_high, radius=6, blur=4,
                gradient={0.4: 'purple', 0.65: 'red', 1.0: 'magenta'}).add_to(fg_high)

    fg_low.add_to(m)
    fg_mid.add_to(m)
    fg_high.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_path)
    return f"   ✅ {filename}: Saved {out_path} (Low: {len(coords_low):,}, Mid: {len(coords_mid):,}, High: {len(coords_high):,})"


def process_daily_heatmaps_by_alt():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*"))
    if not files:
        print(f"❌ File not found: {INPUT_DIR}")
        return

    print(f">>> {len(files)}  days of data detected. Starting altitude-band parallel processing... (workers={MAX_WORKERS})\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")

if __name__ == "__main__":
    process_daily_heatmaps_by_alt()
