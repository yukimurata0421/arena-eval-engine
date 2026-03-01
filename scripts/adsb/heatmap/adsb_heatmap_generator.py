import json

import folium
import pandas as pd
from folium.plugins import HeatMap
import os
import sys
from pathlib import Path


from arena.lib.config import get_site_latlon
from arena.lib.paths import DATA_DIR, OUTPUT_DIR as OUT_ROOT

INPUT_JSONL = str(DATA_DIR / "plao_pos" / "pos_20260218.jsonl")
OUTPUT_HTML = str(OUT_ROOT / "adsb_coverage_heatmap.html")

SITE_LAT, SITE_LON = get_site_latlon()

def generate_heatmap():
    print(">>> Loading data...")
    coordinates = []

    if not os.path.exists(INPUT_JSONL):
        print(f"❌ File not found: {INPUT_JSONL}")
        return

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos' and 'lat' in d and 'lon' in d:
                    coordinates.append([d['lat'], d['lon']])
            except json.JSONDecodeError:
                continue

    if not coordinates:
        print("❌ No valid location data found.")
        return

    print(f">>> Mapping {len(coordinates):,} points to the map...")

    m = folium.Map(location=[SITE_LAT, SITE_LON], zoom_start=7, tiles='CartoDB dark_matter')

    folium.Marker(
        [SITE_LAT, SITE_LON],
        popup="Tsuchiura ADS-B Station",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    HeatMap(coordinates, radius=10, blur=15).add_to(m)

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    m.save(OUTPUT_HTML)
    print(f"✅ Heatmap created: {OUTPUT_HTML}")
    print("Open this HTML file in a browser to view it.")

if __name__ == "__main__":
    generate_heatmap()
