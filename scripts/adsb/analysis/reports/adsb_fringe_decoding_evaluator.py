import os
import sys
from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


from arena.lib.config import get_site_latlon
from arena.lib.geo import haversine_km
from arena.lib.paths import DATA_DIR, OUTPUT_DIR as OUT_ROOT
from arena.lib.platform_setup import resolve_workers
from arena.log import get_script_logger

log = get_script_logger(__name__)

INPUT_DIR = str(DATA_DIR / "plao_pos")
OUTPUT_DIR = str(OUT_ROOT / "fringe_decoding")
TREND_CSV = os.path.join(OUTPUT_DIR, "fringe_decoding_stats.csv")
TREND_LEGACY_CSV = os.path.join(OUTPUT_DIR, "fringe_decoding_trend.csv")
TREND_IMG = os.path.join(OUTPUT_DIR, "fringe_decoding_trend_report.png")

SITE_LAT, SITE_LON = get_site_latlon()



MAX_WORKERS = resolve_workers(default_cap=12)

def get_phase(date):
    from arena.lib.phase_config import get_config
    cfg = get_config()
    d = pd.Timestamp(date)
    # fringe_boundaries: [("2026-01-31", "1_Old_Settings"), ("2026-02-14", "2_New_Filter")]
    for boundary_date, phase_name in cfg.fringe_boundaries:
        if d <= pd.Timestamp(boundary_date):
            return phase_name
    return '3_Post_Cable_Fix'

def process_one_file(f_path: str):
    base = os.path.basename(f_path).split('.')[0]
    try:
        date_str = base.split('_')[1]
        record_date = datetime.strptime(date_str, "%Y%m%d").date()
    except Exception:
        return None

    counts = {'near': 0, 'mid': 0, 'far': 0, 'extreme': 0}
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get('type') == 'pos':
                    dist = haversine_km(SITE_LAT, SITE_LON, d['lat'], d['lon'])
                    if dist < 100:
                        counts['near'] += 1
                    elif dist < 200:
                        counts['mid'] += 1
                    elif dist < 300:
                        counts['far'] += 1
                    else:
                        counts['extreme'] += 1
            except Exception:
                continue

    total = sum(counts.values())
    if total < 500:
        return None

    fringe_count = counts['far'] + counts['extreme']
    ratio = (fringe_count / total) * 100
    p = fringe_count / total
    ci_95 = 1.96 * np.sqrt((p * (1 - p)) / total) * 100

    return {
        'date': record_date, 'phase': get_phase(record_date), 'total': total,
        'fringe_ratio': ratio, 'ci_95_err': ci_95,
        'dist_0_100': counts['near'], 'dist_100_200': counts['mid'],
        'dist_200_300': counts['far'], 'dist_300_plus': counts['extreme']
    }


def process_fringe_decoding():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*pos*.jsonl*")))
    log.info(f">>> Recalculating and re-aggregating distances... (workers={MAX_WORKERS})\n")
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception:
                continue

    if not results:
        df = pd.DataFrame(columns=[
            "date", "phase", "total", "fringe_ratio", "ci_95_err",
            "dist_0_100", "dist_100_200", "dist_200_300", "dist_300_plus"
        ])
        df.to_csv(TREND_CSV, index=False)
        log.info(f"\n ⚠️ No target data. Exported empty CSV: {TREND_CSV}")
        return

    df = pd.DataFrame(results).sort_values('date')
    df.to_csv(TREND_CSV, index=False)
    df_legacy = pd.DataFrame({
        "date": df["date"],
        "total": df["total"],
        "dist_0_100": df["dist_0_100"],
        "dist_100_200": df["dist_100_200"],
        "dist_200_300": df["dist_200_300"],
        "dist_300_plus": df["dist_300_plus"],
        "fringe_ratio_pct": df["fringe_ratio"],
    })
    df_legacy.to_csv(TREND_LEGACY_CSV, index=False)
    log.info(f"\nCreated distance corrected CSV: {TREND_CSV}")

if __name__ == "__main__":
    process_fringe_decoding()
