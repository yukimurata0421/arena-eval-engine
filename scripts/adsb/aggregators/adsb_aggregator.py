import os
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


from arena.lib.paths import RAW_DIR, OUTPUT_DIR as OUT_ROOT

def aggregate_data_v3():
    raw_dir = str(RAW_DIR / "past_log")
    output_dir = str(OUT_ROOT)
    # Avoid overwriting eval_pk_aggregator output
    output_file = os.path.join(output_dir, "adsb_daily_summary_raw.csv")
    
    files = []
    # current dist_1m
    dist_current = str(Path(RAW_DIR).parent / "dist_1m.jsonl")
    if os.path.exists(dist_current):
        files.append(dist_current)
    # archived jsonl
    files.extend(glob.glob(os.path.join(raw_dir, "*jsonl*")))
    
    if not files:
        print(f"❌ File not found: {raw_dir}")
        return

    print(f">>> Scanning {len(files)} files... (parallel)")

    max_workers = min(6, os.cpu_count() or 6)

    def process_file(f_path: str):
        if not os.path.isfile(f_path):
            return {}
        per_date_max = {}
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ts = data.get("ts")
                    if ts is None and "ts_iso" in data:
                        try:
                            ts = datetime.fromisoformat(data["ts_iso"].replace("Z", "")).timestamp()
                        except Exception:
                            ts = None
                    if ts is None:
                        continue
                    if isinstance(ts, str):
                        try:
                            ts = float(ts)
                        except Exception:
                            try:
                                ts = datetime.fromisoformat(ts.replace("Z", "")).timestamp()
                            except Exception:
                                continue
                    dt = datetime.fromtimestamp(ts).date()

                    # allow dist_1m and health logs
                    src = data.get("src", "")
                    count = 0
                    if "counts" in data:
                        count = data["counts"].get("aircraft_total", 0)
                    elif "latest" in data:
                        count = data["latest"].get("n_total", 0)
                    elif "n_total" in data:
                        # prefer dist_1m / dist_pos_health style
                        count = data.get("n_total", 0)
                    elif src == "dist_1m":
                        count = data.get("n_used", 0)

                    if count > 0:
                        prev = per_date_max.get(dt, 0)
                        if count > prev:
                            per_date_max[dt] = count
                except:
                    continue
        return per_date_max

    global_max = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                per_file = fut.result()
                for dt, val in per_file.items():
                    if val > global_max.get(dt, 0):
                        global_max[dt] = val
            except Exception:
                continue

    if not global_max:
        print("⚠️ No valid AUC data. Writing empty CSV.")
        df_daily = pd.DataFrame(
            columns=["date", "auc_n_used", "hnd_nrt_movements", "day_of_week"]
        )
        os.makedirs(output_dir, exist_ok=True)
        df_daily.to_csv(output_file, index=False)
        return

    df_daily = pd.DataFrame(
        [{"date": dt, "auc_n_used": val} for dt, val in global_max.items()]
    )
    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    
    df_daily['hnd_nrt_movements'] = 1000 
    df_daily['day_of_week'] = pd.to_datetime(df_daily['date']).dt.day_name().str[:3]
    
    os.makedirs(output_dir, exist_ok=True)
    df_daily.to_csv(output_file, index=False)
    print(f"✅ Aggregation complete: {output_file} ({len(df_daily)} days)")

if __name__ == "__main__":
    aggregate_data_v3()
