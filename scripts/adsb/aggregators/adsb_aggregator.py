import os
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


from arena.lib.paths import RAW_DIR, OUTPUT_DIR as OUT_ROOT, DATA_DIR
from arena.lib.platform_setup import resolve_workers
from arena.log import get_script_logger

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

log = get_script_logger(__name__)

LOCAL_TZ = "Asia/Tokyo"
_LOCAL_TZ_INFO = ZoneInfo(LOCAL_TZ) if ZoneInfo else None


def aggregate_data_v3():
    raw_dir = str(RAW_DIR / "past_log")
    output_dir = str(OUT_ROOT)
    traffic_csv = str(DATA_DIR / "flight_data" / "airport_movements.csv")
    # Avoid overwriting eval_pk_aggregator output
    output_file = os.path.join(output_dir, "adsb_daily_summary_raw.csv")
    
    files = []
    # current dist_1m
    dist_current = str(Path(RAW_DIR).parent / "dist_1m.jsonl")
    if os.path.exists(dist_current):
        files.append(dist_current)
    # archived dist jsonl
    files.extend(glob.glob(os.path.join(raw_dir, "*dist*.jsonl.till-*")))
    
    if not files:
        log.info(f"[ERROR] File not found: {raw_dir}")
        return

    log.info(f">>> Scanning {len(files)} files... (parallel)")

    max_workers = resolve_workers(default_cap=12)

    def process_file(f_path: str):
        if not os.path.isfile(f_path):
            return {}
        per_date_sum = {}
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # keep only dist_1m payloads
                    src = data.get("src", "")
                    if src and src != "dist_1m":
                        continue

                    ts = data.get("ts")
                    n_used = data.get("n_used")
                    if ts is None or n_used is None:
                        continue

                    try:
                        ts_f = float(ts)
                        n_used_i = int(n_used)
                    except Exception:
                        continue

                    if _LOCAL_TZ_INFO:
                        dt = datetime.fromtimestamp(ts_f, tz=timezone.utc).astimezone(_LOCAL_TZ_INFO).date()
                    else:
                        dt = pd.to_datetime(ts_f, unit="s", utc=True).tz_convert(LOCAL_TZ).date()

                    per_date_sum[dt] = per_date_sum.get(dt, 0) + n_used_i
                except Exception:
                    continue
        return per_date_sum

    global_sum = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                per_file = fut.result()
                for dt, val in per_file.items():
                    global_sum[dt] = global_sum.get(dt, 0) + val
            except Exception:
                continue

    if not global_sum:
        log.info("[WARN] No valid AUC data. Export empty CSV.")
        df_daily = pd.DataFrame(
            columns=["date", "auc_n_used", "hnd_nrt_movements", "day_of_week"]
        )
        os.makedirs(output_dir, exist_ok=True)
        df_daily.to_csv(output_file, index=False)
        return

    df_daily = pd.DataFrame(
        [{"date": dt, "auc_n_used": val} for dt, val in global_sum.items()]
    )
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    # Keep raw output lightweight, but use real traffic when available.
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date
    if os.path.exists(traffic_csv):
        try:
            t = pd.read_csv(traffic_csv)
            if "date" in t.columns and "hnd_nrt_movements" in t.columns:
                t["date"] = pd.to_datetime(t["date"]).dt.date
                t = t[["date", "hnd_nrt_movements"]].copy()
                df_daily = pd.merge(df_daily, t, on="date", how="left")
            else:
                df_daily["hnd_nrt_movements"] = pd.NA
        except Exception:
            df_daily["hnd_nrt_movements"] = pd.NA
    else:
        df_daily["hnd_nrt_movements"] = pd.NA

    df_daily["day_of_week"] = pd.to_datetime(df_daily["date"]).dt.day_name().str[:3]
    df_daily = df_daily[["date", "auc_n_used", "hnd_nrt_movements", "day_of_week"]]
    
    os.makedirs(output_dir, exist_ok=True)
    df_daily.to_csv(output_file, index=False)
    log.info(f"[OK] Aggregation complete: {output_file} ({len(df_daily)} days)")

if __name__ == "__main__":
    aggregate_data_v3()
