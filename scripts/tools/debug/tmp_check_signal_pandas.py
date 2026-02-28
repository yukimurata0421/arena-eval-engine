import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from arena.lib.paths import DATA_DIR, OUTPUT_DIR

f_path = DATA_DIR / "dist_signal_stats_1m.jsonl"
per_date = {}
with f_path.open('r', encoding='utf-8') as f:
    for line in f:
        try:
            d = json.loads(line)
            ts = d.get('ts') or (datetime.fromisoformat(d['ts_iso'].replace('Z', '')).timestamp() if 'ts_iso' in d else None)
            if not ts:
                continue
            dt = datetime.fromtimestamp(ts).date()
            buckets = d.get('buckets', {})
            b_150 = buckets.get('150-175km', {})
            if b_150.get('n_samples', 0) > 0:
                sig = b_150.get('avg_signal')
                snr = b_150.get('avg_snr')
                entry = per_date.setdefault(dt, {"sig_sum": 0.0, "sig_n": 0, "snr_sum": 0.0, "snr_n": 0})
                if sig is not None:
                    entry["sig_sum"] += float(sig)
                    entry["sig_n"] += 1
                if snr is not None:
                    entry["snr_sum"] += float(snr)
                    entry["snr_n"] += 1
        except Exception:
            continue

rows = []
for dt, v in per_date.items():
    row = {"date": dt}
    row["sig_150_175"] = (v["sig_sum"] / v["sig_n"]) if v["sig_n"] > 0 else None
    row["snr_150_175"] = (v["snr_sum"] / v["snr_n"]) if v["snr_n"] > 0 else None
    rows.append(row)

print('rows', len(rows))
print('sample', rows[:2])

out = Path(OUTPUT_DIR) / "_tmp_signal.csv"
if rows:
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
else:
    df = pd.DataFrame(columns=['date','sig_150_175','snr_150_175'])

df.to_csv(out, index=False)
print('wrote', out, 'size', out.stat().st_size)
