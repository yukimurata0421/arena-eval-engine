import csv
import json
from datetime import datetime
from pathlib import Path

from arena.lib.paths import DATA_DIR, OUTPUT_DIR


def _load(path: Path):
    per_date = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get('ts')
            if ts is None and 'ts_iso' in d:
                try:
                    ts = datetime.fromisoformat(d['ts_iso'].replace('Z', '')).timestamp()
                except Exception:
                    ts = None
            if ts is None:
                continue
            if isinstance(ts, str):
                try:
                    ts = float(ts)
                except Exception:
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '')).timestamp()
                    except Exception:
                        continue
            dt = datetime.fromtimestamp(float(ts)).date().isoformat()
            buckets = d.get('buckets', {})
            b_150 = buckets.get('150-175km', {})
            if b_150.get('n_samples', 0) <= 0:
                continue
            sig = b_150.get('avg_signal')
            snr = b_150.get('avg_snr')
            entry = per_date.setdefault(dt, {
                'sig_sum': 0.0,
                'sig_n': 0,
                'snr_sum': 0.0,
                'snr_n': 0,
            })
            if sig is not None:
                entry['sig_sum'] += float(sig)
                entry['sig_n'] += 1
            if snr is not None:
                entry['snr_sum'] += float(snr)
                entry['snr_n'] += 1
    return per_date


def main() -> None:
    src = DATA_DIR / 'dist_signal_stats_1m.jsonl'
    rows = []
    per_date = _load(src)
    for dt, v in per_date.items():
        rows.append({
            'date': dt,
            'sig_150_175': (v['sig_sum'] / v['sig_n']) if v['sig_n'] else None,
            'snr_150_175': (v['snr_sum'] / v['snr_n']) if v['snr_n'] else None,
        })
    rows.sort(key=lambda r: r['date'])

    out = OUTPUT_DIR / '_tmp_signal.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'sig_150_175', 'snr_150_175'])
        writer.writeheader()
        writer.writerows(rows)

    print('rows', len(rows))
    print('wrote', out, 'size', out.stat().st_size if out.exists() else 0)


if __name__ == '__main__':
    main()
