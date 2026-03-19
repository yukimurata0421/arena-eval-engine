# Merged Output for AI

- base_dir: `<SAMPLE_INPUT>`
- generated_at: `1970-01-01T00:00:00`
- include_ext: `.csv, .jsonl, .txt`
- max_bytes_per_file: `2000000`

## File Index (included)

- `README.txt` (113 bytes, mtime=2024-01-01T09:00:00)
- `daily_metrics.csv` (655 bytes, mtime=2024-01-01T09:00:00)
- `signal_events.jsonl` (2678 bytes, mtime=2024-01-01T09:00:00)

## Contents



---
### README.txt
- abs_path: `<SAMPLE_INPUT>\README.txt`
- size_bytes: `113`
- mtime: `2024-01-01T09:00:00`
- encoding: `utf-8-sig`

```
This smoke fixture is synthetic and deterministic.
It is intended only for release-layer reproducibility checks.
```


---
### daily_metrics.csv
- abs_path: `<SAMPLE_INPUT>\daily_metrics.csv`
- size_bytes: `655`
- mtime: `2024-01-01T09:00:00`
- encoding: `utf-8-sig`

```csv
date,distance_band_km,auc,auc_n_used,minutes_covered,message_count
2026-01-01,50,0.8774,23,86,426
2026-01-01,100,0.5853,23,68,380
2026-01-01,150,0.7366,19,68,294
2026-01-02,50,0.6767,24,89,422
2026-01-02,100,0.8675,20,65,379
2026-01-02,150,0.6282,22,89,265
2026-01-03,50,0.6273,22,75,342
2026-01-03,100,0.6205,23,74,270
2026-01-03,150,0.7037,22,79,319
2026-01-04,50,0.7756,24,80,414
2026-01-04,100,0.732,22,88,422
2026-01-04,150,0.6976,26,70,293
2026-01-05,50,0.6369,21,71,323
2026-01-05,100,0.8433,21,71,336
2026-01-05,150,0.8097,20,71,316
2026-01-06,50,0.8456,21,71,357
2026-01-06,100,0.7368,18,78,317
2026-01-06,150,0.6922,18,90,297
```


---
### signal_events.jsonl
- abs_path: `<SAMPLE_INPUT>\signal_events.jsonl`
- size_bytes: `2678`
- mtime: `2024-01-01T09:00:00`
- encoding: `utf-8-sig`

```json
{"date": "2026-01-01", "decode_ok": false, "distance_km": 121.977, "event_id": "evt-0000", "rssi_dbfs": -37.574}
{"date": "2026-01-02", "decode_ok": true, "distance_km": 71.519, "event_id": "evt-0001", "rssi_dbfs": -45.863}
{"date": "2026-01-03", "decode_ok": true, "distance_km": 133.265, "event_id": "evt-0002", "rssi_dbfs": -33.763}
{"date": "2026-01-04", "decode_ok": true, "distance_km": 60.182, "event_id": "evt-0003", "rssi_dbfs": -48.513}
{"date": "2026-01-05", "decode_ok": true, "distance_km": 116.004, "event_id": "evt-0004", "rssi_dbfs": -36.213}
{"date": "2026-01-06", "decode_ok": false, "distance_km": 97.863, "event_id": "evt-0005", "rssi_dbfs": -46.582}
{"date": "2026-01-01", "decode_ok": true, "distance_km": 95.64, "event_id": "evt-0006", "rssi_dbfs": -38.65}
{"date": "2026-01-02", "decode_ok": true, "distance_km": 194.362, "event_id": "evt-0007", "rssi_dbfs": -47.342}
{"date": "2026-01-03", "decode_ok": true, "distance_km": 171.664, "event_id": "evt-0008", "rssi_dbfs": -49.012}
{"date": "2026-01-04", "decode_ok": true, "distance_km": 143.922, "event_id": "evt-0009", "rssi_dbfs": -45.028}
{"date": "2026-01-05", "decode_ok": false, "distance_km": 72.806, "event_id": "evt-0010", "rssi_dbfs": -42.293}
{"date": "2026-01-06", "decode_ok": true, "distance_km": 125.735, "event_id": "evt-0011", "rssi_dbfs": -36.773}
{"date": "2026-01-01", "decode_ok": true, "distance_km": 159.439, "event_id": "evt-0012", "rssi_dbfs": -34.197}
{"date": "2026-01-02", "decode_ok": true, "distance_km": 95.476, "event_id": "evt-0013", "rssi_dbfs": -47.432}
{"date": "2026-01-03", "decode_ok": true, "distance_km": 63.769, "event_id": "evt-0014", "rssi_dbfs": -49.443}
{"date": "2026-01-04", "decode_ok": false, "distance_km": 66.949, "event_id": "evt-0015", "rssi_dbfs": -49.695}
{"date": "2026-01-05", "decode_ok": true, "distance_km": 83.064, "event_id": "evt-0016", "rssi_dbfs": -45.21}
{"date": "2026-01-06", "decode_ok": true, "distance_km": 199.594, "event_id": "evt-0017", "rssi_dbfs": -36.549}
{"date": "2026-01-01", "decode_ok": true, "distance_km": 121.016, "event_id": "evt-0018", "rssi_dbfs": -44.394}
{"date": "2026-01-02", "decode_ok": true, "distance_km": 98.636, "event_id": "evt-0019", "rssi_dbfs": -32.005}
{"date": "2026-01-03", "decode_ok": false, "distance_km": 129.996, "event_id": "evt-0020", "rssi_dbfs": -48.547}
{"date": "2026-01-04", "decode_ok": true, "distance_km": 93.104, "event_id": "evt-0021", "rssi_dbfs": -35.346}
{"date": "2026-01-05", "decode_ok": true, "distance_km": 55.837, "event_id": "evt-0022", "rssi_dbfs": -42.911}
{"date": "2026-01-06", "decode_ok": true, "distance_km": 178.959, "event_id": "evt-0023", "rssi_dbfs": -39.075}
```
