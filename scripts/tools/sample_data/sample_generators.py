from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True)
class BuildResult:
    files_written: list[str]
    row_counts: dict[str, int]
    source_type: str


def _set_fixed_mtime(path: Path, epoch_seconds: int) -> None:
    os.utime(path, (epoch_seconds, epoch_seconds))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return len(records)


def _write_text(path: Path, body: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    return len([line for line in body.splitlines() if line.strip()])


def build_synthetic_smoke_sample(input_dir: Path, seed: int, fixed_mtime_epoch: int) -> BuildResult:
    rng = random.Random(seed)
    files_written: list[str] = []
    row_counts: dict[str, int] = {}

    daily_rows: list[dict[str, object]] = []
    bands = (50, 100, 150)
    start = date(2026, 1, 1)
    for day_offset in range(6):
        day = (start + timedelta(days=day_offset)).isoformat()
        for band in bands:
            auc = round(0.55 + rng.random() * 0.35, 4)
            daily_rows.append(
                {
                    "date": day,
                    "distance_band_km": band,
                    "auc": auc,
                    "auc_n_used": int(18 + rng.randint(0, 8)),
                    "minutes_covered": int(65 + rng.randint(0, 30)),
                    "message_count": int(250 + rng.randint(0, 180)),
                }
            )

    events: list[dict[str, object]] = []
    for index in range(24):
        events.append(
            {
                "event_id": f"evt-{index:04d}",
                "date": (start + timedelta(days=index % 6)).isoformat(),
                "distance_km": round(35.0 + rng.random() * 170.0, 3),
                "rssi_dbfs": round(-32.0 - rng.random() * 18.0, 3),
                "decode_ok": bool(index % 5 != 0),
            }
        )

    files = {
        "daily_metrics.csv": _write_csv(
            input_dir / "daily_metrics.csv",
            ["date", "distance_band_km", "auc", "auc_n_used", "minutes_covered", "message_count"],
            daily_rows,
        ),
        "signal_events.jsonl": _write_jsonl(input_dir / "signal_events.jsonl", events),
        "README.txt": _write_text(
            input_dir / "README.txt",
            (
                "This smoke fixture is synthetic and deterministic.\n"
                "It is intended only for release-layer reproducibility checks.\n"
            ),
        ),
    }

    for filename, row_count in files.items():
        file_path = input_dir / filename
        _set_fixed_mtime(file_path, fixed_mtime_epoch)
        files_written.append(str(file_path.relative_to(input_dir.parent)).replace("\\", "/"))
        row_counts[str(file_path.relative_to(input_dir.parent)).replace("\\", "/")] = row_count

    return BuildResult(files_written=sorted(files_written), row_counts=row_counts, source_type="synthetic")


def build_from_real_cutout_sample(_input_dir: Path, _seed: int, _fixed_mtime_epoch: int) -> BuildResult:
    raise NotImplementedError(
        "mode=from-real is not implemented yet. Planned extension point: deterministic cutout from private local real-data roots."
    )


def build_sample(mode: str, input_dir: Path, seed: int, fixed_mtime_epoch: int) -> BuildResult:
    if mode == "synthetic":
        return build_synthetic_smoke_sample(input_dir=input_dir, seed=seed, fixed_mtime_epoch=fixed_mtime_epoch)
    if mode == "from-real":
        return build_from_real_cutout_sample(input_dir=input_dir, seed=seed, fixed_mtime_epoch=fixed_mtime_epoch)
    raise ValueError(f"unsupported mode: {mode}")

