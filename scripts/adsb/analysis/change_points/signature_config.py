from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_QUANTILES: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)
DEFAULT_COVERAGE_GRID_KM: tuple[int, ...] = tuple(range(30, 301, 30))


@dataclass(frozen=True)
class TrafficAdjustmentConfig:
    method: str = "residual_log_linear"
    min_points: int = 20
    clip_min_traffic: float = 1.0


@dataclass(frozen=True)
class DailySignatureConfig:
    data_root: Path
    output_root: Path
    traffic_csv: Path
    min_days: int = 30
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    coverage_grid_km: tuple[int, ...] = DEFAULT_COVERAGE_GRID_KM
    cp_model: str = "rank_scan"
    effect_model: str = "hl_mwu"
    dry_run: bool = False
    timezone: str = "Asia/Tokyo"
    min_segment_days: int = 7
    bootstrap_iterations: int = 1000
    random_seed: int = 42
    traffic_adjustment: TrafficAdjustmentConfig = field(default_factory=TrafficAdjustmentConfig)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_root"] = str(self.output_root)
        payload["traffic_csv"] = str(self.traffic_csv)
        payload["quantiles"] = list(self.quantiles)
        payload["coverage_grid_km"] = list(self.coverage_grid_km)
        return payload


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("empty float list")
    return tuple(values)


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("empty int list")
    return tuple(values)


def make_timestamped_run_dir(output_root: Path, prefix: str = "daily_signature_cp") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{prefix}_{timestamp}"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = output_root / f"{prefix}_{timestamp}_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1
