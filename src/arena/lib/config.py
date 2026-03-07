from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arena.lib.runtime_config import load_settings


@dataclass(frozen=True)
class SiteConfig:
    lat: float
    lon: float


@dataclass(frozen=True)
class QualityConfig:
    min_auc_n_used: int
    min_minutes_covered: int


@dataclass(frozen=True)
class DistanceBinsConfig:
    km: list[float]


def _get_settings() -> dict[str, Any]:
    settings = load_settings().data or {}
    return settings if isinstance(settings, dict) else {}


def get_site_config() -> SiteConfig:
    data = _get_settings().get("site", {})
    lat = float(data.get("lat", 0.0))
    lon = float(data.get("lon", 0.0))
    return SiteConfig(lat=lat, lon=lon)


def get_quality_config() -> QualityConfig:
    data = _get_settings().get("quality", {})
    min_auc = int(data.get("min_auc_n_used", 5000))
    min_minutes = int(data.get("min_minutes_covered", 1380))
    return QualityConfig(min_auc_n_used=min_auc, min_minutes_covered=min_minutes)


def get_distance_bins_config() -> DistanceBinsConfig:
    data = _get_settings().get("distance_bins", {})
    raw = data.get("km", [0, 50, 100, 150, 200, 9999])
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        raw = parts
    km = [float(x) for x in raw]
    return DistanceBinsConfig(km=km)


def get_site_latlon() -> tuple[float, float]:
    cfg = get_site_config()
    return cfg.lat, cfg.lon


def get_distance_bins_km() -> list[float]:
    return get_distance_bins_config().km


def get_quality_thresholds() -> tuple[int, int]:
    cfg = get_quality_config()
    return cfg.min_auc_n_used, cfg.min_minutes_covered
