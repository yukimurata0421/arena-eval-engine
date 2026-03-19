from __future__ import annotations

from arena.lib.geo import haversine_km


def test_haversine_zero_distance() -> None:
    assert haversine_km(35.0, 140.0, 35.0, 140.0) == 0.0


def test_haversine_known_distance_is_reasonable() -> None:
    # Tokyo Station -> Narita Airport is roughly 58-60 km in great-circle distance.
    d_km = haversine_km(35.681236, 139.767125, 35.772043, 140.392852)
    assert 55.0 <= d_km <= 65.0

