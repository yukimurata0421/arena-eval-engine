from __future__ import annotations

from arena.pipeline.stages import build_pipeline


def test_stage_s1_06_uses_logical_data_output_path() -> None:
    steps = build_pipeline(dynamic_date="2026-02-14", full_mode=False, skip_plao=False)
    opensky = next(s for s in steps if s.script_rel == "adsb/data_fetch/get_opensky_traffic.py")
    assert opensky.expected_outputs == ["data://flight_data/airport_movements.csv"]
