from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from arena.pipeline import entrypoint
from arena.pipeline.stages import Step


def test_run_validate_only_returns_1_on_missing_outputs(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    out.mkdir()
    data.mkdir()

    steps = [
        Step(stage=1, script_rel="a.py", label="A", expected_outputs=["x.txt"], expected_min_bytes=1),
        Step(stage=1, script_rel="b.py", label="B", expected_outputs=[]),
    ]

    monkeypatch.setattr(entrypoint, "validate_outputs", lambda *_args, **_kwargs: (False, ["missing1", "missing2", "missing3", "missing4"]))
    rc = entrypoint._run_validate_only(steps, out, data)
    assert rc == 1


def test_validate_stage1_wave_indices_warns_on_missing_keywords(caplog) -> None:
    steps = [
        Step(stage=1, wave=1, script_rel="adsb/aggregators/adsb_aggregator.py", label="agg"),
        Step(stage=1, wave=2, script_rel="adsb/ops/dist_1m_health_check.py", label="health"),
    ]
    with caplog.at_level("WARNING"):
        entrypoint._validate_stage1_wave_indices(steps)
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "expected keyword" in msgs


def test_run_stage_group_returns_none_when_only_one_stage_to_run(monkeypatch) -> None:
    runner = SimpleNamespace()
    steps = [Step(stage=2, script_rel="s2.py", label="s2", est_s=1)]

    def should_run_stage(n: int) -> bool:
        return n == 3  # only stage 3 planned, but group starts at 2

    rc = entrypoint._run_stage_group(
        runner,
        steps,
        group_stages=(2, 3, 4),
        should_run_stage=should_run_stage,
        resolved_workers=4,
        fail_fast=False,
        current_st=2,
        exclude_stages=set(),
    )
    assert rc is None


def test_check_change_point_contract_writes_contract_file(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    out.mkdir()
    data.mkdir()

    monkeypatch.setattr(entrypoint, "validate_outputs", lambda *_args, **_kwargs: (False, ["a", "b"]))
    ok = entrypoint._check_change_point_contract(out, data)
    assert ok is False

    contract = out / "performance" / "change_point_contract_latest.txt"
    assert contract.exists()
    text = contract.read_text(encoding="utf-8")
    assert "change_point_required_ok: 0" in text
    assert "missing_outputs:" in text

