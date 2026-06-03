from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena.pipeline import entrypoint
from arena.pipeline.stages import Step


def _steps(*stages: int) -> list[Step]:
    return [Step(stage=s, script_rel=f"s{s}.py", label=f"S{s}") for s in stages]


def test_run_pipeline_stages_uses_parallel_stage_group_and_skips_individual_stages(monkeypatch) -> None:
    runner = SimpleNamespace()
    steps = _steps(1, 2, 3, 4, 5, 7, 8)

    called_single: list[int] = []
    called_group: list[tuple[int, ...]] = []

    def should_run_stage(n: int) -> bool:
        return n >= 1

    def fake_launch_early(*_args, **_kwargs):
        return {}, None, set()

    def fake_run_stage_group(
        _runner,
        _steps,
        group_stages,
        _should_run_stage,
        _resolved_workers,
        _fail_fast,
        _current_st,
        exclude_stages=None,
    ):
        called_group.append(tuple(group_stages))
        return True

    def fake_run_single_stage(_runner, _steps, st: int, _resolved_workers: int, _fail_fast: bool):
        called_single.append(st)
        return True

    monkeypatch.setattr(entrypoint, "_launch_early_stages", fake_launch_early)
    monkeypatch.setattr(entrypoint, "_run_stage_group", fake_run_stage_group)
    monkeypatch.setattr(entrypoint, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(entrypoint, "_collect_early", lambda *_args, **_kwargs: True)

    ok = entrypoint._run_pipeline_stages(
        runner=runner,
        steps=steps,
        cfg=SimpleNamespace(fail_fast=False),
        should_run_stage=should_run_stage,
        resolved_workers=4,
    )

    assert ok is True
    assert called_group == [entrypoint.PARALLEL_STAGE_GROUPS[0]]
    # Stage 2/4/5/7/8 should be executed by group; Stage 3 remains individual.
    assert called_single == [1, 3]


def test_run_pipeline_stages_skips_early_launched_stages(monkeypatch) -> None:
    runner = SimpleNamespace()
    steps = _steps(1, 2, 7, 8)

    called_single: list[int] = []
    launched_early = {7}

    def should_run_stage(n: int) -> bool:
        return True

    monkeypatch.setattr(entrypoint, "_launch_early_stages", lambda *_args, **_kwargs: ({}, None, launched_early))
    monkeypatch.setattr(
        entrypoint,
        "_run_stage_group",
        lambda *_args, **_kwargs: pytest.fail("_run_stage_group should not be called with workers=1"),
    )

    def fake_run_single_stage(_runner, _steps, st: int, _resolved_workers: int, _fail_fast: bool):
        called_single.append(st)
        return True

    monkeypatch.setattr(entrypoint, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(entrypoint, "_collect_early", lambda *_args, **_kwargs: True)

    ok = entrypoint._run_pipeline_stages(
        runner=runner,
        steps=steps,
        cfg=SimpleNamespace(fail_fast=False),
        should_run_stage=should_run_stage,
        # Use 1 to bypass PARALLEL_STAGE_GROUPS so we only test early-stage skipping here.
        resolved_workers=1,
    )

    assert ok is True
    assert 7 not in called_single
    assert set(called_single) == {1, 2, 8}


def test_run_pipeline_stages_aborts_and_collects_early_on_group_failure(monkeypatch) -> None:
    runner = SimpleNamespace()
    steps = _steps(1, 2, 3, 7)

    collect_called: list[bool] = []

    def should_run_stage(n: int) -> bool:
        return True

    monkeypatch.setattr(entrypoint, "_launch_early_stages", lambda *_args, **_kwargs: ({"F": "S"}, object(), {7}))

    def fake_collect_early(*_args, **_kwargs):
        collect_called.append(True)
        return True

    monkeypatch.setattr(entrypoint, "_collect_early", fake_collect_early)
    monkeypatch.setattr(entrypoint, "_run_stage_group", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(entrypoint, "_run_single_stage", lambda *_args, **_kwargs: True)

    ok = entrypoint._run_pipeline_stages(
        runner=runner,
        steps=steps,
        cfg=SimpleNamespace(fail_fast=False),
        should_run_stage=should_run_stage,
        resolved_workers=4,
    )

    assert ok is False
    assert collect_called, "expected early futures to be collected before abort"


def test_run_pipeline_stages_aborts_after_critical_single_stage_failure(monkeypatch) -> None:
    runner = SimpleNamespace()
    steps = [
        Step(stage=1, script_rel="critical.py", label="critical", critical=True),
        Step(stage=3, script_rel="later.py", label="later"),
    ]
    collect_called: list[bool] = []
    called_single: list[int] = []

    def should_run_stage(_n: int) -> bool:
        return True

    def fake_run_single_stage(_runner, _steps, st: int, _resolved_workers: int, _fail_fast: bool):
        called_single.append(st)
        return st != 1

    def fake_collect_early(*_args, **_kwargs):
        collect_called.append(True)
        return True

    monkeypatch.setattr(entrypoint, "_launch_early_stages", lambda *_args, **_kwargs: ({}, None, set()))
    monkeypatch.setattr(entrypoint, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(entrypoint, "_collect_early", fake_collect_early)

    ok = entrypoint._run_pipeline_stages(
        runner=runner,
        steps=steps,
        cfg=SimpleNamespace(fail_fast=False),
        should_run_stage=should_run_stage,
        resolved_workers=1,
    )

    assert ok is False
    assert called_single == [1]
    assert collect_called
