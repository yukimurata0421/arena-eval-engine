from __future__ import annotations

from pathlib import Path

from arena.pipeline.backend import Backend
from arena.pipeline.runner import PipelineRunner
from arena.pipeline.stages import Step


def _build_runner(tmp_path: Path, *, skip_existing: bool = False, validate: bool = True, fail_fast: bool = False) -> tuple[PipelineRunner, Path]:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    backend = Backend(
        kind="native",
        scripts_root_native=scripts_root,
        output_root_native=output_root,
        data_root_native=data_root,
    )
    runner = PipelineRunner(
        backend=backend,
        dry_run=False,
        validate=validate,
        jsonl_log_path=output_root / "performance" / "pipeline_runs.jsonl",
        jax_platforms="cpu",
        skip_existing=skip_existing,
        fail_fast=fail_fast,
        phase_config_path="",
        workers=1,
        steps=[],
    )
    return runner, scripts_root


def test_run_step_skips_when_configured_input_glob_is_missing(tmp_path: Path) -> None:
    runner, _scripts_root = _build_runner(tmp_path, validate=True)

    step = Step(
        stage=7,
        script_rel="plao/analysis/plao_distance_auc_eval.py",
        label="plao",
        skip_if_no_inputs=True,
        input_dir="plao_pos",
        input_pattern="pos_*.jsonl",
    )

    ok = runner.run_step(step)
    assert ok is True
    assert len(runner.records) == 1
    assert runner.records[0].status == "SKIP(no input)"


def test_run_step_skips_existing_outputs(tmp_path: Path) -> None:
    runner, _scripts_root = _build_runner(tmp_path, skip_existing=True, validate=False)

    step = Step(
        stage=1,
        script_rel="dummy_script.py",
        label="dummy",
        expected_outputs=["already_there.txt"],
    )

    runner._should_skip_existing = lambda _step: True  # type: ignore[method-assign]

    ok = runner.run_step(step)
    assert ok is True
    assert len(runner.records) == 1
    assert runner.records[0].status == "SKIP(existing)"


def test_run_step_not_found_returns_false_for_critical_or_fail_fast(tmp_path: Path) -> None:
    runner, _scripts_root = _build_runner(tmp_path, validate=False, fail_fast=False)
    step = Step(
        stage=1,
        script_rel="missing_script.py",
        label="missing",
        critical=True,
    )

    ok = runner.run_step(step)
    assert ok is False
    assert runner.records[0].status == "NOT_FOUND"


def test_execute_step_sets_fail_output_and_soft_warn(monkeypatch, tmp_path: Path) -> None:
    runner, scripts_root = _build_runner(tmp_path, validate=True, fail_fast=False)
    script = scripts_root / "dummy_script.py"
    script.write_text("print('hello')\n", encoding="utf-8")

    step = Step(
        stage=1,
        script_rel=str(script.relative_to(scripts_root)),
        label="dummy",
        expected_outputs=["out.txt"],
        expected_min_bytes=1,
    )

    class DummyCompleted:
        def __init__(self) -> None:
            self.returncode = 0
            self.stderr = ""
            self.stdout = "ok"

    monkeypatch.setattr(
        "arena.pipeline.runner.subprocess.run",
        lambda *args, **kwargs: DummyCompleted(),
    )
    monkeypatch.setattr(
        "arena.pipeline.runner.validate_outputs",
        lambda *_args, **_kwargs: (False, ["missing"]),
    )
    monkeypatch.setattr(
        "arena.pipeline.runner.can_soft_fail",
        lambda *_args, **_kwargs: (True, []),
    )

    ok = runner.run_step(step)
    assert ok is True
    assert len(runner.records) == 1
    assert runner.records[0].status == "WARN"


def test_execute_step_handles_timeout(monkeypatch, tmp_path: Path) -> None:
    runner, scripts_root = _build_runner(tmp_path, validate=False, fail_fast=False)
    script = scripts_root / "dummy_script.py"
    script.write_text("print('hello')\n", encoding="utf-8")

    step = Step(
        stage=1,
        script_rel=str(script.relative_to(scripts_root)),
        label="dummy",
        expected_outputs=[],
        timeout_s=1,
    )

    import arena.pipeline.runner as runner_mod

    def raise_timeout(*_args, **_kwargs):
        raise runner_mod.subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(runner_mod.subprocess, "run", raise_timeout)

    ok = runner.run_step(step)
    assert ok is True
    assert any(rec.status.startswith("WARN") or rec.status.startswith("TIMEOUT") for rec in runner.records)


def test_execute_step_handles_oserror(monkeypatch, tmp_path: Path) -> None:
    runner, scripts_root = _build_runner(tmp_path, validate=False, fail_fast=False)
    script = scripts_root / "dummy_script.py"
    script.write_text("print('hello')\n", encoding="utf-8")

    step = Step(
        stage=1,
        script_rel=str(script.relative_to(scripts_root)),
        label="dummy",
        expected_outputs=[],
    )

    import arena.pipeline.runner as runner_mod

    def raise_oserror(*_args, **_kwargs):
        raise OSError("synthetic error")

    monkeypatch.setattr(runner_mod.subprocess, "run", raise_oserror)
    ok = runner.run_step(step)
    assert ok is True
    assert any(rec.status in ("WARN", "ERROR") for rec in runner.records)
