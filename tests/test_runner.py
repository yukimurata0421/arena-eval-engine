from __future__ import annotations

import json
from pathlib import Path
from subprocess import TimeoutExpired
from types import SimpleNamespace

from arena.pipeline.backend import Backend
from arena.pipeline.runner import PipelineRunner
from arena.pipeline.stages import RunRecord, Step


def test_run_step_skips_when_configured_input_glob_is_missing(tmp_path: Path) -> None:
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
        validate=True,
        jsonl_log_path=output_root / "performance" / "pipeline_runs.jsonl",
        jax_platforms="cpu",
        skip_existing=False,
        fail_fast=False,
        phase_config_path="",
        workers=1,
        steps=[],
    )

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


def _make_runner(tmp_path: Path, *, validate: bool = False, fail_fast: bool = False) -> tuple[PipelineRunner, Path]:
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
        skip_existing=False,
        fail_fast=fail_fast,
        phase_config_path="",
        workers=1,
        steps=[],
    )
    return runner, scripts_root


def test_runner_records_stage_failure(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "fail.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(stage=3, script_rel="adsb/analysis/fail.py", label="failing step", error_code_base="S3-09")

    monkeypatch.setattr(
        "arena.pipeline.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stderr="boom\n", stdout=""),
    )

    ok = runner.run_step(step)

    assert ok is True
    assert runner.records[-1].status == "FAIL"
    assert runner.records[-1].error_code == "S3-09-E10"
    payloads = [
        json.loads(line)
        for line in runner.jsonl_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["status"] == "FAIL"
    assert payloads[-1]["error_code"] == "S3-09-E10"


def test_runner_handles_validate_outputs_failure(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path, validate=True)
    script_path = scripts_root / "adsb" / "analysis" / "validate.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(
        stage=4,
        script_rel="adsb/analysis/validate.py",
        label="validate outputs",
        critical=True,
        error_code_base="S4-02",
        expected_outputs=["performance/result.csv"],
    )

    monkeypatch.setattr(
        "arena.pipeline.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )
    monkeypatch.setattr(
        "arena.pipeline.runner.validate_outputs",
        lambda *args, **kwargs: (False, ["performance/result.csv (too small: 12 bytes)"]),
    )

    ok = runner.run_step(step)

    assert ok is False
    assert runner.records[-1].status == "FAIL_OUTPUT"
    assert runner.records[-1].missing_outputs == ["performance/result.csv (too small: 12 bytes)"]
    assert runner.records[-1].error_code == "S4-02-E22"


def test_runner_preserves_audit_log_on_failure(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "explode.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(stage=5, script_rel="adsb/analysis/explode.py", label="explode", error_code_base="S5-03")

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("arena.pipeline.runner.subprocess.run", raise_runtime_error)

    ok = runner.run_step(step)

    assert ok is True
    assert runner.records[-1].status == "ERROR"
    assert runner.records[-1].stderr_tail == "synthetic failure"
    payloads = [
        json.loads(line)
        for line in runner.jsonl_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["status"] == "ERROR"
    assert payloads[-1]["stderr_tail"] == "synthetic failure"


def test_runner_handles_partial_success_consistently(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "soft_fail.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(
        stage=1,
        script_rel="adsb/analysis/soft_fail.py",
        label="soft fail",
        error_code_base="S1-06",
        soft_fail_on_error=True,
    )

    monkeypatch.setattr(
        "arena.pipeline.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="rate limit reached\n", stdout=""),
    )
    monkeypatch.setattr(PipelineRunner, "_can_soft_fail", lambda self, step, min_mtime: (True, []))

    ok = runner.run_step(step)

    assert ok is True
    assert runner.records[-1].status == "WARN"
    assert runner.records[-1].outputs_ok is True
    assert runner.records[-1].missing_outputs == []
    assert runner.records[-1].error_code == "S1-06-W02"


def test_runner_classifies_timeout_failure(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "timeout.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(
        stage=2,
        script_rel="adsb/analysis/timeout.py",
        label="timeout step",
        critical=True,
        timeout_s=7,
        error_code_base="S2-03",
    )

    def raise_timeout(*args, **kwargs):
        raise TimeoutExpired(cmd=["python", "timeout.py"], timeout=7)

    monkeypatch.setattr("arena.pipeline.runner.subprocess.run", raise_timeout)

    ok = runner.run_step(step)

    assert ok is False
    assert runner.records[-1].status == "TIMEOUT"
    assert runner.records[-1].error_code == "S2-03-E31"
    assert runner.records[-1].returncode is None


def test_runner_preserves_audit_log_on_timeout(tmp_path: Path, monkeypatch) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "timeout_log.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(stage=6, script_rel="adsb/analysis/timeout_log.py", label="timeout log", timeout_s=3, error_code_base="S6-01")

    def raise_timeout(*args, **kwargs):
        raise TimeoutExpired(cmd=["python", "timeout_log.py"], timeout=3)

    monkeypatch.setattr("arena.pipeline.runner.subprocess.run", raise_timeout)

    ok = runner.run_step(step)

    assert ok is True
    payloads = [
        json.loads(line)
        for line in runner.jsonl_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["status"] == "TIMEOUT"
    assert payloads[-1]["error_code"] == "S6-01-E31"


def test_runner_classifies_not_found_failure(tmp_path: Path) -> None:
    runner, _scripts_root = _make_runner(tmp_path, fail_fast=True)
    step = Step(
        stage=8,
        script_rel="adsb/analysis/missing.py",
        label="missing script",
        critical=False,
        error_code_base="S8-02",
    )

    ok = runner.run_step(step)

    assert ok is False
    assert runner.records[-1].status == "NOT_FOUND"
    assert runner.records[-1].error_code == "S8-02-E32"
    assert runner.records[-1].missing_outputs[0].endswith("adsb\\analysis\\missing.py")


def test_runner_records_skip_existing_consistently(tmp_path: Path, monkeypatch) -> None:
    runner, _scripts_root = _make_runner(tmp_path)
    step = Step(
        stage=3,
        script_rel="adsb/analysis/existing.py",
        label="existing outputs",
        expected_outputs=["performance/existing.csv"],
        error_code_base="S3-05",
    )
    monkeypatch.setattr(PipelineRunner, "_should_skip_existing", lambda self, step: True)

    ok = runner.run_step(step)

    assert ok is True
    assert runner.records[-1].status == "SKIP(existing)"
    assert runner.records[-1].error_code == ""
    payloads = [
        json.loads(line)
        for line in runner.jsonl_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["status"] == "SKIP(existing)"
    assert payloads[-1]["error_code"] == ""


def test_runner_failure_summary_formatting_remains_stable(tmp_path: Path, capsys) -> None:
    runner, _scripts_root = _make_runner(tmp_path)
    runner.records.extend(
        [
            RunRecord(
                ts_start="t0",
                ts_end="t1",
                backend="native",
                stage=2,
                label="timeout",
                script_rel="timeout.py",
                status="TIMEOUT",
                elapsed_s=3.0,
                returncode=None,
                cmd=["python", "timeout.py"],
                expected_outputs=[],
                outputs_ok=False,
                missing_outputs=[],
                step_code="S2-03",
            ),
            RunRecord(
                ts_start="t0",
                ts_end="t1",
                backend="native",
                stage=4,
                label="missing",
                script_rel="missing.py",
                status="NOT_FOUND",
                elapsed_s=0.0,
                returncode=None,
                cmd=["python", "missing.py"],
                expected_outputs=[],
                outputs_ok=False,
                missing_outputs=["missing.py"],
                step_code="S4-08",
            ),
        ]
    )

    runner.print_summary()
    report_path = runner.write_error_code_report()

    captured = capsys.readouterr()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Pipeline Summary" in captured.out
    assert "Error Code Details" in captured.out
    assert "S2-03-E31" in captured.out
    assert "S4-08-E32" in captured.out
    assert "Detected Issues" in report_text
    assert "S2-03-E31 | TIMEOUT" in report_text
    assert "S4-08-E32 | NOT_FOUND" in report_text


def test_runner_timeout_soft_fail_warn(tmp_path: Path, monkeypatch, capsys) -> None:
    runner, scripts_root = _make_runner(tmp_path)
    script_path = scripts_root / "adsb" / "analysis" / "timeout_soft.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('placeholder')\n", encoding="utf-8")
    step = Step(
        stage=1,
        script_rel="adsb/analysis/timeout_soft.py",
        label="timeout soft fail",
        timeout_s=5,
        error_code_base="S1-09",
        soft_fail_on_error=True,
    )

    def raise_timeout(*args, **kwargs):
        raise TimeoutExpired(cmd=["python", "timeout_soft.py"], timeout=5)

    monkeypatch.setattr("arena.pipeline.runner.subprocess.run", raise_timeout)
    monkeypatch.setattr(PipelineRunner, "_can_soft_fail", lambda self, step, min_mtime: (True, []))

    ok = runner.run_step(step)

    assert ok is True
    assert runner.records[-1].status == "WARN"
    assert runner.records[-1].error_code == "S1-09-W99"
    payloads = [
        json.loads(line)
        for line in runner.jsonl_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[-1]["status"] == "WARN"
    assert payloads[-1]["error_code"] == "S1-09-W99"

    runner.print_summary()
    captured = capsys.readouterr()
    assert "WARN" in captured.out
