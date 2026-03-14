from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from arena.pipeline.entrypoint import run
from arena.pipeline.stages import RunConfig, Step


def test_entrypoint_run_dry_run_with_custom_roots(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    cfg = RunConfig(
        only=1,
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)
    assert rc == 0
    assert (output_root / "performance" / "pipeline_runs.jsonl").exists()


def test_entrypoint_reports_missing_modules(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: ["numpyro"])
    cfg = RunConfig(
        dry_run=False,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
    )

    rc = run(cfg)

    assert rc == 1


def test_entrypoint_rejects_invalid_log_jsonl_mode(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        log_jsonl_mode="invalid",
    )

    rc = run(cfg)

    assert rc == 1


def test_entrypoint_handles_validate_only_failure(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [
            Step(stage=1, script_rel="adsb/analysis/check.py", label="check", expected_outputs=["missing.txt"])
        ],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr(
        "arena.pipeline.entrypoint.validate_outputs",
        lambda *args, **kwargs: (False, ["missing.txt"]),
    )
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate_only=True,
    )

    rc = run(cfg)

    assert rc == 1


def test_entrypoint_handles_stage_selection_edge_case(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [
            Step(stage=1, script_rel="adsb/analysis/one.py", label="one"),
            Step(stage=2, script_rel="adsb/analysis/two.py", label="two"),
        ],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    cfg = RunConfig(
        only=99,
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 0
    assert (output_root / "performance" / "pipeline_runs.jsonl").exists()


@dataclass
class _DummyRunner:
    records: list = field(default_factory=list)

    def __init__(self, backend, dry_run, validate, jsonl_log_path, jax_platforms, skip_existing, fail_fast, phase_config_path="", workers=0, steps=None):
        self.backend = backend
        self.jsonl_log_path = jsonl_log_path
        self.records = []

    def log_config_snapshot(self, snapshot: dict) -> None:
        self.jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_log_path.write_text("", encoding="utf-8")

    def run_step(self, step: Step) -> bool:
        return True

    def print_summary(self) -> None:
        return None

    def write_error_code_report(self) -> Path:
        report_path = self.backend.output_root_native / "performance" / "pipeline_error_codes_latest.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("NO_ISSUES\n", encoding="utf-8")
        return report_path


class _CapturingRunner(_DummyRunner):
    last_instance = None
    run_calls: list[str]

    def __init__(self, backend, dry_run, validate, jsonl_log_path, jax_platforms, skip_existing, fail_fast, phase_config_path="", workers=0, steps=None):
        super().__init__(backend, dry_run, validate, jsonl_log_path, jax_platforms, skip_existing, fail_fast, phase_config_path=phase_config_path, workers=workers, steps=steps)
        self.jax_platforms = jax_platforms
        self.fail_fast = fail_fast
        self.run_calls = []
        _CapturingRunner.last_instance = self

    def run_step(self, step: Step) -> bool:
        self.run_calls.append(step.label)
        return True


class _FailFastRunner(_CapturingRunner):
    def run_step(self, step: Step) -> bool:
        self.run_calls.append(step.label)
        return step.label != "first"


class _IssueRecordingRunner(_CapturingRunner):
    def run_step(self, step: Step) -> bool:
        self.run_calls.append(step.label)
        self.records.append(
            SimpleNamespace(
                status="FAIL",
            )
        )
        return True


def test_entrypoint_fails_stage5_contract_when_required_outputs_missing(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [Step(stage=5, script_rel="adsb/analysis/change.py", label="change-point")],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _DummyRunner)
    monkeypatch.setattr(
        "arena.pipeline.entrypoint.validate_outputs",
        lambda *args, **kwargs: (False, ["change_point/change_point_report.txt"]),
    )
    cfg = RunConfig(
        only=5,
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 1
    contract_path = output_root / "performance" / "change_point_contract_latest.txt"
    assert contract_path.exists()
    assert "change_point/change_point_report.txt" in contract_path.read_text(encoding="utf-8")


def test_entrypoint_auto_backend_prefers_expected_backend(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.is_windows", lambda: True)
    monkeypatch.setattr("arena.pipeline.entrypoint.wsl_available", lambda: True)
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _CapturingRunner)
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="auto",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 0
    assert _CapturingRunner.last_instance.backend.kind == "wsl"


def test_entrypoint_handles_wsl_backend_resolution(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.default_roots_exec_for_wsl", lambda s, o, d: ("/wsl/scripts", "/wsl/output", "/wsl/data"))
    monkeypatch.setattr("arena.pipeline.entrypoint._windows_to_wsl_path", lambda p: "/wsl/src")
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _CapturingRunner)
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="wsl",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 0
    backend = _CapturingRunner.last_instance.backend
    assert backend.kind == "wsl"
    assert backend.scripts_root_exec == "/wsl/scripts"
    assert backend.pythonpath_exec == "/wsl/src"


def test_entrypoint_handles_gpu_probe_failure_gracefully(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.detect_gpu_jax", lambda backend, env: (_ for _ in ()).throw(RuntimeError("probe boom")))
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _CapturingRunner)
    cfg = RunConfig(
        dry_run=False,
        no_gpu=False,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 0
    assert _CapturingRunner.last_instance.jax_platforms == "cpu"


def test_entrypoint_uses_dynamic_date_fallback_when_missing(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()
    captured: dict[str, str] = {}

    def capture_build_pipeline(dynamic_date, full_mode, skip_plao):
        captured["dynamic_date"] = dynamic_date
        return []

    monkeypatch.setattr("arena.pipeline.entrypoint.load_phase_config", lambda path: (_ for _ in ()).throw(RuntimeError("missing phase")))
    monkeypatch.setattr("arena.pipeline.entrypoint.build_pipeline", capture_build_pipeline)
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _CapturingRunner)
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="",
        validate=False,
    )

    rc = run(cfg)

    assert rc == 0
    assert captured["dynamic_date"] == "2026-02-11"


def test_entrypoint_fail_fast_stops_after_first_hard_failure(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [
            Step(stage=1, script_rel="adsb/analysis/first.py", label="first"),
            Step(stage=1, script_rel="adsb/analysis/second.py", label="second"),
        ],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _FailFastRunner)
    cfg = RunConfig(
        dry_run=False,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
        fail_fast=True,
    )

    rc = run(cfg)

    assert rc == 1
    assert _FailFastRunner.last_instance.run_calls == ["first"]


def test_entrypoint_validate_only_success(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [Step(stage=1, script_rel="adsb/analysis/check.py", label="check", expected_outputs=[])],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    cfg = RunConfig(
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate_only=True,
    )

    rc = run(cfg)

    assert rc == 0


def test_entrypoint_post_run_ng_warning(monkeypatch, tmp_path: Path, capsys) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda **kwargs: [Step(stage=1, script_rel="adsb/analysis/check.py", label="check")],
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint.build_snapshot", lambda phase_config_path: {})
    monkeypatch.setattr("arena.pipeline.entrypoint.PipelineRunner", _IssueRecordingRunner)
    cfg = RunConfig(
        dry_run=False,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    rc = run(cfg)

    captured = capsys.readouterr()
    assert rc == 1
    assert "WARNING: 1 failures detected." in captured.out
