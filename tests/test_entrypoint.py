from __future__ import annotations

from pathlib import Path

import pytest

from arena.lib.runtime_config import clear_settings_cache
from arena.pipeline import entrypoint
from arena.pipeline.entrypoint import run
from arena.pipeline.stages import RunConfig


def test_entrypoint_run_dry_run_with_custom_roots(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    config_dir = scripts_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.toml").write_text(
        "[site]\nlat = 35.0\nlon = 140.0\n\n[quality]\nmin_auc_n_used = 5\nmin_minutes_covered = 10\n",
        encoding="utf-8",
    )
    (config_dir / "phases.txt").write_text(
        "[events]\n2026-01-01 = Init\n\n[settings]\nintervention_date = 2026-02-14\n",
        encoding="utf-8",
    )

    # run() sets env vars directly on os.environ; use setenv so monkeypatch
    # tracks AND restores them at teardown (delenv on missing keys is a no-op).
    leaked_vars = ("ARENA_SETTINGS", "ADSB_SETTINGS", "ARENA_PHASE_CONFIG", "ADSB_PHASE_CONFIG")
    for var in leaked_vars:
        monkeypatch.setenv(var, "")
    for var in leaked_vars:
        monkeypatch.delenv(var)

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

    clear_settings_cache()


def test_entrypoint_run_returns_error_on_config_resolution_failure(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

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

    leaked_vars = ("ARENA_SETTINGS", "ADSB_SETTINGS", "ARENA_PHASE_CONFIG", "ADSB_PHASE_CONFIG")
    for var in leaked_vars:
        monkeypatch.setenv(var, "")
    for var in leaked_vars:
        monkeypatch.delenv(var)

    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_runtime_config_metadata",
        lambda **kwargs: {"resolved_settings_path": scripts_root / "missing_settings.toml", "resolved_phase_config_path": scripts_root / "missing_phases.txt"},
    )
    monkeypatch.setattr("arena.pipeline.entrypoint.validate_resolved_config_paths", lambda meta: ["settings missing"])

    rc = run(cfg)
    assert rc == 1


def test_entrypoint_run_fails_when_required_modules_missing(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    config_dir = scripts_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.toml").write_text(
        "[site]\nlat = 35.0\nlon = 140.0\n\n[quality]\nmin_auc_n_used = 5\nmin_minutes_covered = 10\n",
        encoding="utf-8",
    )
    (config_dir / "phases.txt").write_text(
        "[events]\n2026-01-01 = Init\n\n[settings]\nintervention_date = 2026-02-14\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: ["numpy"])
    cfg = RunConfig(
        only=1,
        dry_run=False,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
    )

    leaked_vars = ("ARENA_SETTINGS", "ADSB_SETTINGS", "ARENA_PHASE_CONFIG", "ADSB_PHASE_CONFIG")
    for var in leaked_vars:
        monkeypatch.setenv(var, "")
    for var in leaked_vars:
        monkeypatch.delenv(var)

    rc = run(cfg)
    assert rc == 1


def test_entrypoint_run_uses_dynamic_date_fallback_on_phase_config_error(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    config_dir = scripts_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.toml").write_text(
        "[site]\nlat = 35.0\nlon = 140.0\n\n[quality]\nmin_auc_n_used = 5\nmin_minutes_covered = 10\n",
        encoding="utf-8",
    )
    (config_dir / "phases.txt").write_text(
        "[events]\n2026-01-01 = Init\n\n[settings]\nintervention_date = 2026-02-14\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("arena.pipeline.entrypoint.load_phase_config", lambda path: (_ for _ in ()).throw(OSError("broken phases")))
    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])

    cfg = RunConfig(
        only=1,
        dry_run=True,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="",
        validate=False,
    )

    leaked_vars = ("ARENA_SETTINGS", "ADSB_SETTINGS", "ARENA_PHASE_CONFIG", "ADSB_PHASE_CONFIG")
    for var in leaked_vars:
        monkeypatch.setenv(var, "")
    for var in leaked_vars:
        monkeypatch.delenv(var)

    rc = run(cfg)
    assert rc == 0


@pytest.mark.parametrize("planned_has_stage5,contract_ok,expected_rc", [(True, True, 0), (True, False, 1), (False, False, 0)])
def test_entrypoint_run_checks_change_point_contract(monkeypatch, tmp_path: Path, planned_has_stage5: bool, contract_ok: bool, expected_rc: int) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    config_dir = scripts_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.toml").write_text(
        "[site]\nlat = 35.0\nlon = 140.0\n\n[quality]\nmin_auc_n_used = 5\nmin_minutes_covered = 10\n",
        encoding="utf-8",
    )
    (config_dir / "phases.txt").write_text(
        "[events]\n2026-01-01 = Init\n\n[settings]\nintervention_date = 2026-02-14\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("arena.pipeline.entrypoint.missing_modules", lambda backend, modules, env: [])
    monkeypatch.setattr("arena.pipeline.entrypoint._check_change_point_contract", lambda output_root_native, data_root_native: contract_ok)

    class DummyStep:
        def __init__(self, stage: int) -> None:
            self.stage = stage
            self.label = f"S{stage}"
            self.critical = False
            self.est_s = 1
            self.script_rel = "dummy.py"
            self.expected_outputs = []
            self.expected_min_bytes = 0

    steps = [DummyStep(1), DummyStep(5)] if planned_has_stage5 else [DummyStep(1)]
    monkeypatch.setattr(
        "arena.pipeline.entrypoint.build_pipeline",
        lambda *args, **kwargs: steps,
    )
    monkeypatch.setattr(
        "arena.pipeline.entrypoint.PipelineRunner",
        lambda **kwargs: type(
            "DummyRunner",
            (),
            {
                "records": [],
                "log_config_snapshot": lambda self, snapshot: None,
                "print_summary": lambda self: None,
                "write_error_code_report": lambda self: tmp_path / "dummy.txt",
                "run_step": lambda self, step: True,
            },
        )(),
    )

    cfg = RunConfig(
        stage=1,
        only=None,
        dry_run=False,
        no_gpu=True,
        backend="native",
        scripts_root=str(scripts_root),
        output_root=str(output_root),
        data_root=str(data_root),
        dynamic_date="2026-02-14",
        validate=False,
        validate_only=False,
        skip_existing=False,
        fail_fast=False,
        log_jsonl="",
        skip_plao=False,
        workers=1,
    )

    leaked_vars = ("ARENA_SETTINGS", "ADSB_SETTINGS", "ARENA_PHASE_CONFIG", "ADSB_PHASE_CONFIG")
    for var in leaked_vars:
        monkeypatch.setenv(var, "")
    for var in leaked_vars:
        monkeypatch.delenv(var)

    rc = entrypoint.run(cfg)
    assert rc == expected_rc
