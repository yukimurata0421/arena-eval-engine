from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from arena import cli as arena_cli


def _guard_env(monkeypatch) -> None:
    # cli helpers set env vars directly on os.environ; setenv/delenv makes monkeypatch
    # track and restore the original state at teardown.
    leaked = (
        "ARENA_SCRIPTS_ROOT",
        "ARENA_DATA_DIR",
        "ARENA_OUTPUT_DIR",
        "ARENA_SETTINGS",
        "ARENA_PHASE_CONFIG",
        "ADSB_SETTINGS",
        "ADSB_PHASE_CONFIG",
        "ARENA_ANALYSIS_START_DATE",
        "ARENA_ANALYSIS_END_DATE",
    )
    for k in leaked:
        monkeypatch.setenv(k, "")
    for k in leaked:
        monkeypatch.delenv(k)


def test_cmd_artifacts_verify_propagates_valid_flag_and_errors(capsys, monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    def fake_verify(path: Path) -> dict[str, object]:
        return {
            "valid": False,
            "bundle_sha256": "x" * 64,
            "integrity_summary": {"passed": False},
            "errors": ["artifact_index bundle_sha256 mismatch"],
        }

    monkeypatch.setattr("arena.cli.verify_artifact_bundle", fake_verify)

    args = argparse.Namespace(artifact_bundle=str(bundle_dir))
    rc = arena_cli.cmd_artifacts_verify(args)
    captured = capsys.readouterr()

    assert rc == 1
    assert "valid: 0" in captured.out
    assert "bundle_sha256:" in captured.out
    assert "errors:" in captured.out
    assert "- artifact_index bundle_sha256 mismatch" in captured.out


def test_cmd_artifacts_replay_wraps_exceptions(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    def fake_replay(_path: Path) -> int:
        raise ValueError("broken bundle")

    monkeypatch.setattr("arena.cli.replay_artifact_bundle", fake_replay)
    args = argparse.Namespace(artifact_bundle=str(tmp_path / "missing"))

    rc = arena_cli.cmd_artifacts_replay(args)
    assert rc == 1


def test_cmd_fetch_opensky_returns_error_when_script_missing(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(tmp_path))
    args = argparse.Namespace(
        scripts_root=str(tmp_path),
        data_dir="",
        output_dir="",
        settings="",
        phase_config="",
        analysis_start_date="",
        analysis_end_date="",
    )

    rc = arena_cli.cmd_fetch_opensky(args)
    assert rc == 1


def test_cmd_sync_rpi_logs_returns_error_when_script_missing(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(tmp_path))
    args = argparse.Namespace(
        scripts_root=str(tmp_path),
        data_dir="",
        output_dir="",
        settings="",
        phase_config="",
        analysis_start_date="",
        analysis_end_date="",
        host="",
        user="",
        port=0,
        remote_dir="",
        plao_remote_dir="",
        plao_local_dir="",
        ssh_key="",
        strict_host_key_checking="",
        output_json="",
        dry_run=False,
        fail_on_missing_remote=False,
        skip_plao_sync=False,
    )

    rc = arena_cli.cmd_sync_rpi_logs(args)
    assert rc == 1


def test_cmd_validate_fails_when_settings_missing_required_keys(monkeypatch, cli_env, caplog) -> None:

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])

    # Missing: site/quality/distance_bins
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(path=str(cli_env.settings_path), data={"quality": {}}),
    )

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    with caplog.at_level("INFO"):
        rc = arena_cli.cmd_validate(args)
    assert rc == 1
    assert any("settings.toml に必須キーがありません" in rec.getMessage() for rec in caplog.records)


def test_cmd_validate_warns_when_lat_lon_are_zero(monkeypatch, cli_env, caplog) -> None:

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": 0.0, "lon": 0.0},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    with caplog.at_level("WARNING"):
        rc = arena_cli.cmd_validate(args)
    assert rc == 0
    assert any("site.lat/lon が 0.0 です" in rec.getMessage() for rec in caplog.records)


def test_cmd_validate_creates_data_and_output_dirs(monkeypatch, cli_env) -> None:
    data_dir = cli_env.tmp_path / "data_created"
    output_dir = cli_env.tmp_path / "out_created"
    monkeypatch.setenv("ARENA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_dir))

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": 35.0, "lon": 140.0},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=True,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 0
    assert data_dir.exists()
    assert output_dir.exists()


def test_cmd_run_returns_error_when_config_errors_present(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {"resolved_settings_path": str(tmp_path / "settings.toml"), "resolved_phase_config_path": str(tmp_path / "phases.txt")},
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: ["bad config"])

    called = {"run": 0}
    monkeypatch.setattr(arena_cli.pipeline, "run", lambda _cfg: called.__setitem__("run", called["run"] + 1) or 0)

    args = argparse.Namespace(
        scripts_root="",
        data_dir="",
        output_dir="",
        settings="",
        phase_config="",
        analysis_start_date="",
        analysis_end_date="",
        stage=1,
        only=None,
        dry_run=True,
        no_gpu=True,
        full=False,
        backend="native",
        dynamic_date="",
        no_validate=False,
        validate_only=False,
        skip_existing=False,
        fail_fast=False,
        log_jsonl="",
        skip_plao=False,
        workers=0,
    )

    rc = arena_cli.cmd_run(args)
    assert rc == 1
    assert called["run"] == 0


def test_cmd_run_builds_runconfig_and_passes_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    scripts_root = tmp_path / "scripts"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    scripts_root.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {"resolved_settings_path": str(tmp_path / "settings.toml"), "resolved_phase_config_path": str(tmp_path / "phases.txt")},
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])

    captured: dict[str, object] = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(arena_cli.pipeline, "run", fake_run)

    args = argparse.Namespace(
        scripts_root=str(scripts_root),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        settings="",
        phase_config="",
        analysis_start_date="",
        analysis_end_date="",
        stage=3,
        only=8,
        dry_run=False,
        no_gpu=True,
        full=True,
        backend="wsl",
        dynamic_date="2026-02-14",
        no_validate=True,
        validate_only=False,
        skip_existing=True,
        fail_fast=True,
        log_jsonl="custom.jsonl",
        skip_plao=True,
        workers=5,
    )

    rc = arena_cli.cmd_run(args)
    assert rc == 0

    cfg = captured["cfg"]
    assert cfg.stage == 3
    assert cfg.only == 8
    assert cfg.no_gpu is True
    assert cfg.full is True
    assert cfg.backend == "wsl"
    assert cfg.scripts_root == str(scripts_root.resolve())
    assert cfg.data_root == str(data_dir.resolve())
    assert cfg.output_root == str(output_dir.resolve())
    assert cfg.dynamic_date == "2026-02-14"
    assert cfg.validate is False
    assert cfg.skip_existing is True
    assert cfg.fail_fast is True
    assert cfg.log_jsonl == "custom.jsonl"
    assert cfg.skip_plao is True
    assert cfg.workers == 5


def test_cmd_validate_fails_when_scripts_root_missing(monkeypatch, cli_env) -> None:
    scripts_root = cli_env.tmp_path / "scripts_missing"
    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": 35.0, "lon": 140.0},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 1


def test_cmd_validate_fails_when_scripts_adsb_missing(monkeypatch, cli_env) -> None:
    scripts_root = cli_env.tmp_path / "scripts_no_adsb"
    scripts_root.mkdir()
    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": 35.0, "lon": 140.0},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 1


def test_cmd_validate_fails_when_phases_txt_missing(monkeypatch, cli_env) -> None:
    missing_phase = cli_env.tmp_path / "phases_missing.txt"

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(missing_phase),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": 35.0, "lon": 140.0},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(missing_phase),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 1


def test_cmd_validate_fails_when_settings_path_missing(monkeypatch, cli_env) -> None:
    missing_settings = cli_env.tmp_path / "settings_missing.toml"

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(missing_settings),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(arena_cli, "load_settings", lambda force_reload=True: SimpleNamespace(path=str(missing_settings), data={}))

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(missing_settings),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 1


def test_cmd_validate_fails_when_lat_lon_are_not_numbers(monkeypatch, cli_env) -> None:

    monkeypatch.setattr(
        arena_cli,
        "_resolve_config_metadata",
        lambda _args: {
            "resolved_settings_path": str(cli_env.settings_path),
            "resolved_phase_config_path": str(cli_env.phase_path),
            "used_default_settings": False,
            "used_default_phase_config": False,
            "experimental_mode": False,
        },
    )
    monkeypatch.setattr(arena_cli, "validate_resolved_config_paths", lambda _meta: [])
    monkeypatch.setattr(
        arena_cli,
        "load_settings",
        lambda force_reload=True: SimpleNamespace(
            path=str(cli_env.settings_path),
            data={
                "site": {"lat": "x", "lon": "y"},
                "quality": {"min_auc_n_used": 5, "min_minutes_covered": 10},
                "distance_bins": {"km": [0, 50, 100]},
            },
        ),
    )

    args = argparse.Namespace(
        scripts_root=str(cli_env.scripts_root),
        data_dir=str(cli_env.data_dir),
        output_dir=str(cli_env.output_dir),
        settings=str(cli_env.settings_path),
        phase_config=str(cli_env.phase_path),
        analysis_start_date="",
        analysis_end_date="",
        create_dirs=False,
    )

    rc = arena_cli.cmd_validate(args)
    assert rc == 1


def test_cmd_sync_rpi_logs_builds_expected_cli_args(monkeypatch, tmp_path: Path) -> None:
    _guard_env(monkeypatch)
    scripts_root = tmp_path / "scripts"
    script_path = scripts_root / "adsb" / "ops" / "rpi_log_sync.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('noop')\n", encoding="utf-8", newline="\n")

    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))

    captured: dict[str, object] = {}

    class DummyProc:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode

    def fake_run(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        return DummyProc(0)

    monkeypatch.setattr(arena_cli.subprocess, "run", fake_run)

    args = argparse.Namespace(
        scripts_root=str(scripts_root),
        data_dir="",
        output_dir="",
        settings="",
        phase_config="",
        analysis_start_date="",
        analysis_end_date="",
        host="rpi.local",
        user="pi",
        port=2222,
        remote_dir="/var/logs",
        plao_remote_dir="/plao_pos",
        plao_local_dir=str(tmp_path / "plao_local"),
        ssh_key=str(tmp_path / "id_rsa"),
        strict_host_key_checking="accept-new",
        output_json=str(tmp_path / "sync.json"),
        dry_run=True,
        fail_on_missing_remote=True,
        skip_plao_sync=True,
    )

    rc = arena_cli.cmd_sync_rpi_logs(args)
    assert rc == 0

    cmd = captured["cmd"]
    assert str(script_path) in cmd
    for expected in [
        "--host",
        "rpi.local",
        "--user",
        "pi",
        "--port",
        "2222",
        "--remote-dir",
        "/var/logs",
        "--plao-remote-dir",
        "/plao_pos",
        "--plao-local-dir",
        str(tmp_path / "plao_local"),
        "--ssh-key",
        str(tmp_path / "id_rsa"),
        "--strict-host-key-checking",
        "accept-new",
        "--output-json",
        str(tmp_path / "sync.json"),
        "--dry-run",
        "--fail-on-missing-remote",
        "--skip-plao-sync",
    ]:
        assert expected in cmd

