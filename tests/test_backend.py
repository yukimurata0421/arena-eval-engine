from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arena.pipeline import backend as be


def test_windows_to_wsl_path_converts_drive(monkeypatch) -> None:
    p = Path("E:/arena/scripts")
    assert be._windows_to_wsl_path(p).startswith("/mnt/e/")


def test_backend_build_script_cmd_native(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    out = tmp_path / "out"
    data = tmp_path / "data"
    scripts_root.mkdir()
    out.mkdir()
    data.mkdir()

    b = be.Backend(kind="native", scripts_root_native=scripts_root, output_root_native=out, data_root_native=data, python_native="python")
    cmd, cwd = b.build_script_cmd("adsb/x.py", extra_args=["--a", "1"])
    assert cmd[0] == "python"
    assert str(scripts_root) in cmd[1]
    assert cmd[-2:] == ["--a", "1"]
    assert cwd == str(scripts_root)


def test_backend_build_script_cmd_wsl_includes_pythonpath(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    out = tmp_path / "out"
    data = tmp_path / "data"
    scripts_root.mkdir()
    out.mkdir()
    data.mkdir()

    b = be.Backend(
        kind="wsl",
        scripts_root_native=scripts_root,
        output_root_native=out,
        data_root_native=data,
        scripts_root_exec="/mnt/e/arena/scripts",
        pythonpath_exec="/mnt/e/arena/src",
    )
    cmd, cwd = b.build_script_cmd("adsb/x.py", extra_args=["--a", "1"])
    assert cmd[:3] == ["wsl", "-e", "bash"]
    assert cwd is None
    assert "PYTHONPATH=" in cmd[-1]
    assert "python3" in cmd[-1]
    assert "--a" in cmd[-1]


def test_missing_modules_returns_all_on_subprocess_error(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    out = tmp_path / "out"
    data = tmp_path / "data"
    scripts_root.mkdir()
    out.mkdir()
    data.mkdir()
    b = be.Backend(kind="native", scripts_root_native=scripts_root, output_root_native=out, data_root_native=data)

    def raise_os(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(b, "run_python_snippet", raise_os)
    miss = be.missing_modules(b, ["a", "b"], env={})
    assert miss == ["a", "b"]


def test_wsl_available_returns_false_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(be, "is_windows", lambda: False)
    assert be.wsl_available() is False


def test_detect_gpu_jax_handles_timeout(monkeypatch, tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    out = tmp_path / "out"
    data = tmp_path / "data"
    scripts_root.mkdir()
    out.mkdir()
    data.mkdir()
    b = be.Backend(kind="native", scripts_root_native=scripts_root, output_root_native=out, data_root_native=data)

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(b, "run_python_snippet", raise_timeout)
    info = be.detect_gpu_jax(b, env={})
    assert info["available"] is False
    assert "CPU" in str(info["device"])

