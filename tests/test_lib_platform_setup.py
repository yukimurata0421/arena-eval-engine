from __future__ import annotations

import os
import types

from arena.lib import platform_setup as ps


def test_resolve_workers_respects_env_and_cpu(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("ADSB_MAX_WORKERS", "16")
    assert ps.resolve_workers(default_cap=12) == 8

    monkeypatch.setenv("ADSB_MAX_WORKERS", "3")
    assert ps.resolve_workers(default_cap=12) == 3

    monkeypatch.setenv("ADSB_MAX_WORKERS", "bad")
    assert ps.resolve_workers(default_cap=6) == 6


def test_init_numpyro_platform_force_cpu(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_numpyro = types.SimpleNamespace(
        set_platform=lambda x: calls.append(("set_platform", x)),
        set_host_device_count=lambda x: calls.append(("set_host_device_count", x)),
    )
    monkeypatch.setitem(__import__("sys").modules, "numpyro", fake_numpyro)

    platform = ps.init_numpyro_platform(n_data=10, force_cpu=True)
    assert platform == "cpu"
    assert ("set_platform", "cpu") in calls


def test_init_numpyro_platform_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_numpyro = types.SimpleNamespace(
        set_platform=lambda x: calls.append(("set_platform", x)),
        set_host_device_count=lambda x: calls.append(("set_host_device_count", x)),
    )
    fake_jax = types.SimpleNamespace(devices=lambda _kind: (_ for _ in ()).throw(RuntimeError("no cuda")))

    monkeypatch.setitem(__import__("sys").modules, "numpyro", fake_numpyro)
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    monkeypatch.setattr(ps, "_link_nvidia_dlls", lambda: None)

    platform = ps.init_numpyro_platform(n_data=ps.GPU_THRESHOLD + 1, force_cpu=False)
    assert platform == "cpu"
    assert ("set_platform", "cpu") in calls


def test_init_numpyro_platform_respects_gpu_min_n_override(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_numpyro = types.SimpleNamespace(
        set_platform=lambda x: calls.append(("set_platform", x)),
        set_host_device_count=lambda x: calls.append(("set_host_device_count", x)),
    )
    monkeypatch.setitem(__import__("sys").modules, "numpyro", fake_numpyro)
    monkeypatch.setenv("ADSB_GPU_MIN_N", "200")

    platform = ps.init_numpyro_platform(n_data=199, force_cpu=False)

    assert platform == "cpu"
    assert ("set_platform", "cpu") in calls
