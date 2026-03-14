from __future__ import annotations

from pathlib import Path

from arena.lib import config as arena_config
from arena.lib import paths
from arena.lib.phase_config import clear_config_cache, get_config, load_phase_config
from arena.lib.runtime_config import clear_settings_cache, load_settings
from arena.lib.settings_loader import find_scripts_root, find_settings_path
from arena.pipeline.backend import default_roots_native


def test_runtime_config_uses_env_settings_file(tmp_path: Path, monkeypatch) -> None:
    settings = tmp_path / "settings.toml"
    settings.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 35.0",
                "lon = 140.0",
                "",
                "[quality]",
                "min_auc_n_used = 123",
                "min_minutes_covered = 456",
                "",
                "[distance_bins]",
                "km = [0, 10, 20, 9999]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_SETTINGS", str(settings))
    clear_settings_cache()

    snap = load_settings(force_reload=True)
    assert Path(snap.path) == settings
    assert snap.data["quality"]["min_auc_n_used"] == 123


def test_config_reads_values_from_runtime_settings(tmp_path: Path, monkeypatch) -> None:
    settings = tmp_path / "settings.toml"
    settings.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 1.5",
                "lon = 2.5",
                "",
                "[quality]",
                "min_auc_n_used = 10",
                "min_minutes_covered = 20",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_SETTINGS", str(settings))
    clear_settings_cache()

    site = arena_config.get_site_config()
    quality = arena_config.get_quality_config()
    assert site.lat == 1.5
    assert site.lon == 2.5
    assert quality.min_auc_n_used == 10
    assert quality.min_minutes_covered == 20


def test_phase_config_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    phase_file = tmp_path / "phases.txt"
    phase_file.write_text(
        "\n".join(
            [
                "[events]",
                "2026-01-01 = Init | rtl-sdr | #111111",
                "",
                "[settings]",
                "intervention_date = 2026-02-20",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_PHASE_CONFIG", str(phase_file))

    cfg = load_phase_config()
    assert Path(cfg.config_path) == phase_file
    assert cfg.intervention_date == "2026-02-20"


def test_paths_resolve_from_updated_settings_path_without_reimport(tmp_path: Path, monkeypatch) -> None:
    settings1 = tmp_path / "settings1.toml"
    settings2 = tmp_path / "settings2.toml"
    data1 = tmp_path / "data1"
    data2 = tmp_path / "data2"
    settings1.write_text(f"[paths]\ndata_dir = '{data1.as_posix()}'\n", encoding="utf-8")
    settings2.write_text(f"[paths]\ndata_dir = '{data2.as_posix()}'\n", encoding="utf-8")

    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARENA_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("ARENA_SETTINGS", str(settings1))
    clear_settings_cache()
    assert paths.resolve_data_dir() == data1

    monkeypatch.setenv("ARENA_SETTINGS", str(settings2))
    assert paths.resolve_data_dir() == data2


def test_backend_default_roots_reflect_env_overrides(tmp_path: Path, monkeypatch) -> None:
    scripts_root = tmp_path / "scripts"
    data_dir = tmp_path / "data_override"
    output_dir = tmp_path / "output_override"

    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))
    monkeypatch.setenv("ARENA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_dir))

    sr, out, data = default_roots_native()
    assert sr == scripts_root
    assert out == output_dir
    assert data == data_dir


def test_phase_config_get_config_reloads_when_env_path_changes(tmp_path: Path, monkeypatch) -> None:
    phase1 = tmp_path / "phases1.txt"
    phase1.write_text("[settings]\nintervention_date = 2026-02-20\n", encoding="utf-8")
    phase2 = tmp_path / "phases2.txt"
    phase2.write_text("[settings]\nintervention_date = 2026-02-22\n", encoding="utf-8")

    clear_config_cache()
    monkeypatch.setenv("ARENA_PHASE_CONFIG", str(phase1))
    cfg1 = get_config()
    assert cfg1.intervention_date == "2026-02-20"

    monkeypatch.setenv("ARENA_PHASE_CONFIG", str(phase2))
    cfg2 = get_config()
    assert cfg2.intervention_date == "2026-02-22"


def test_resolve_output_dir_prefers_env_over_settings(tmp_path: Path, monkeypatch) -> None:
    settings = tmp_path / "settings.toml"
    from_settings = tmp_path / "from_settings"
    from_env = tmp_path / "from_env"
    settings.write_text(f"[paths]\noutput_dir = '{from_settings.as_posix()}'\n", encoding="utf-8")

    monkeypatch.setenv("ARENA_SETTINGS", str(settings))
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(from_env))
    clear_settings_cache()
    assert paths.resolve_output_dir() == from_env


def test_resolve_runtime_roots_uses_scripts_env_and_settings_paths(tmp_path: Path, monkeypatch) -> None:
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    settings = tmp_path / "settings.toml"
    data_dir = tmp_path / "data_from_settings"
    output_dir = tmp_path / "output_from_settings"
    settings.write_text(
        "\n".join(
            [
                "[paths]",
                f"data_dir = '{data_dir.as_posix()}'",
                f"output_dir = '{output_dir.as_posix()}'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))
    monkeypatch.setenv("ARENA_SETTINGS", str(settings))
    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARENA_OUTPUT_DIR", raising=False)
    clear_settings_cache()

    resolved_scripts, resolved_output, resolved_data = paths.resolve_runtime_roots()
    assert resolved_scripts == scripts_root
    assert resolved_output == output_dir
    assert resolved_data == data_dir


def test_find_scripts_root_uses_cwd_repo_parents_for_non_editable_install(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    scripts_root = repo_root / "scripts"
    nested_cwd = repo_root / "subdir" / "deeper"
    (scripts_root / "adsb").mkdir(parents=True)
    nested_cwd.mkdir(parents=True)

    monkeypatch.delenv("ARENA_SCRIPTS_ROOT", raising=False)
    monkeypatch.delenv("ADSB_SCRIPTS_ROOT", raising=False)
    monkeypatch.chdir(nested_cwd)

    assert find_scripts_root() == scripts_root


def test_find_settings_path_uses_cwd_repo_parents_for_non_editable_install(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    settings_path = repo_root / "scripts" / "config" / "settings.toml"
    nested_cwd = repo_root / "tests"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("[quality]\nmin_auc_n_used = 1\n", encoding="utf-8")
    (repo_root / "scripts" / "adsb").mkdir(parents=True)
    nested_cwd.mkdir(parents=True)

    monkeypatch.delenv("ARENA_SETTINGS", raising=False)
    monkeypatch.delenv("ADSB_SETTINGS", raising=False)
    monkeypatch.delenv("ARENA_SCRIPTS_ROOT", raising=False)
    monkeypatch.delenv("ADSB_SCRIPTS_ROOT", raising=False)
    monkeypatch.chdir(nested_cwd)

    assert find_settings_path() == settings_path
