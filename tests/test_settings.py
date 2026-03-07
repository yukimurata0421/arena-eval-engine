from pathlib import Path

from arena.lib.config import get_distance_bins_km, get_quality_thresholds, get_site_latlon
from arena.lib.runtime_config import load_settings


def test_settings_load_from_env_path(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text(
        "\n".join(
            [
                "[site]",
                "lat = 35.5",
                "lon = 139.5",
                "",
                "[quality]",
                "min_auc_n_used = 1234",
                "min_minutes_covered = 1111",
                "",
                "[distance_bins]",
                "km = [0, 10, 20]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_SETTINGS", str(settings_path))

    snap = load_settings()
    assert Path(snap.path) == settings_path
    assert isinstance(snap.data, dict)

    lat, lon = get_site_latlon()
    assert lat == 35.5
    assert lon == 139.5
    assert get_quality_thresholds() == (1234, 1111)
    assert get_distance_bins_km() == [0.0, 10.0, 20.0]
