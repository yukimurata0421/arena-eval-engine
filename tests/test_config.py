from arena.lib.config import get_distance_bins_km, get_quality_thresholds, get_site_latlon


def test_config_reads_settings():
    lat, lon = get_site_latlon()
    assert isinstance(lat, float)
    assert isinstance(lon, float)

    min_auc, min_minutes = get_quality_thresholds()
    assert min_auc > 0
    assert min_minutes > 0

    bins = get_distance_bins_km()
    assert len(bins) >= 2
