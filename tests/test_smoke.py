def test_paths_resolve():
    from arena.lib import paths

    assert paths.SCRIPTS_ROOT.exists(), "SCRIPTS_ROOT should exist"
    assert (paths.SCRIPTS_ROOT / "adsb").exists(), "scripts/adsb should exist"
    assert paths.OUTPUT_DIR.name == "output"


def test_phase_config_loads():
    from arena.lib.paths import SCRIPTS_ROOT
    from arena.lib.phase_config import load_phase_config

    cfg_path = SCRIPTS_ROOT / "config" / "phases.txt"
    cfg = load_phase_config(str(cfg_path))
    assert cfg.intervention_date, "intervention_date should be set"
