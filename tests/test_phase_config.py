from arena.lib.paths import SCRIPTS_ROOT
from arena.lib.phase_config import load_phase_config


def test_phase_config_parses_events():
    cfg_path = SCRIPTS_ROOT / "config" / "phases.txt"
    cfg = load_phase_config(str(cfg_path))

    assert cfg.events, "events should be parsed"
    assert cfg.intervention_date, "intervention_date should be set"
    assert cfg.phase_names, "phase_names should be derived"


def test_phase_config_helpers():
    cfg_path = SCRIPTS_ROOT / "config" / "phases.txt"
    cfg = load_phase_config(str(cfg_path))

    assert cfg.hardware_at(cfg.events[0].date)
    assert cfg.fringe_phase(cfg.events[0].date)
