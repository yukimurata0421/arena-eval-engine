"""
phase_config.py — phases.txt parser (v2, consolidated)

phases.txt has two sections only:
  [events]   — intervention events in chronological order
  [settings] — analysis parameters

Each script calls get_config() to obtain PhaseConfig and derives the required
views through its properties.

Usage:
    from phase_config import get_config
    cfg = get_config()

    # events (all events)
    cfg.events           # [Event(date, label, hardware, color), ...]

    # settings (individual parameters)
    cfg.post_change_date     # "2026-01-10"
    cfg.intervention_date    # "2026-02-11"
    cfg.report_start_date    # "2026-01-08"

    # derived views (formats used by scripts)
    cfg.hardware_at(date)    # hardware name on that date
    cfg.hardware_transitions # [("2026-01-14", "airspy_mini"), ...]
    cfg.default_hardware     # "rtl-sdr"
    cfg.hardware_map         # {"rtl-sdr": 0, "airspy_mini": 1, ...}
    cfg.phase_names          # {0: "RTL-SDR", 1: "Airspy Mini", ...}
    cfg.master_log_phases    # [{"date": ..., "name": ..., "color": ...}, ...]
    cfg.signal_phases        # events where hardware changes
    cfg.fringe_phase(date)   # "1_Old_Settings" / "2_New_Filter" / "3_Post_Cable_Fix"
"""

import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# Event dataclass
# ============================================================
@dataclass
class Event:
    date: str       # "2026-01-14"
    label: str      # "Airspy Mini Intro"
    hardware: str   # "airspy_mini" or "" (no change)
    color: str      # "#e74c3c"


# ============================================================
# Default colors (when not specified)
# ============================================================
_DEFAULT_COLORS = [
    "#2c3e50", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
    "#d35400", "#16a085", "#f39c12", "#c0392b", "#7f8c8d",
    "#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#34495e",
]

_PRETTY_HW = {
    "rtl-sdr": "RTL-SDR",
    "airspy_mini": "Airspy Mini",
    "airspy_mini_plus_cable": "Airspy+Cable",
}


# ============================================================
# Config finder
# ============================================================
def _find_config() -> Path:
    env_path = os.getenv("ARENA_PHASE_CONFIG") or os.getenv("ADSB_PHASE_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    here = Path(__file__).resolve().parent   # lib/
    try:
        from arena.lib.paths import SCRIPTS_ROOT
        preferred = SCRIPTS_ROOT / "config" / "phases.txt"
        if preferred.exists():
            return preferred
    except Exception:
        pass
    # Prefer repo/scripts/config when present (source-of-truth in this repo layout)
    try:
        repo_root = here.parents[2]  # .../src/scripts/lib -> .../
        legacy = repo_root / "scripts" / "config" / "phases.txt"
        if legacy.exists():
            return legacy
    except Exception:
        pass

    candidate = here.parent / "config" / "phases.txt"
    if candidate.exists():
        return candidate

    return candidate


# ============================================================
# PhaseConfig
# ============================================================
@dataclass
class PhaseConfig:
    config_path: str = ""
    events: list = field(default_factory=list)  # List[Event]

    # [settings]
    post_change_date: str = "2026-01-10"
    intervention_date: str = "2026-02-11"
    report_start_date: str = "2026-01-08"
    _fringe_boundary: list = field(default_factory=lambda: ["2026-01-31", "2026-02-14"])

    # ---- Backward-compat fields/methods ----

    @property
    def time_resolved_date(self) -> str:
        """Compatibility: previously used by bayesian/time-resolved scripts."""
        return self.intervention_date

    def get_hardware_map(self) -> dict:
        """Compatibility wrapper."""
        return self.hardware_map

    def get_phase_names(self) -> dict:
        """Compatibility wrapper."""
        return self.phase_names

    def get_phase_fallback_dates(self) -> list:
        """Compatibility wrapper."""
        return self.phase_fallback_dates

    # ---- Hardware views ----

    @property
    def default_hardware(self) -> str:
        """Hardware of the first event (initial hardware)."""
        for e in self.events:
            if e.hardware:
                return e.hardware
        return "rtl-sdr"

    @property
    def hardware_transitions(self) -> list:
        """Events where hardware changes (excluding initial): [("2026-01-14", "airspy_mini"), ...]"""
        first = True
        result = []
        for e in self.events:
            if e.hardware:
                if first:
                    first = False
                    continue
                result.append((e.date, e.hardware))
        return result

    def hardware_at(self, date_str: str) -> str:
        """Hardware name on the given date."""
        current = self.default_hardware
        for e in self.events:
            if e.date > date_str:
                break
            if e.hardware:
                current = e.hardware
        return current

    @property
    def hardware_map(self) -> dict:
        """{"rtl-sdr": 0, "airspy_mini": 1, ...}"""
        names = []
        for e in self.events:
            if e.hardware and e.hardware not in names:
                names.append(e.hardware)
        return {name: i for i, name in enumerate(names)}

    @property
    def phase_names(self) -> dict:
        """{0: "RTL-SDR", 1: "Airspy Mini", ...}"""
        return {v: _PRETTY_HW.get(k, k) for k, v in self.hardware_map.items()}

    @property
    def phase_fallback_dates(self) -> list:
        """Bayesian fallback: date-based phase assignment when hardware column is missing."""
        return [d for d, _ in self.hardware_transitions]

    # ---- Report views ----

    @property
    def master_log_phases(self) -> list:
        """For consolidated report: [{"date": ..., "name": ..., "color": ...}, ...]"""
        return [{"date": e.date, "name": e.label, "color": e.color} for e in self.events]

    @property
    def signal_phases(self) -> list:
        """For signal strength evaluation: events where hardware changes (including initial)."""
        return [{"date": e.date, "name": e.label} for e in self.events if e.hardware]

    @property
    def vertical_phases(self) -> list:
        """For LOS efficiency trend: hardware change events (excluding initial) with colors."""
        result = []
        first = True
        for e in self.events:
            if e.hardware:
                if first:
                    first = False
                    continue
                result.append({"date": e.date, "name": e.label, "color": e.color})
        return result

    @property
    def phase_dates(self) -> list:
        """For phase evaluator: hardware change dates = [(date, label), ...]."""
        return [(e.date, e.label) for e in self.events if e.hardware]

    # ---- Fringe decoding ----

    def fringe_phase(self, date_str: str) -> str:
        """Phase mapping for fringe decoding."""
        boundaries = self._fringe_boundary
        names = ["1_Old_Settings", "2_New_Filter", "3_Post_Cable_Fix"]
        for i, bd in enumerate(boundaries):
            if date_str <= bd:
                return names[i] if i < len(names) else f"Phase{i+1}"
        return names[-1] if names else "Unknown"

    @property
    def fringe_boundaries(self) -> list:
        """Compatibility: [(boundary_date, phase_name), ...]."""
        names = ["1_Old_Settings", "2_New_Filter"]
        return list(zip(self._fringe_boundary, names, strict=False))


# ============================================================
# Parser
# ============================================================
def _parse_event_line(key: str, value: str, idx: int) -> Event:
    """Parse: 2026-01-14 = Label | hardware | color"""
    parts = [p.strip() for p in value.split("|")]
    label = parts[0] if len(parts) > 0 else f"Event_{idx}"
    hardware = parts[1] if len(parts) > 1 and parts[1] else ""
    color = parts[2] if len(parts) > 2 and parts[2] else _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
    return Event(date=key.strip(), label=label, hardware=hardware, color=color)


def load_phase_config(path: Optional[str] = None) -> PhaseConfig:
    config_path = Path(path) if path else _find_config()
    cfg = PhaseConfig(config_path=str(config_path))

    if not config_path.exists():
        print(f"  [phase_config] WARNING: {config_path} not found, using defaults")
        return cfg

    cp = ConfigParser()
    cp.read(str(config_path), encoding="utf-8")

    # [events]
    if cp.has_section("events"):
        events = []
        for idx, (key, val) in enumerate(cp.items("events")):
            events.append(_parse_event_line(key, val, idx))
        events.sort(key=lambda e: e.date)
        cfg.events = events

    # [settings]
    if cp.has_section("settings"):
        cfg.post_change_date = cp.get("settings", "post_change_date", fallback=cfg.post_change_date).strip()
        cfg.intervention_date = cp.get("settings", "intervention_date", fallback=cfg.intervention_date).strip()
        cfg.report_start_date = cp.get("settings", "report_start_date", fallback=cfg.report_start_date).strip()
        fb = cp.get("settings", "fringe_boundary", fallback="").strip()
        if fb:
            cfg._fringe_boundary = [d.strip() for d in fb.split(",")]

    return cfg


# ---- Module-level singleton ----
_cached: Optional[PhaseConfig] = None


def get_config(path: Optional[str] = None) -> PhaseConfig:
    global _cached
    if _cached is None or path is not None:
        _cached = load_phase_config(path)
    return _cached
