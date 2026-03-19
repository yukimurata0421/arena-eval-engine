from __future__ import annotations

import builtins

import pandas as pd

from arena.lib import input_utils as iu


def test_parse_date_supports_dash_and_slash() -> None:
    assert str(iu.parse_date("2026-03-20")) == "2026-03-20"
    assert str(iu.parse_date("2026/03/20")) == "2026-03-20"
    assert iu.parse_date("20-03-2026") is None


def test_prompt_intervention_date_uses_default_on_empty(monkeypatch) -> None:
    responses = iter([""])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(responses))
    ts = iu.prompt_intervention_date("2026-02-11")
    assert isinstance(ts, pd.Timestamp)
    assert ts.strftime("%Y-%m-%d") == "2026-02-11"


def test_prompt_phase_dates_collects_baseline_and_interventions(monkeypatch) -> None:
    responses = iter(
        [
            "bad-date",
            "2026-01-01",
            "2026-01-14,airspy_introduce",
            "2026/02/28,",
            "done",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(responses))

    phases = iu.prompt_phase_dates(default_labels={"2026-02-28": "fallback_label"})
    assert phases[0] == {"date": "2026-01-01", "name": "Initial Baseline"}
    assert phases[1] == {"date": "2026-01-14", "name": "airspy_introduce"}
    assert phases[2] == {"date": "2026-02-28", "name": "fallback_label"}

