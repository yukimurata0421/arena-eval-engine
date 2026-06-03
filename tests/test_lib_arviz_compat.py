from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from arena.lib.arviz_compat import hdi_bounds


def test_hdi_bounds_uses_hdi_prob_when_available(monkeypatch) -> None:
    calls: list[dict[str, float]] = []

    def fake_hdi(_samples, **kwargs):
        calls.append(kwargs)
        return np.array([1.0, 2.0])

    monkeypatch.setitem(sys.modules, "arviz", SimpleNamespace(hdi=fake_hdi))

    assert hdi_bounds([1, 2, 3], hdi_prob=0.8) == (1.0, 2.0)
    assert calls == [{"hdi_prob": 0.8}]


def test_hdi_bounds_falls_back_to_prob_keyword(monkeypatch) -> None:
    calls: list[dict[str, float]] = []

    def fake_hdi(_samples, **kwargs):
        calls.append(kwargs)
        if "hdi_prob" in kwargs:
            raise TypeError("hdi got an unexpected keyword argument: 'hdi_prob'")
        return np.array([3.0, 4.0])

    monkeypatch.setitem(sys.modules, "arviz", SimpleNamespace(hdi=fake_hdi))

    assert hdi_bounds([1, 2, 3], hdi_prob=0.94) == (3.0, 4.0)
    assert calls == [{"hdi_prob": 0.94}, {"prob": 0.94}]
