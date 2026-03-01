from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import statsmodels.api as sm


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n: int = 20000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    diffs = []
    for _ in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(bb) - np.mean(aa)))
    diffs = np.asarray(diffs)
    return (float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def fit_nb_glm(y: np.ndarray, x: np.ndarray, offset: np.ndarray | None = None) -> sm.GLM:
    x = sm.add_constant(x, has_constant='add')
    model = sm.GLM(y, x, family=sm.families.NegativeBinomial(), offset=offset)
    return model.fit()
