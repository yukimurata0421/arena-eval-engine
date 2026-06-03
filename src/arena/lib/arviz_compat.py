from __future__ import annotations

from typing import Any

import numpy as np


def hdi_bounds(samples: Any, *, hdi_prob: float = 0.94) -> tuple[float, float]:
    """Return ArviZ HDI bounds across old and new keyword/result shapes."""
    import arviz as az

    errors: list[TypeError] = []
    for kwargs in ({"hdi_prob": hdi_prob}, {"prob": hdi_prob}, {"ci_prob": hdi_prob}):
        try:
            result = az.hdi(samples, **kwargs)
            break
        except TypeError as exc:
            errors.append(exc)
    else:
        raise errors[-1]

    if isinstance(result, np.ndarray):
        arr = np.asarray(result).reshape(-1)
    elif hasattr(result, "keys"):
        first_key = next(iter(result.keys()))
        arr = np.asarray(result[first_key]).reshape(-1)
    else:
        arr = np.asarray(next(iter(result.data_vars.values()))).reshape(-1)
    return float(arr[0]), float(arr[1])
