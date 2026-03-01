import numpy as np

from arena.lib.stats_utils import bootstrap_mean_diff, fit_nb_glm


def test_bootstrap_mean_diff_positive() -> None:
    a = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    b = np.array([4.0, 5.0, 6.0, 5.5, 4.5])
    mean_diff, lo, hi = bootstrap_mean_diff(a, b, n=1000, seed=1)
    assert mean_diff > 0
    assert lo > 0
    assert hi > lo


def test_nb_glm_positive_effect() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=40)
    baseline = 2.0
    effect = 1.2
    lam = np.exp(baseline + effect * x)
    y = rng.poisson(lam)

    res = fit_nb_glm(y, x)
    # coefficient for x should be positive
    assert res.params[1] > 0
