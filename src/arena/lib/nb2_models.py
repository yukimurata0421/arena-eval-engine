from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd


@dataclass
class Nb2Inputs:
    y: jnp.ndarray
    log_traffic: jnp.ndarray
    n_days: int


def clean_nb2_df(df: pd.DataFrame, require_log_traffic: bool = True) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    subset = ["auc_n_used"]
    if require_log_traffic:
        subset.append("log_traffic")
    return df.dropna(subset=subset)


def prepare_nb2_inputs(df: pd.DataFrame) -> Nb2Inputs:
    y = jnp.array(df["auc_n_used"].values, dtype=jnp.float32)
    log_traffic = jnp.array(df["log_traffic"].values, dtype=jnp.float32)
    n_days = len(df)
    return Nb2Inputs(y=y, log_traffic=log_traffic, n_days=n_days)


def make_single_change_point_model(alpha_name: str = "alpha_inv") -> Callable:
    def model(y, log_traffic, n_days):
        tau = numpyro.sample("tau", dist.DiscreteUniform(0, n_days - 1))
        alpha_before = numpyro.sample("alpha_before", dist.Normal(10.0, 5.0))
        alpha_after = numpyro.sample("alpha_after", dist.Normal(10.0, 5.0))
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        alpha = numpyro.sample(alpha_name, dist.Exponential(1.0))

        idx = jnp.arange(n_days)
        intercept = jnp.where(idx < tau, alpha_before, alpha_after)
        mu = jnp.exp(intercept + beta_traffic * log_traffic)

        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, alpha), obs=y)

    return model


def make_multi_change_point_model(K: int) -> Callable:
    def model(y, log_traffic, n_days, K):
        beta_traffic = numpyro.sample("beta_traffic", dist.Normal(1.0, 0.5))
        alpha_inv = numpyro.sample("alpha_inv", dist.Exponential(1.0))
        taus = numpyro.sample("taus", dist.DiscreteUniform(0, n_days - 1).expand([K]))
        alphas = numpyro.sample("alphas", dist.Normal(10.0, 5.0).expand([K + 1]))

        idx = jnp.arange(n_days)[:, None]
        phase_idx = jnp.sum(idx >= jnp.sort(taus), axis=-1)
        intercept = alphas[phase_idx]
        mu = jnp.exp(intercept + beta_traffic * log_traffic)

        numpyro.sample("y_obs", dist.NegativeBinomial2(mu, alpha_inv), obs=y)

    return model
