"""Simulated data for a Pinterest-style home feed experiment.

Each user has a latent session rate, save rate and hide rate. The same traits drive
both the pre-period and the experiment period, so pre-period saves predict experiment
saves. That predictable part of the variance is what CUPED removes.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

IMPRESSIONS_PER_SESSION = 25


@dataclass
class Scenario:
    n_users: int = 60_000
    days: int = 14
    treatment_share: float = 0.5
    save_lift: float = 0.03            # long-run relative lift in save rate
    novelty_lift: float = 0.08         # extra lift on day 0 that fades out
    novelty_halflife_days: float = 2.0
    hide_lift: float = 0.02            # treatment also nudges hides up slightly
    old_app_share: float = 0.20
    old_app_logging_loss: float = 0.0  # share of treated old-app users never logged as assigned
    seed: int = 7


def draw_users(n, rng, old_app_share=0.20):
    old_app = rng.random(n) < old_app_share
    save_rate = rng.beta(2, 60, n)
    return pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "app_version": np.where(old_app, "old", "current"),
        "session_rate": rng.gamma(0.8, 0.6, n),  # sessions per day
        # The old app shows the save button less prominently.
        "save_rate": np.where(old_app, 0.7 * save_rate, save_rate),
        "hide_rate": rng.beta(1, 200, n),
    })


def _sessions_per_user(users, rng, days):
    # A gamma shock per period lets activity drift, so pre and post are correlated but not identical.
    drift = rng.gamma(5.0, 1 / 5.0, len(users))
    return rng.poisson(users["session_rate"].to_numpy() * drift * days)


def _session_rows(users, rng, s, day_offset, treated):
    n_sessions = _sessions_per_user(users, rng, s.days)
    idx = np.repeat(np.arange(len(users)), n_sessions)
    day = rng.integers(0, s.days, idx.size)
    impressions = rng.poisson(IMPRESSIONS_PER_SESSION, idx.size) + 1

    save_p = users["save_rate"].to_numpy()[idx]
    hide_p = users["hide_rate"].to_numpy()[idx]
    t = treated[idx]
    save_mult = 1 + s.save_lift + s.novelty_lift * 0.5 ** (day / s.novelty_halflife_days)
    save_p = np.where(t, np.minimum(save_p * save_mult, 1.0), save_p)
    hide_p = np.where(t, hide_p * (1 + s.hide_lift), hide_p)

    return pd.DataFrame({
        "user_id": users["user_id"].to_numpy()[idx],
        "day": day + day_offset,
        "impressions": impressions,
        "saves": rng.binomial(impressions, save_p),
        "hides": rng.binomial(impressions, hide_p),
    })


def simulate_experiment(s: Scenario):
    """Return (assignments, sessions) tables like the ones an experiment platform would log.

    Sessions cover a pre-period (days -14 to -1) and the experiment (days 0 to 13).
    With old_app_logging_loss > 0, some treated old-app users use the product but their
    assignment is never logged, which is the bug the SRM check should catch.
    """
    rng = np.random.default_rng(s.seed)
    users = draw_users(s.n_users, rng, s.old_app_share)
    treated = rng.random(s.n_users) < s.treatment_share
    users["variant"] = np.where(treated, "treatment", "control")

    pre = _session_rows(users, rng, s, day_offset=-s.days, treated=np.zeros(s.n_users, dtype=bool))
    experiment = _session_rows(users, rng, s, day_offset=0, treated=treated)
    sessions = pd.concat([pre, experiment], ignore_index=True)

    old_app = (users["app_version"] == "old").to_numpy()
    lost = treated & old_app & (rng.random(s.n_users) < s.old_app_logging_loss)
    assignments = users.loc[~lost, ["user_id", "variant", "app_version"]].reset_index(drop=True)
    return assignments, sessions


def simulate_user_metrics(n_users, rng, save_lift=0.0, days=14, treatment_share=0.5):
    """User-level pre-period and experiment saves without session rows.

    Same model as simulate_experiment (sums of per-session Poisson/Binomial draws), but
    fast enough to rerun thousands of times for power and false-positive checks.
    """
    users = draw_users(n_users, rng)
    treated = rng.random(n_users) < treatment_share
    save_rate = users["save_rate"].to_numpy()
    out = {}
    for column, mult in (("pre_saves", np.ones(n_users)),
                         ("saves", np.where(treated, 1 + save_lift, 1.0))):
        n_sessions = _sessions_per_user(users, rng, days)
        impressions = rng.poisson(IMPRESSIONS_PER_SESSION * n_sessions) + n_sessions
        out[column] = rng.binomial(impressions, np.minimum(save_rate * mult, 1.0))
    return pd.DataFrame({"variant": np.where(treated, "treatment", "control"), **out})
