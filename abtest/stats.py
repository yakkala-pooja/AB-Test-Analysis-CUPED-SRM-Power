"""Effect estimates for user-randomized experiments.

Samples are in the tens of thousands, so normal approximations are used throughout.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Effect:
    metric: str
    control_mean: float
    treatment_mean: float
    lift: float      # relative change, treatment vs. control
    ci_low: float    # confidence interval on the lift
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int


def _lift_ci(mc, mt, var_c, var_t, alpha):
    # Delta method for the ratio of two independent means.
    se = np.sqrt(var_t / mc**2 + mt**2 * var_c / mc**4)
    z = norm.ppf(1 - alpha / 2)
    lift = mt / mc - 1
    return lift, lift - z * se, lift + z * se


def relative_lift(control, treatment, metric="", alpha=0.05):
    """Relative lift in a per-user mean, with a Welch-style z-test on the difference."""
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    mc, mt = c.mean(), t.mean()
    var_c, var_t = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    p = 2 * norm.sf(abs(mt - mc) / np.sqrt(var_c + var_t))
    lift, lo, hi = _lift_ci(mc, mt, var_c, var_t, alpha)
    return Effect(metric, mc, mt, lift, lo, hi, p, c.size, t.size)


def _ratio_and_var(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    r = num.mean() / den.mean()
    cov = np.cov(num, den, ddof=1)
    var = (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1]) / (num.size * den.mean() ** 2)
    return r, var


def ratio_lift(num_c, den_c, num_t, den_t, metric="", alpha=0.05):
    """Lift in sum(num) / sum(den), e.g. saves per impression.

    Users are randomized, not impressions, so impressions from one user are correlated.
    Treating them as independent understates the variance; the delta method does not.
    """
    rc, var_c = _ratio_and_var(num_c, den_c)
    rt, var_t = _ratio_and_var(num_t, den_t)
    p = 2 * norm.sf(abs(rt - rc) / np.sqrt(var_c + var_t))
    lift, lo, hi = _lift_ci(rc, rt, var_c, var_t, alpha)
    return Effect(metric, rc, rt, lift, lo, hi, p, len(num_c), len(num_t))
