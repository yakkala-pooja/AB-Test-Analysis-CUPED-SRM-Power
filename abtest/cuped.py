"""CUPED variance reduction (Deng, Xu, Kohavi & Walker, 2013).

The adjusted metric is Y - theta * (X - mean(X)), where X is the same metric in the
pre-period. X is measured before assignment, so the adjustment cannot bias the effect;
it only removes variance that X predicts (a share of roughly corr(X, Y)^2).
"""
import numpy as np

from .stats import relative_lift


def theta(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)


def cuped_lift(y_c, x_c, y_t, x_t, metric="", alpha=0.05):
    """Return (Effect on the CUPED-adjusted metric, share of control variance removed)."""
    y_c, x_c, y_t, x_t = (np.asarray(a, dtype=float) for a in (y_c, x_c, y_t, x_t))
    x_all = np.concatenate([x_c, x_t])
    # Theta is estimated on both arms pooled, as in the original paper.
    th = theta(np.concatenate([y_c, y_t]), x_all)
    x_mean = x_all.mean()
    adj_c = y_c - th * (x_c - x_mean)
    adj_t = y_t - th * (x_t - x_mean)
    reduction = 1 - adj_c.var(ddof=1) / y_c.var(ddof=1)
    return relative_lift(adj_c, adj_t, metric, alpha), reduction
