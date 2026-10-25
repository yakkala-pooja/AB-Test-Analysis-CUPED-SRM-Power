"""Sample size and power for a two-arm test on a per-user mean."""
import numpy as np
from scipy.stats import norm


def sample_size_per_arm(sd, mde, alpha=0.05, power=0.8, rho=0.0):
    """Users per arm to detect an absolute difference `mde` with a two-sided test.

    rho is the correlation between the metric and its pre-period value. CUPED cuts the
    variance by a factor of (1 - rho^2), and the required sample size with it.
    """
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * z**2 * sd**2 * (1 - rho**2) / mde**2))


def power_at(n_per_arm, sd, mde, alpha=0.05, rho=0.0):
    se = sd * np.sqrt(1 - rho**2) * np.sqrt(2 / np.asarray(n_per_arm, dtype=float))
    return norm.sf(norm.ppf(1 - alpha / 2) - mde / se)


def plan_from_history(y, x, mde_relative, alpha=0.05, power=0.8):
    """Size an experiment from historical data: y is the metric, x the same metric one period earlier."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mean, sd = y.mean(), y.std(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    mde = mde_relative * mean
    return {
        "mean": mean,
        "sd": sd,
        "rho": rho,
        "mde_relative": mde_relative,
        "mde_absolute": mde,
        "n_per_arm_naive": sample_size_per_arm(sd, mde, alpha, power),
        "n_per_arm_cuped": sample_size_per_arm(sd, mde, alpha, power, rho),
    }
