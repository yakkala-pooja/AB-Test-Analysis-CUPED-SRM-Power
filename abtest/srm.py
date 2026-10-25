"""Sample ratio mismatch (SRM) check.

If the observed split between arms is too far from the designed split, users were lost
or added unevenly (logging bugs, redirects, bot filtering). Any effect estimate from
that experiment is suspect, so SRM is checked before looking at metrics.
"""
from dataclasses import dataclass

from scipy.stats import chisquare


@dataclass
class SRMResult:
    label: str
    counts: dict
    expected: dict
    chi2: float
    p_value: float
    passed: bool


def srm_test(counts, expected_shares=None, threshold=0.001, label="overall"):
    """Chi-square goodness-of-fit test. A strict threshold keeps false alarms rare."""
    names = sorted(counts)
    if expected_shares is None:
        expected_shares = {k: 1 / len(names) for k in names}
    total = sum(counts[k] for k in names)
    expected = {k: total * expected_shares[k] for k in names}
    chi2, p = chisquare([counts[k] for k in names], [expected[k] for k in names])
    return SRMResult(label, dict(counts), expected, float(chi2), float(p), bool(p >= threshold))
