"""Turn experiment results into a ship decision with stated reasons."""
from dataclasses import dataclass

from .srm import SRMResult
from .stats import Effect


@dataclass
class Guardrail:
    effect: Effect
    max_harm: float          # largest acceptable relative regression, e.g. 0.05 for 5%
    higher_is_better: bool

    @property
    def violated(self):
        # Non-inferiority: the CI has to rule out a regression bigger than max_harm.
        # A wide CI fails too, because it cannot show the metric is safe.
        if self.higher_is_better:
            return self.effect.ci_low < -self.max_harm
        return self.effect.ci_high > self.max_harm


def decide(srm_checks: list[SRMResult], primary: Effect, guardrails: list[Guardrail]):
    """Return (decision, reasons). Decision is SHIP, DO NOT SHIP, INCONCLUSIVE or DO NOT TRUST.

    srm_checks holds the overall check plus one per segment: a bug confined to one
    segment can be too small to trip the overall test.
    """
    mismatches = [s for s in srm_checks if not s.passed]
    if mismatches:
        return "DO NOT TRUST", [
            f"Sample ratio mismatch in {s.label} (p = {s.p_value:.1e}). Find and fix the "
            "assignment or logging bug before reading any metric."
            for s in mismatches
        ]

    reasons = [
        f"Guardrail '{g.effect.metric}' cannot rule out a regression worse than {g.max_harm:.0%} "
        f"(CI {g.effect.ci_low:+.1%} to {g.effect.ci_high:+.1%})."
        for g in guardrails if g.violated
    ]
    if reasons:
        return "DO NOT SHIP", reasons

    ci = f"{primary.lift:+.1%} (95% CI {primary.ci_low:+.1%} to {primary.ci_high:+.1%})"
    guardrails_ok = "all guardrails are within their margins"
    if primary.ci_low > 0:
        return "SHIP", [f"{primary.metric} rose {ci} and {guardrails_ok}."]
    if primary.ci_high < 0:
        return "DO NOT SHIP", [f"{primary.metric} fell {ci}."]
    return "INCONCLUSIVE", [f"{primary.metric} changed {ci}; the CI includes zero."]
