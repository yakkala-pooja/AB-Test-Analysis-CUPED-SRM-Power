"""Check that the methods behave as claimed by rerunning simulated experiments many times.

- A/A tests (no true effect): about 5% of tests should be significant, and about 0.1%
  should flag an SRM.
- A/B tests at the planned size with a true 3% lift: CUPED should reach about 80% power,
  and its 95% CIs should contain the true lift about 95% of the time.

Writes reports/validation.md.
"""
import numpy as np

from abtest.cuped import cuped_lift
from abtest.simulate import simulate_user_metrics
from abtest.srm import srm_test
from abtest.stats import relative_lift
from run_experiment import MDE_RELATIVE, REPORTS, plan

N_SIMS = 1_000


def run(n_per_arm, save_lift, rng):
    df = simulate_user_metrics(2 * n_per_arm, rng, save_lift=save_lift)
    c = df[df["variant"] == "control"]
    t = df[df["variant"] == "treatment"]
    naive = relative_lift(c["saves"], t["saves"])
    cuped, _ = cuped_lift(c["saves"], c["pre_saves"], t["saves"], t["pre_saves"])
    srm = srm_test(df["variant"].value_counts().to_dict())
    return naive, cuped, srm


def summarize(results, true_lift):
    naive, cuped, srm = zip(*results)
    return {
        "naive_significant": np.mean([e.p_value < 0.05 for e in naive]),
        "cuped_significant": np.mean([e.p_value < 0.05 for e in cuped]),
        "cuped_coverage": np.mean([e.ci_low <= true_lift <= e.ci_high for e in cuped]),
        "srm_flagged": np.mean([not s.passed for s in srm]),
    }


def main():
    p = plan()
    n = p["n_per_arm_cuped"]
    rng = np.random.default_rng(2024)
    aa = summarize([run(n, 0.0, rng) for _ in range(N_SIMS)], 0.0)
    ab = summarize([run(n, MDE_RELATIVE, rng) for _ in range(N_SIMS)], MDE_RELATIVE)

    # Monte Carlo standard error of a rate near 5% over 1,000 runs is about 0.7 points.
    lines = [
        "# Validation by simulation",
        "",
        f"{N_SIMS:,} simulated experiments per scenario, {n:,} users per arm (the CUPED plan).",
        "",
        "| Scenario | Check | Expected | Observed |",
        "| --- | --- | --- | --- |",
        f"| A/A (no effect) | False-positive rate, raw t-test | 5% | {aa['naive_significant']:.1%} |",
        f"| A/A (no effect) | False-positive rate, CUPED | 5% | {aa['cuped_significant']:.1%} |",
        f"| A/A (no effect) | CUPED 95% CI covers 0 | 95% | {aa['cuped_coverage']:.1%} |",
        f"| A/A (no effect) | SRM false alarms | 0.1% | {aa['srm_flagged']:.1%} |",
        f"| A/B (3% lift) | Power, raw t-test | below 80% | {ab['naive_significant']:.1%} |",
        f"| A/B (3% lift) | Power, CUPED | 80% | {ab['cuped_significant']:.1%} |",
        f"| A/B (3% lift) | CUPED 95% CI covers 3% | 95% | {ab['cuped_coverage']:.1%} |",
        "",
    ]
    (REPORTS / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
