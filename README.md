# A/B Test Analysis: Power, CUPED, SRM and a Ship Decision

An end-to-end analysis of a home feed experiment, from sizing the test to deciding whether to ship it.

**Scenario.** A Pinterest-style home feed ranking change shows more fresh Pins. Does it make users save more, without hurting sessions or driving up hides?

The data is simulated, with known true effects, so every method here can be checked against the right answer. The pipeline itself (event logs → SQL → statistics → decision) is the same one you would run on real logs.

## Results

| Step | What I did | Result |
| --- | --- | --- |
| Power analysis | Sized the test from 2 periods of historical data for a 3% minimum detectable effect, 80% power, α = 0.05 | 58,957 users per arm for a plain t-test; **31,082 with CUPED (47% fewer)** |
| SQL metrics | Built user-level metrics from session logs in SQLite (CTEs, LEFT JOINs that keep users with zero sessions, weekly splits, window functions) | One row per assigned user, with pre-period covariates |
| SRM check | Chi-square test on the arm split, overall and per app-version segment | Healthy run passed (p = 0.08) |
| CUPED | Used pre-period saves as a covariate | Removed **50% of variance**; CI width went from 6.1 to 4.4 points |
| Effects | Primary metric, ratio metric (delta method), 2 guardrails, week-by-week novelty check | Saves per user **+4.4% (95% CI +2.2% to +6.6%)**; sessions −0.0%; hides +0.5% |
| Decision | Primary metric CI above zero, guardrails passing non-inferiority margins (sessions ≥ −2%, hides ≤ +5%) | **SHIP**, but expect about **+2.6%** long term, since week 1 is inflated by novelty (+6.2% vs. +2.6% in week 2) |
| Logging-bug run | Same test, but the old app version fails to log 15% of treated users | Overall SRM p = 0.028 **missed it**. The per-segment check flagged the old app (p = 1.6e-17) → **DO NOT TRUST**. |

![Effects](reports/figures/effects.png)

The full write-up is in [reports/report.md](reports/report.md).

## Validating the methods

Running a method once does not show it works. [`validate.py`](validate.py) reruns 1,000 simulated experiments per scenario at the planned size:

| Scenario | Check | Expected | Observed |
| --- | --- | --- | --- |
| A/A (no effect) | False-positive rate, raw t-test | 5% | 5.5% |
| A/A (no effect) | False-positive rate, CUPED | 5% | 4.1% |
| A/A (no effect) | SRM false alarms | 0.1% | 0.0% |
| A/B (3% lift) | Power, raw t-test | below 80% | 51.4% |
| A/B (3% lift) | Power, CUPED | 80% | 80.2% |
| A/B (3% lift) | CUPED 95% CI covers the true lift | 95% | 95.4% |

The power analysis predicted 80%, and CUPED delivered 80.2%. Without CUPED, the same test would miss a real 3% lift about half the time.

## What I learned

- **Check SRM by segment, not only overall.** A bug confined to 20% of users moved the overall split by less than 1% and passed the overall test. The segment check found it, and pointed at the cause.
- **SRM bias is quiet.** The lost users saved less than average, so the buggy run reported +4.6% instead of +4.4%. That number looks believable, and that is what makes it dangerous.
- **CUPED is close to free power.** Pre-period saves correlate 0.69 with experiment saves. CUPED therefore removes about 0.69² ≈ 48% of the variance. That halves the users needed without biasing the estimate, which the A/A tests confirm.
- **Novelty inflates short tests.** Week 1 lift was more than twice week 2. Reporting the full-period average would overstate the lasting impact.
- **Ratio metrics need the delta method.** Users are randomized, not impressions, so saves per impression cannot be treated as independent Bernoulli trials.

## Project layout

```
abtest/
  simulate.py   users with latent traits; session logs for pre-period and experiment
  power.py      sample size, power curves, planning from historical data
  srm.py        chi-square sample ratio mismatch test
  stats.py      lift with delta-method CIs; ratio metrics
  cuped.py      CUPED adjustment and variance reduction
  decision.py   ship rules: SRM gate, guardrail non-inferiority, primary metric
  db.py         loads logs into SQLite and runs sql/
sql/
  user_metrics.sql     user-level experiment and pre-period metrics
  srm_by_segment.sql   arm split per app version
run_experiment.py      plan → simulate → SQL → analysis → decision → report
validate.py            A/A and power simulations (about 5 minutes)
tests/                 unit tests for each module
```

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py   # writes reports/report.md and figures (~30s)
python validate.py         # writes reports/validation.md (~5 min)
python -m pytest
```

## Next steps at production scale

- Sequential testing (e.g. mSPRT), so teams can check results early without inflating false positives.
- Heterogeneous effects by user tenure and platform, with multiple-comparison correction.
- Holdback groups to measure the lasting effect beyond 2 weeks directly.
