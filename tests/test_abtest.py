import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from abtest import db
from abtest.cuped import cuped_lift, theta
from abtest.decision import Guardrail, decide
from abtest.power import power_at, sample_size_per_arm
from abtest.simulate import Scenario, simulate_experiment, simulate_user_metrics
from abtest.srm import srm_test
from abtest.stats import Effect, ratio_lift, relative_lift


def effect(lift, lo, hi, p=0.01, metric="m"):
    return Effect(metric, 1.0, 1.0 + lift, lift, lo, hi, p, 1000, 1000)


def test_sample_size_matches_textbook_formula():
    # 2 * (1.96 + 0.8416)^2 * 1^2 / 0.1^2 = 1569.8
    assert sample_size_per_arm(sd=1.0, mde=0.1) == 1570


def test_cuped_shrinks_sample_size_by_one_minus_rho_squared():
    naive = sample_size_per_arm(sd=1.0, mde=0.05)
    cuped = sample_size_per_arm(sd=1.0, mde=0.05, rho=0.6)
    assert cuped == pytest.approx(naive * (1 - 0.36), rel=0.001)


def test_power_at_planned_size_is_eighty_percent():
    n = sample_size_per_arm(sd=2.0, mde=0.1)
    assert power_at(n, sd=2.0, mde=0.1) == pytest.approx(0.8, abs=0.002)


def test_srm_passes_balanced_split_and_flags_skewed_one():
    assert srm_test({"control": 50_000, "treatment": 50_100}).passed
    assert not srm_test({"control": 50_000, "treatment": 48_500}).passed


def test_relative_lift_p_value_matches_welch_t_test():
    rng = np.random.default_rng(0)
    c = rng.exponential(1.0, 20_000)
    t = rng.exponential(1.03, 20_000)
    e = relative_lift(c, t)
    assert e.p_value == pytest.approx(ttest_ind(t, c, equal_var=False).pvalue, rel=0.01)
    assert e.ci_low < e.lift < e.ci_high


def test_ratio_lift_with_equal_rates_is_near_zero():
    rng = np.random.default_rng(1)
    den_c, den_t = rng.poisson(50, 10_000), rng.poisson(50, 10_000)
    e = ratio_lift(rng.binomial(den_c, 0.1), den_c, rng.binomial(den_t, 0.1), den_t)
    assert e.ci_low < 0 < e.ci_high


def test_theta_recovers_regression_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(size=50_000)
    y = 2.5 * x + rng.normal(size=50_000)
    assert theta(y, x) == pytest.approx(2.5, abs=0.02)


def test_cuped_keeps_estimate_and_cuts_variance():
    df = simulate_user_metrics(40_000, np.random.default_rng(3), save_lift=0.05)
    c, t = df[df.variant == "control"], df[df.variant == "treatment"]
    naive = relative_lift(c.saves, t.saves)
    cuped, reduction = cuped_lift(c.saves, c.pre_saves, t.saves, t.pre_saves)
    assert reduction > 0.3
    assert (cuped.ci_high - cuped.ci_low) < (naive.ci_high - naive.ci_low)
    assert cuped.ci_low < 0.05 < cuped.ci_high


def test_decision_rules():
    ok_srm = [srm_test({"control": 1000, "treatment": 1000})]
    bad_srm = [srm_test({"control": 1000, "treatment": 800}, label="segment")]
    safe = [Guardrail(effect(0.0, -0.01, 0.01), max_harm=0.02, higher_is_better=True)]
    harmful = [Guardrail(effect(-0.02, -0.04, 0.0), max_harm=0.02, higher_is_better=True)]

    assert decide(ok_srm, effect(0.04, 0.02, 0.06), safe)[0] == "SHIP"
    assert decide(bad_srm, effect(0.04, 0.02, 0.06), safe)[0] == "DO NOT TRUST"
    assert decide(ok_srm, effect(0.04, 0.02, 0.06), harmful)[0] == "DO NOT SHIP"
    assert decide(ok_srm, effect(0.01, -0.01, 0.03, p=0.3), safe)[0] == "INCONCLUSIVE"
    assert decide(ok_srm, effect(-0.04, -0.06, -0.02), safe)[0] == "DO NOT SHIP"


def test_user_metrics_sql_keeps_users_with_no_sessions(tmp_path):
    assignments = pd.DataFrame({"user_id": [1, 2], "variant": ["control", "treatment"],
                                "app_version": ["current", "current"]})
    sessions = pd.DataFrame({"user_id": [1, 1], "day": [-3, 2], "impressions": [20, 30],
                             "saves": [1, 4], "hides": [0, 1]})
    conn = db.load(assignments, sessions, tmp_path / "t.db")
    m = db.query(conn, "user_metrics.sql").set_index("user_id")
    assert m.loc[1, "saves"] == 4 and m.loc[1, "pre_saves"] == 1
    assert m.loc[2, ["sessions", "saves", "pre_saves"]].tolist() == [0, 0, 0]


def test_logging_bug_removes_only_treated_old_app_users():
    assignments, _ = simulate_experiment(Scenario(n_users=20_000, old_app_logging_loss=0.5))
    shares = assignments.groupby("app_version")["variant"].value_counts(normalize=True)
    assert shares["current"]["treatment"] == pytest.approx(0.5, abs=0.02)
    assert shares["old"]["treatment"] == pytest.approx(1 / 3, abs=0.03)
