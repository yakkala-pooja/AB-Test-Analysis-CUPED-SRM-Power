"""Plan, run and analyze the simulated home feed experiment.

Steps: size the test from historical data, simulate it, compute user metrics in SQL,
check SRM, estimate effects with and without CUPED, and make a ship decision. A second
run with an assignment-logging bug shows the SRM check catching it.

Writes reports/report.md and reports/figures/*.png.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abtest import db
from abtest.cuped import cuped_lift
from abtest.decision import Guardrail, decide
from abtest.power import plan_from_history, power_at
from abtest.simulate import Scenario, simulate_experiment, simulate_user_metrics
from abtest.srm import srm_test
from abtest.stats import ratio_lift, relative_lift

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MDE_RELATIVE = 0.03         # smallest lift in saves per user worth shipping
HISTORY_USERS = 50_000
SESSIONS_MAX_HARM = 0.02    # guardrail: sessions per user may not drop more than 2%
HIDES_MAX_HARM = 0.05       # guardrail: hides per user may not rise more than 5%
LOGGING_BUG_LOSS = 0.15     # share of treated old-app users whose assignment is not logged


def plan():
    """Size the experiment from two consecutive 14-day periods of historical data."""
    history = simulate_user_metrics(HISTORY_USERS, np.random.default_rng(1))
    return plan_from_history(history["saves"], history["pre_saves"], MDE_RELATIVE)


def analyze(metrics):
    c = metrics[metrics["variant"] == "control"]
    t = metrics[metrics["variant"] == "treatment"]

    def cuped(col, pre_col, name):
        return cuped_lift(c[col], c[pre_col], t[col], t[pre_col], name)

    primary, reduction = cuped("saves", "pre_saves", "Saves per user (CUPED)")
    sessions, _ = cuped("sessions", "pre_sessions", "Sessions per user (CUPED)")
    hides, _ = cuped("hides", "pre_hides", "Hides per user (CUPED)")
    week1, _ = cuped("saves_week1", "pre_saves", "Saves per user, week 1 (CUPED)")
    week2, _ = cuped("saves_week2", "pre_saves", "Saves per user, week 2 (CUPED)")
    srm = srm_test(metrics["variant"].value_counts().to_dict())
    segment_srm = [srm_test(rows["variant"].value_counts().to_dict(), label=f"app version '{v}'")
                   for v, rows in metrics.groupby("app_version")]
    guardrails = [Guardrail(sessions, SESSIONS_MAX_HARM, higher_is_better=True),
                  Guardrail(hides, HIDES_MAX_HARM, higher_is_better=False)]
    decision, reasons = decide([srm, *segment_srm], primary, guardrails)
    return {
        "srm": srm,
        "segment_srm": segment_srm,
        "naive": relative_lift(c["saves"], t["saves"], "Saves per user (no CUPED)"),
        "primary": primary,
        "variance_reduction": reduction,
        "save_rate": ratio_lift(c["saves"], c["impressions"], t["saves"], t["impressions"],
                                "Saves per impression"),
        "sessions": sessions,
        "hides": hides,
        "week1": week1,
        "week2": week2,
        "decision": decision,
        "reasons": reasons,
    }


def plot_power(p, n_run):
    n = np.linspace(2_000, p["n_per_arm_naive"] * 1.3, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, power_at(n, p["sd"], p["mde_absolute"]), label="t-test on raw saves")
    ax.plot(n, power_at(n, p["sd"], p["mde_absolute"], rho=p["rho"]), label="CUPED")
    ax.axhline(0.8, color="grey", lw=0.8, ls="--")
    ax.axvline(n_run, color="grey", lw=0.8, ls=":")
    ax.annotate("users per arm\nin this test", (n_run, 0.1), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color="grey")
    ax.set(xlabel="Users per arm", ylabel="Power", ylim=(0, 1),
           title=f"Power to detect a {MDE_RELATIVE:.0%} lift in saves per user")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "power_curve.png", dpi=150)
    plt.close(fig)


def plot_effects(r):
    rows = [r["naive"], r["primary"], r["week1"], r["week2"], r["save_rate"], r["sessions"], r["hides"]]
    labels = ["Saves/user, raw", "Saves/user, CUPED", "  week 1 (CUPED)", "  week 2 (CUPED)",
              "Saves/impression", "Sessions/user (guardrail)", "Hides/user (guardrail)"]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    for yi, e in zip(y, rows):
        ax.plot([e.ci_low * 100, e.ci_high * 100], [yi, yi], color="#3b6ea8", lw=2)
        ax.plot(e.lift * 100, yi, "o", color="#3b6ea8")
    ax.axvline(0, color="black", lw=0.8)
    # Guardrail margins, drawn on the sessions (y=1) and hides (y=0) rows.
    for margin, yi in ((-SESSIONS_MAX_HARM, 1), (HIDES_MAX_HARM, 0)):
        ax.plot([margin * 100] * 2, [yi - 0.3, yi + 0.3], color="#c0392b", lw=1.5, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Relative change vs. control (%), 95% CI")
    ax.set_title("Treatment effects (red dashes: guardrail margins)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "effects.png", dpi=150)
    plt.close(fig)


def fmt(e):
    return f"{e.lift:+.2%} | {e.ci_low:+.2%} to {e.ci_high:+.2%} | {e.p_value:.2g}"


def write_report(p, scenario, r, bug, bug_segments):
    srm = r["srm"]
    lines = [
        "# Home feed experiment: analysis report",
        "",
        "Generated by `run_experiment.py` from simulated data. True effects built into the "
        f"simulation: {scenario.save_lift:.0%} long-run lift in save rate, plus an extra "
        f"{scenario.novelty_lift:.0%} novelty lift on day 0 that halves every "
        f"{scenario.novelty_halflife_days:g} days; hides up {scenario.hide_lift:.0%}; sessions unchanged.",
        "",
        "## 1. Plan",
        "",
        f"Historical data ({HISTORY_USERS:,} users, two consecutive 14-day periods): saves per user "
        f"mean {p['mean']:.2f}, SD {p['sd']:.2f}, correlation with the previous period {p['rho']:.2f}.",
        "",
        "| Design | Users per arm for 80% power at a 3% MDE |",
        "| --- | --- |",
        f"| t-test on raw saves | {p['n_per_arm_naive']:,} |",
        f"| CUPED (pre-period saves as covariate) | {p['n_per_arm_cuped']:,} |",
        "",
        f"CUPED needs {1 - p['n_per_arm_cuped'] / p['n_per_arm_naive']:.0%} fewer users. "
        f"The test ran with {p['n_per_arm_cuped']:,} users per arm.",
        "",
        "![Power curve](figures/power_curve.png)",
        "",
        "## 2. Sample ratio check",
        "",
        f"Control {srm.counts['control']:,} vs. treatment {srm.counts['treatment']:,}; "
        f"chi-square p = {srm.p_value:.2f}. Per-segment checks: "
        + ", ".join(f"{s.label} p = {s.p_value:.2f}" for s in r["segment_srm"])
        + ". All passed (an SRM is flagged at p < 0.001).",
        "",
        "## 3. Results",
        "",
        "| Metric | Lift | 95% CI | p-value |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("naive", "primary", "week1", "week2", "save_rate", "sessions", "hides"):
        lines.append(f"| {r[key].metric} | {fmt(r[key])} |")
    lines += [
        "",
        f"CUPED removed {r['variance_reduction']:.0%} of the variance in saves per user and narrowed the "
        f"CI from {(r['naive'].ci_high - r['naive'].ci_low) * 100:.2f} to "
        f"{(r['primary'].ci_high - r['primary'].ci_low) * 100:.2f} percentage points.",
        "",
        f"Novelty check: the week-1 lift ({r['week1'].lift:+.1%}) is larger than week 2 ({r['week2'].lift:+.1%}). "
        "Week 2 is the better estimate of the lasting effect (true values in the simulation: week 1 about 6.6%, week 2 about 3.3%), "
        "though its CI is wide.",
        "",
        "![Effects](figures/effects.png)",
        "",
        f"## 4. Decision: {r['decision']}",
        "",
        *[f"- {reason}" for reason in r["reasons"]],
        f"- Expected lasting impact: about {r['week2'].lift:+.1%} saves per user (week-2 estimate), "
        f"not the {r['primary'].lift:+.1%} full-period average, which includes novelty.",
        "",
        "## 5. Same experiment with an assignment-logging bug",
        "",
        f"Here {scenario.old_app_share:.0%} of users are on an old app version that fails to log "
        f"assignment for {LOGGING_BUG_LOSS:.0%} of treated users. Those users still use the app; "
        "they are just missing from the experiment tables.",
        "",
        f"Overall, control has {bug['srm'].counts['control']:,} users and treatment "
        f"{bug['srm'].counts['treatment']:,} (p = {bug['srm'].p_value:.3f}). "
        + ("That is not below the 0.001 threshold, so an overall-only check would miss the bug. "
           if bug["srm"].passed else "")
        + f"The per-segment check catches it, and the decision is **{bug['decision']}**.",
        "",
        "| App version | Variant | Users | Share | Segment SRM p-value |",
        "| --- | --- | --- | --- | --- |",
    ]
    segment_p = {s.label: s.p_value for s in bug["segment_srm"]}
    for _, row in bug_segments.iterrows():
        p_value = segment_p[f"app version '{row.app_version}'"]
        lines.append(f"| {row.app_version} | {row.variant} | {row.users:,} | {row.share:.1%} | "
                     f"{p_value:.1e} |")
    lines += [
        "",
        "The imbalance sits entirely in the old app version, which points straight at the bug.",
        f"Had we ignored SRM, the CUPED estimate would read {bug['primary'].lift:+.2%} instead of "
        f"{r['primary'].lift:+.2%}: the lost users save less than average, so dropping them "
        "inflates treatment.",
        "",
    ]
    (REPORTS / "report.md").write_text("\n".join(lines), encoding="utf-8")



def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = plan()
    n_per_arm = p["n_per_arm_cuped"]
    scenario = Scenario(n_users=2 * n_per_arm)

    conn = db.load(*simulate_experiment(scenario), DATA / "experiment.db")
    results = analyze(db.query(conn, "user_metrics.sql"))

    bug_scenario = Scenario(n_users=2 * n_per_arm, old_app_logging_loss=LOGGING_BUG_LOSS)
    bug_conn = db.load(*simulate_experiment(bug_scenario), DATA / "experiment_logging_bug.db")
    bug = analyze(db.query(bug_conn, "user_metrics.sql"))
    bug_segments = db.query(bug_conn, "srm_by_segment.sql")

    plot_power(p, n_per_arm)
    plot_effects(results)
    write_report(p, scenario, results, bug, bug_segments)

    print(f"Plan: {p['n_per_arm_naive']:,} users/arm raw, {n_per_arm:,} with CUPED (rho={p['rho']:.3f})")
    for key in ("naive", "primary", "week1", "week2", "save_rate", "sessions", "hides"):
        print(f"  {results[key].metric:<28} {fmt(results[key])}")
    print(f"Variance reduction: {results['variance_reduction']:.1%}")
    print(f"SRM p={results['srm'].p_value:.3f}  Decision: {results['decision']}")
    for reason in results["reasons"]:
        print("  -", reason)
    print(f"Bug run: overall SRM p={bug['srm'].p_value:.2e} -> {bug['decision']}: {bug['reasons']}; "
          f"biased lift {bug['primary'].lift:+.2%}")
    print(bug_segments.to_string(index=False))


if __name__ == "__main__":
    main()
