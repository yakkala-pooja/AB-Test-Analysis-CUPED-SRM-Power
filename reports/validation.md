# Validation by simulation

1,000 simulated experiments per scenario, 31,082 users per arm (the CUPED plan).

| Scenario | Check | Expected | Observed |
| --- | --- | --- | --- |
| A/A (no effect) | False-positive rate, raw t-test | 5% | 5.5% |
| A/A (no effect) | False-positive rate, CUPED | 5% | 4.1% |
| A/A (no effect) | CUPED 95% CI covers 0 | 95% | 95.8% |
| A/A (no effect) | SRM false alarms | 0.1% | 0.0% |
| A/B (3% lift) | Power, raw t-test | below 80% | 51.4% |
| A/B (3% lift) | Power, CUPED | 80% | 80.2% |
| A/B (3% lift) | CUPED 95% CI covers 3% | 95% | 95.4% |
