# Multiplicity-Aware LIME Policy Analysis

## Material Passport

- Material ID: `lime-multiplicity-aware-policy-v1`
- Type: diagnostic policy comparison
- Verification status: `VERIFIED`
- Source experiments: held-out LIME validations on Breast Cancer Wisconsin and binary Wine
- Calibration seeds: 30 per dataset
- Held-out evaluation seeds: 30 per dataset
- Monitored metrics: 4
- Family-level alpha: 0.05
- External data upload: none

## Research purpose

The held-out LIME policy selects a run for review when any monitored metric
exceeds its calibration-derived run limit. Applying four separate 95th-percentile
limits can increase the probability that a fixed-model run is selected at least
once. This follow-up compares that original rule with a Bonferroni rule that
uses the `1 - 0.05 / 4 = 0.9875` calibration quantile for each metric.

Both rules are applied to exactly the same calibration, held-out fixed-model
control and cross-version candidate samples. The comparison therefore changes
only the run-level decision rule; it does not retrain a model, regenerate an
explanation or change a random seed.

## Results

| Dataset | Policy | Control review rate | Candidate review rate | Exact paired p | Status |
|---|---|---:|---:|---:|---|
| Breast Cancer Wisconsin | Unadjusted | 10.0% (3/30), 95% CI [3.5%, 25.6%] | 66.7% (20/30), 95% CI [48.8%, 80.8%] | 0.000015 | SEPARATED |
| Breast Cancer Wisconsin | Bonferroni | 10.0% (3/30), 95% CI [3.5%, 25.6%] | 63.3% (19/30), 95% CI [45.5%, 78.1%] | 0.000031 | SEPARATED |
| Wine (binary) | Unadjusted | 13.3% (4/30), 95% CI [5.3%, 29.7%] | 63.3% (19/30), 95% CI [45.5%, 78.1%] | 0.000061 | SEPARATED |
| Wine (binary) | Bonferroni | 3.3% (1/30), 95% CI [0.6%, 16.7%] | 43.3% (13/30), 95% CI [27.4%, 60.8%] | 0.000488 | SEPARATED |

Every discordant held-out pair under the Bonferroni rule changed from
control-clear to candidate-review: 16 pairs for Breast Cancer and 12 for Wine.
No seed changed in the opposite direction.

## Metric-level diagnosis

The adjustment reduced the Wine fixed-model control selections from four runs
to one while retaining 13 candidate selections. On Breast Cancer, all three
control selections were caused by `top_k_overlap`; its empirical run limit
remained unchanged because the 95th and 98.75th percentiles both fell on the
same discrete calibration value. Consequently, the adjustment did not reduce
the Breast Cancer control rate.

The candidate signal remained separated from the fixed-model control on both
datasets. Under the adjusted rule, cosine similarity caused most candidate
selections, either alone or together with another metric.

## Descriptive interpretation

The result shows that a multiplicity adjustment can reduce fixed-model review
flags without eliminating the observed cross-version signal, but the effect is
dataset dependent. Bonferroni improved the Wine control rate and had almost no
effect on Breast Cancer because a discrete overlap metric could not support a
finer empirical threshold with only 30 calibration runs.

`SEPARATED` remains an explanation-level advisory result. It does not show that
the candidate model is defective, unsafe or less accurate.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression lime-multiplicity
```

The command reads the completed held-out reports and writes:

- `artifacts/experiment_17_lime_multiplicity_policy/reports/summary.json`
- `artifacts/experiment_17_lime_multiplicity_policy/reports/policy_comparison.csv`
- `artifacts/experiment_17_lime_multiplicity_policy/reports/metric_limits.csv`
- `artifacts/experiment_17_lime_multiplicity_policy/reports/run_decisions.csv`
- `artifacts/experiment_17_lime_multiplicity_policy/figures/multiplicity_policy_comparison.png`

The complete analysis was executed twice after implementation. The generated
summary matched exactly on both runs (SHA-256:
`ab6abd968ed84497dda419907c5259695af61d1af35b1c2a4b6320d996ed5c75`).

## Limitations

- This is a post-hoc diagnostic motivated by the earlier observed control rates,
  not a preregistered confirmatory experiment.
- Thirty calibration seeds give coarse empirical quantiles; the nominal 98.75th
  percentile does not by itself guarantee an exact 5% family-wise error rate.
- The same held-out seeds are reused to compare the two rules, so another
  independent evaluation seed set is needed before choosing a final policy.
- `top_k_overlap` is discrete and its adjusted threshold did not change on the
  Breast Cancer dataset.
- The evidence still covers only two small tabular binary-classification tasks.
