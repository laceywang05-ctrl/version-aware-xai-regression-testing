# Held-out LIME Policy Validation

## Material Passport

- Material ID: `lime-heldout-policy-validation-v1`
- Type: repeated code experiment
- Verification status: `VERIFIED`
- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine
- Explanation method: LIME
- Calibration seeds: 30 per dataset
- Evaluation seeds: 30 per dataset, disjoint from calibration
- Fixed probes: 20 samples per dataset
- LIME perturbation samples: 1,000 per explanation
- External data upload: none

## Research purpose

The earlier LIME comparisons calibrated and evaluated the policy using the same
set of explainer seeds. This follow-up tests whether the signal persists when
the policy thresholds are estimated from one seed set and evaluated on a
completely separate seed set.

Each held-out evaluation seed produces two paired runs:

1. a fixed-model control using the reference model;
2. a cross-version run using the candidate model.

A run is selected for review when at least one metric's sample breach rate is
above its calibration-derived run-level limit. Control and candidate review
flags are compared with an exact paired McNemar test. Review-rate uncertainty is
reported using Wilson 95% confidence intervals.

## Results

| Dataset | Prediction agreement | Held-out control review rate | Candidate review rate | Exact paired p | Status |
|---|---:|---:|---:|---:|---|
| Breast Cancer Wisconsin | 95.0% | 10.0% (3/30), 95% CI [3.5%, 25.6%] | 66.7% (20/30), 95% CI [48.8%, 80.8%] | 0.000015 | SEPARATED |
| Wine (binary) | 100.0% | 13.3% (4/30), 95% CI [5.3%, 29.7%] | 63.3% (19/30), 95% CI [45.5%, 78.1%] | 0.000061 | SEPARATED |

In both datasets, every discordant pair changed from control-clear to
candidate-review: 17 such pairs for Breast Cancer and 15 for Wine. No paired
seed changed in the opposite direction.

## Explanation metrics

| Dataset | Group | Mean cosine | Mean rank correlation | Mean top-5 overlap | Mean direction-flip rate |
|---|---|---:|---:|---:|---:|
| Breast Cancer | Held-out control | 0.8515 | 0.5850 | 0.7663 | 0.2602 |
| Breast Cancer | Candidate | 0.8078 | 0.5510 | 0.7163 | 0.2711 |
| Wine | Held-out control | 0.9753 | 0.7661 | 0.8450 | 0.2165 |
| Wine | Candidate | 0.9688 | 0.7629 | 0.8280 | 0.2313 |

Cosine similarity was the only metric whose aggregate candidate breach rate
exceeded its calibration-derived run-level limit in both datasets. The paired
same-seed cosine similarities remained high (0.9514 for Breast Cancer and
0.9935 for Wine), confirming that some LIME variation is removed when the
reference and candidate use identical perturbation seeds.

## Descriptive interpretation

The candidate review rates were substantially higher than the held-out
fixed-model control rates on both datasets. This provides stronger evidence
that the implemented policy detects a repeatable change in the LIME explanation
distribution rather than only re-identifying the calibration seeds.

`SEPARATED` is an explanation-level result. It does not establish that the
candidate model is defective, unsafe or less accurate. On Wine, reference and
candidate predictive metrics were identical despite the explanation signal.

## Control-rate note

The held-out control review rates were 10.0% and 13.3%, not zero. A run can be
selected when any of four metrics breaches its limit, so the family-level review
rate can exceed the nominal tail probability of one metric. The Wilson
intervals are also wide with 30 runs. These control rates should therefore be
reported rather than hidden and considered when selecting production policy
thresholds.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/lime_heldout.yaml

PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/wine_lime_heldout.yaml
```

## Output directories

- `artifacts/experiment_15_breast_lime_heldout/`
- `artifacts/experiment_16_wine_lime_heldout/`

Each directory contains the calibration samples, held-out control and candidate
samples, paired-seed comparisons, per-run decisions, policy table, summary JSON
and a validation figure.

The complete experiments were repeated after implementation. The summary files
matched exactly before and after rerunning:

- Breast Cancer SHA-256: `ec754a0751b3aba8bba1b4dd8dc1dd539a5de7a344b5fb95380796970fd5dfc9`
- Wine SHA-256: `5a0d158fa0a9ae4a4b7dd2804f88f1b047cf7c94e8ec09a74a25bd3c166d9c48`

## Limitations

- Both experiments reuse one fixed train/test split and one fixed probe subset.
- Only one candidate model configuration is evaluated per dataset.
- Thirty held-out seeds still produce relatively wide review-rate intervals.
- The current policy does not adjust the run decision for simultaneously
  checking four explanation metrics.
- Both datasets are small tabular binary-classification tasks.
