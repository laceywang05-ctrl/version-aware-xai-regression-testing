# Gradient Boosting Model-Family Validation

## Purpose

The earlier controlled Tree SHAP benchmarks used only random forests. This
experiment repeats the same held-out calibration and paired-candidate design
with `GradientBoostingClassifier` to test whether the monitoring result is tied
to one model family.

The experiment does not compare random forests and gradient boosting directly.
Instead, it asks whether the same regression-testing method can separate normal
retraining variation from controlled structural changes within a second model
family.

## Design

- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine.
- Explainer: Tree SHAP for the configured positive class.
- Reference model: 100 gradient-boosting stages with depth 3 trees.
- Calibration: 30 benign retraining seeds.
- Evaluation: 30 disjoint held-out seeds per scenario.
- Control: same model settings with only the random seed changed.
- Candidates: depth reduced to 1, minimum leaf size increased to 5, and both
  constraints combined.
- Decision metrics: cosine similarity, absolute feature-rank correlation,
  top-5 overlap, and contribution-direction flip rate.
- Statistical comparison: paired exact McNemar tests against the seed-only
  control, with Bonferroni adjustment across the three candidates.

Class balancing is implemented through per-sample weights because scikit-learn's
gradient-boosting classifier does not expose the random-forest
`class_weight` parameter.

## Results

| Dataset | Scenario | Review rate | 95% Wilson CI | Prediction agreement | Mean cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Breast cancer | Seed-only control | 7/30 (23.3%) | 11.8%–40.9% | 100.0% | 0.999 |
| Breast cancer | Shallow depth | 30/30 (100%) | 88.6%–100% | 98.0% | 0.897 |
| Breast cancer | Larger leaves | 30/30 (100%) | 88.6%–100% | 99.8% | 0.970 |
| Breast cancer | Combined constraints | 30/30 (100%) | 88.6%–100% | 98.0% | 0.882 |
| Wine | Seed-only control | 0/30 (0%) | 0%–11.4% | 100.0% | 0.994 |
| Wine | Shallow depth | 30/30 (100%) | 88.6%–100% | 100.0% | 0.812 |
| Wine | Larger leaves | 30/30 (100%) | 88.6%–100% | 100.0% | 0.783 |
| Wine | Combined constraints | 30/30 (100%) | 88.6%–100% | 100.0% | 0.819 |

For Breast Cancer, every candidate increased the paired review rate by 76.7
percentage points relative to control. Each Bonferroni-adjusted McNemar p-value
was approximately `7.15e-7`.

For Wine, every candidate increased the paired review rate by 100 percentage
points. Each adjusted p-value was approximately `5.59e-9`.

## Interpretation

The policy separates all three controlled structural changes from the
seed-only control on both datasets. The Wine result is particularly useful:
prediction agreement remains 100% in every scenario, while the Tree SHAP
attributions change enough for every candidate run to be reviewed. The signal
therefore cannot be reduced to changed predicted labels.

The result extends the evidence beyond random forests, but it does not prove
universal model-family generalisation. Gradient boosting has less seed-driven
variation than a random forest, so its empirical baseline is narrow. The 23.3%
control review rate on Breast Cancer also shows that the current per-run policy
still produces non-trivial warnings under benign retraining. Review flags remain
triage signals, not automatic declarations of harmful regression.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/gradient_boosting_breast_calibration.yaml \
  --scenarios config/gradient_boosting_breast_scenarios.yaml

PYTHONPATH=src python -m xai_regression benchmark \
  --config config/gradient_boosting_wine_calibration.yaml \
  --scenarios config/gradient_boosting_wine_scenarios.yaml
```

Detailed CSV, JSON and figure outputs are written under
`artifacts/experiment_28_breast_gradient_boosting/` and
`artifacts/experiment_29_wine_gradient_boosting/`.
