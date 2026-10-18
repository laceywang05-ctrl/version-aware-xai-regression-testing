# Wine LIME Validation Result

## Material Passport

- Material ID: `wine-lime-validation-v1`
- Type: code experiment result
- Verification status: VERIFIED
- Dataset: scikit-learn Wine, classes 0 and 1 only
- Model family: random forest classifier
- Explainer: LIME tabular classification
- Execution: local CPU
- Reproducibility: exact summary checksum match on a repeated run

## Purpose

This experiment checks whether the explanation-regression workflow generalises
from Tree SHAP to LIME on the second dataset. It separates fixed-model LIME
randomness from the explanation change produced by a controlled candidate model.

## Design

- Binary Wine dataset: 130 samples and 13 numerical features.
- Stratified 70/30 train/test split with split seed 42.
- Twenty fixed probe samples.
- Reference model: 200-tree random forest, seed 42.
- Candidate model: 200-tree random forest, seed 99,
  `min_samples_leaf=5` and no depth limit.
- Ten LIME random seeds and 1,000 perturbation samples per explanation.
- The same LIME seed is paired across reference and candidate explanations.

## Fixed-model stability baseline

Across 200 matched sample-seed comparisons, the fixed-model means were:

| Metric | Mean |
| --- | ---: |
| Cosine similarity | 0.9752 |
| Feature-rank correlation | 0.7544 |
| Top-5 overlap | 0.8470 |
| Direction-flip rate | 0.2262 |

These values estimate LIME explainer variation while the trained model and probe
data remain fixed.

## Cross-version result

The reference and candidate models had identical predictions on all probe
samples. Both also achieved 1.000 accuracy, balanced accuracy, positive-class F1
and ROC AUC on this fixed test split.

| Metric | Fixed-model mean | Noise-matched cross-version mean | Paired-seed cross-version mean |
| --- | ---: | ---: | ---: |
| Cosine similarity | 0.9752 | 0.9684 | 0.9935 |
| Feature-rank correlation | 0.7544 | 0.7499 | 0.9307 |
| Top-5 overlap | 0.8470 | 0.8320 | 0.9330 |
| Direction-flip rate | 0.2262 | 0.2308 | 0.0654 |

Only cosine similarity crossed its calibrated run-level limit. Its candidate
breach rate was 16%, compared with a 10% run-level limit. The other three metrics
remained within their fixed-model variation limits. The combined advisory status
was therefore `REVIEW`.

## Interpretation

The candidate preserved predictions and test performance but produced a small
distributional change in LIME explanations relative to the fixed-model baseline.
The paired-seed comparison remained highly similar, suggesting that much of the
unpaired difference is still associated with LIME perturbation randomness.

The `REVIEW` result is an investigation signal, not evidence that the candidate
model is defective or less accurate. It also does not establish that LIME and
SHAP respond identically to this model change.

## Limitations

- The dataset and fixed test split are small; perfect test performance should not
  be interpreted as general performance.
- The LIME analysis uses ten explainer seeds and twenty probes, so run-level
  thresholds remain preliminary.
- The same seed set supports baseline calibration and the candidate comparison;
  a future held-out-seed LIME benchmark would provide a stronger validation.
- LIME settings such as kernel width, discretisation and perturbation count may
  change the observed variation.

## Outputs

- `artifacts/experiment_12_wine_lime_stability/reports/summary.json`
- `artifacts/experiment_12_wine_lime_stability/figures/lime_stability_distributions.png`
- `artifacts/experiment_13_wine_lime_cross_version/reports/summary.json`
- `artifacts/experiment_13_wine_lime_cross_version/reports/policy_evaluation.csv`
- `artifacts/experiment_13_wine_lime_cross_version/figures/lime_cross_version_comparison.png`
