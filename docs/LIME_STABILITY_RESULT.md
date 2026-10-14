# LIME Fixed-Model Stability Result

## Purpose

This experiment estimates variation caused by the LIME explainer itself. The
trained random forest, training data, fixed probe samples and LIME settings are
held constant; only LIME's random seed changes.

## Design

- Dataset: Breast Cancer Wisconsin Diagnostic
- Model: one fixed random forest (`random_state=42`)
- Probe set: 20 fixed samples
- Baseline LIME seed: 42
- Repeated LIME seeds: 10
- Perturbed samples per explanation: 1,000
- Matched explanation comparisons: 200
- Compared class: malignant

## Results

| Metric | Mean |
|---|---:|
| Cosine similarity | 0.8494 |
| Absolute rank correlation | 0.5843 |
| Top-5 feature overlap | 0.7680 |
| Direction-flip rate | 0.2537 |

The per-seed mean cosine similarity ranged from 0.8362 to 0.8676. Even though
the model did not change, the feature-weight vectors, feature rankings and
contribution directions varied across LIME runs.

## Interpretation

These results are a natural-variation baseline for LIME, not evidence of model
drift or explanation regression. A version-aware regression policy should
compare cross-version changes against this baseline and should control the LIME
seed, sampling size, kernel, discretisation and probe order. A difference
between reference and candidate explanations should trigger investigation only
when it is meaningfully larger than repeated-run explainer variation.

This is an initial experiment with 20 probe samples and one dataset. The values
are provisional and should not be treated as universal thresholds.
