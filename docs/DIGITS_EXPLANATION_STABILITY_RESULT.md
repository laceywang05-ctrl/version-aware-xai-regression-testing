# Handwritten-digit explanation stability

## Objective

This experiment checks whether closely performing handwritten-digit model
versions rely on similar pixel-contribution patterns. It specifically looks for
hidden explanation drift: the two versions predict the same class, but the
direction of their explanation vectors has cosine similarity below 0.80.

V1, V2 and V3 are trained on the same seed-42 split. Their Tree SHAP
predicted-class contribution maps are compared on the same first 120 test
samples.

## Results

| Comparison | Mean cosine | Median cosine | 10th percentile | Prediction agreement | Below 0.80 | Hidden drift |
|---|---:|---:|---:|---:|---:|---:|
| V1 vs V2 | 0.939 | 0.945 | 0.903 | 100.0% | 0 | 0 |
| V1 vs V3 | 0.931 | 0.943 | 0.899 | 99.2% | 1 | 0 |
| V2 vs V3 | 0.954 | 0.967 | 0.939 | 99.2% | 1 | 0 |

The three pairs have high average explanation similarity. Two individual
comparisons fall below the inspection threshold, but both occur where the
models also predict different classes. No tested sample therefore meets the
experiment's definition of hidden explanation drift.

The threshold is an inspection rule, not a statistically calibrated failure
boundary. A value below 0.80 should trigger review rather than automatically
prove that one explanation is incorrect.

## Interpretation

For these close Random Forest versions and this sample, prediction stability is
accompanied by explanation stability. V1 and V2 agree on every tested class,
while V1/V3 and V2/V3 each disagree once. The experiment does not establish
that explanation drift is impossible; it shows that the current controlled
versions did not expose hidden drift under the stated threshold.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression digits-explanation-stability \
  --output-dir artifacts/digits_explanation_stability
```

The machine-readable report is saved to
`artifacts/digits_explanation_stability/summary.json`.

## Limitations

- Only 120 samples from one fixed train-test split are inspected.
- The compared versions are closely related Random Forest configurations.
- Predicted-class maps are not directly equivalent when two versions predict
  different classes, so hidden drift is counted only for matching predictions.
- Cosine similarity measures direction, not the absolute scale of attribution.
- The 0.80 inspection threshold is pragmatic and has not been calibrated from
  expert-labelled acceptable and unacceptable explanation changes.
