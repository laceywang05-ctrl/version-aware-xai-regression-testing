# CIFAR-10 RGB explanation calibration

## Objective

The initial CIFAR-10 comparison used the dashboard's provisional 0.80 cosine
similarity threshold. All 100 Random-Forest-versus-Extra-Trees explanation
pairs fell below that boundary, including 61 cases where both models predicted
the same class. This experiment tests whether that result reflects harmful
explanation drift or a threshold that does not transfer to RGB models.

The calibration baseline uses benign retraining: every model has the same
Random Forest configuration, training images and preprocessing, while only the
random seed changes. RGB Tree SHAP contributions are summed across the three
channels at each pixel to form comparable 32×32 spatial maps.

## Experimental design

- Dataset: official CIFAR-10 Python training and test batches.
- Training subset: first 1,000 official training images.
- Explanation subset: first 50 official test images.
- Model: `RandomForestClassifier` with 60 trees.
- Calibration seeds: 11, 29 and 47.
- Held-out validation seeds: 59 and 71.
- Calibration statistic: fifth percentile of same-prediction cosine
  similarities.

The calibration and validation seeds are disjoint. The threshold is therefore
estimated without using the model pair that measures its held-out flag rate.

## Calibration result

The three calibration models produced three pairs and 81 same-prediction
explanation comparisons.

| Metric | Result |
|---|---:|
| Mean similarity | 0.140 |
| Median similarity | 0.124 |
| Fifth-percentile threshold | **0.047** |
| Comparisons below the old 0.80 threshold | 100.0% |

Pair-level mean similarity ranged from 0.133 to 0.147. Pair-level prediction
agreement ranged from 48% to 60%. The very low absolute similarities show that
RGB spatial SHAP maps are sensitive even to benign retraining in this small,
flattened-pixel model setting. Consequently, the earlier 0.80 threshold is not
a defensible CIFAR-10 review boundary.

## Held-out validation

Seeds 59 and 71 produced 29 same-prediction comparisons that were not used to
set the threshold.

| Metric | Result |
|---|---:|
| Mean similarity | 0.123 |
| Median similarity | 0.116 |
| Flags below 0.047 | 2 of 29 |
| Held-out benign flag rate | **6.9%** |

The observed 6.9% rate is close to the intended 5% lower-tail policy, although
the validation sample is small. This supports 0.047 as a provisional
CIFAR-10-specific review threshold rather than a universal boundary.

## Reinterpretation of the cross-model comparison

Applying 0.047 to the 100 Random-Forest-versus-Extra-Trees maps gives:

- 39 of 100 flags when all pairs are included;
- 4 of 61 flags when restricted to hidden drift, where predictions agree;
- a hidden-drift review rate of 6.6%.

Explanation maps for changed predictions describe contributions toward
different predicted classes and should not be interpreted as the same kind of
stability test. The same-prediction result is therefore the more appropriate
review workload. At the calibrated threshold, it is close to benign variation
and does not support a claim of abnormal cross-model explanation drift.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression cifar10-explanation-calibration \
  --input cifar-10-python.tar.gz
```

The JSON report is written to
`artifacts/cifar10_explanation_calibration/summary.json`.

## Interpretation and limitations

- The 0.047 threshold is specific to this dataset, explanation representation,
  model configuration and training size.
- Only 50 test images and one held-out seed pair were used.
- Summing RGB channels hides channel-specific attribution differences.
- The flattened-pixel Random Forest is a lightweight demonstration model, not a
  competitive CIFAR-10 architecture.
- A review flag selects a case for inspection; it is not proof of a model defect
  or harmful drift.

The main finding is methodological: explanation thresholds must be calibrated
against natural variation for the same data and explanation pipeline. A
threshold transferred from another dataset can turn ordinary retraining noise
into a misleading 100% alert rate.
