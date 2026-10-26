# CIFAR-10 multi-model upload validation

## Objective

This experiment validates the upload and comparison workflow on the official
CIFAR-10 Python dataset. Unlike the grayscale MNIST experiment, CIFAR-10 uses
32×32 RGB images, ten object classes, and 3,072 flattened pixel features per
sample. The official training and test batches were preserved. For a
lightweight local demonstration, the first 1,000 training images and first
1,000 test images were used.

The official `cifar-10-python.tar.gz` archive was read directly without being
extracted. Four scikit-learn classifiers were trained and uploaded together.
The two closest ensemble models were then selected for paired comparison.

## Four-model overview

| Model | Accuracy | Macro F1 | ECE |
|---|---:|---:|---:|
| Random Forest | 33.4% | 32.5% | 6.1% |
| Extra Trees | 32.1% | 30.9% | 4.7% |
| Decision Tree | 21.2% | 18.4% | 19.6% |
| Most-frequent dummy | 8.9% | 1.6% | 91.1% |

The ranking is internally consistent: both ensemble methods outperform the
single decision tree, and all trained models outperform the most-frequent
baseline. The absolute scores are intentionally modest because these are
lightweight tree models trained on only 1,000 flattened images, not modern
convolutional image classifiers.

## Extra Trees versus Random Forest

Extra Trees was used as V1 and Random Forest as V2.

| Metric | Extra Trees (V1) | Random Forest (V2) |
|---|---:|---:|
| Accuracy | 32.1% | 33.4% |
| Macro F1 | 30.88% | 32.52% |

- Prediction agreement: 57.1%.
- Accuracy change: +1.3 percentage points.
- Discordant correctness cases: 187.
- Exact McNemar p-value: 0.3803.
- Paired accuracy conclusion: inconclusive at α=0.05.
- Both correct: 234; V1-only correct: 87; V2-only correct: 100; both wrong:
  579.

Random Forest made 13 more correct predictions, but the paired evidence does
not establish a reliable improvement. The result should therefore be reported
as an observed advantage rather than a confirmed model-version improvement.

## Class-level changes

The largest V2 accuracy gains were:

- automobile: +11.2 percentage points;
- dog: +7.0 percentage points;
- cat: +5.8 percentage points;
- bird: +5.0 percentage points.

The largest decreases were:

- ship: −5.7 percentage points;
- horse: −4.9 percentage points;
- truck: −2.8 percentage points.

This demonstrates why aggregate accuracy alone is insufficient: the candidate
can improve overall while still regressing for particular classes.

## Confidence and explanation observations

The ensemble models had ECE values of 6.1% and 4.7% on this sample. Neither
model produced an incorrect prediction at or above the dashboard's 80%
high-confidence threshold, so no high-confidence regression was recorded.

RGB Tree SHAP maps were generated for the first 100 test images. Contributions
from the red, green and blue feature channels were summed at each spatial pixel,
producing one 32×32 heatmap per model and sample. All 100 pairs were comparable:

- mean cosine similarity: 0.088;
- median cosine similarity: 0.091;
- 100 of 100 pairs fell below the provisional 0.80 review threshold;
- 61 pairs retained the same predicted class, and all 61 fell below 0.80.

These results demonstrate that RGB heatmaps can now be inspected in the same
sample-level dialog as grayscale explanations. They do **not** establish that
all 100 explanations failed. The models use different ensemble algorithms, and
the 0.80 boundary was not calibrated for CIFAR-10 RGB explanations. The flags
therefore identify a cross-model spatial-attribution difference that requires a
CIFAR-specific benign-retraining baseline before it can be interpreted as
regression evidence.

The follow-up benign-retraining experiment calibrated the fifth-percentile
boundary at 0.047. Under that dataset-specific threshold, 4 of 61
same-prediction cross-model comparisons were flagged (6.6%), rather than all
61. The full calibration design and held-out validation are documented in
[`CIFAR10_EXPLANATION_CALIBRATION_RESULT.md`](CIFAR10_EXPLANATION_CALIBRATION_RESULT.md).

## Reproduction

Download the official CIFAR-10 Python archive and run:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_model_demo \
  --limit 1000
```

Upload the four `.joblib` files, `samples.csv`, `images.zip`, and
`manifest.json`. For the detailed comparison, select
`model_4_extra_trees.joblib` as the baseline and
`model_3_random_forest.joblib` as the candidate.

## Limitations

- Only 1,000 training and 1,000 test images were used.
- Flattened pixels discard spatial structure and are not a competitive CIFAR-10
  modelling approach.
- The two ensemble models are different model families rather than two releases
  of one production model.
- The observed +1.3-point change was not statistically significant.
- RGB channel aggregation can hide channel-specific attribution differences.
- The 0.80 explanation threshold is provisional and was not calibrated for
  CIFAR-10.
- Uploaded `.joblib` models and CIFAR pickle archives must come from trusted
  sources.

The experiment validates multi-model ingestion, RGB sample display, ranking,
paired classification statistics, confusion diagnostics, and per-sample error
inspection on a second real image dataset. A future experiment can replace the
lightweight estimators with stronger image models while keeping the same
comparison workflow.
