# MNIST multi-model upload validation

## Objective

This experiment validates the end-to-end upload workflow on the real MNIST
dataset rather than scikit-learn's smaller 8×8 digits dataset. The official
Keras-style split contains 60,000 training images and 10,000 test images. For a
lightweight local demonstration, the first 1,000 training and first 1,000 test
images were used; every image has 28×28 grayscale pixels.

Four scikit-learn classifiers were trained and uploaded together. Random Forest
and Extra Trees were then selected as the close-performing model pair for
sample-level and explanation comparison.

## Four-model overview

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Extra Trees | 84.8% | 84.4% |
| Random Forest | 84.7% | 84.4% |
| Decision Tree | 60.7% | 59.7% |
| Most-frequent dummy | 9.9% | 1.8% |

The multi-model view correctly ranks all four uploaded models on the same 1,000
labelled test images. The dummy model provides a sanity-check lower bound, not a
candidate for explanation-stability analysis.

## Random Forest versus Extra Trees

| Metric | Random Forest | Extra Trees |
|---|---:|---:|
| Accuracy | 84.7% | 84.8% |
| Macro F1 | 84.41% | 84.40% |

- Prediction agreement: 89.3%.
- Accuracy change: +0.1 percentage points.
- Discordant correctness cases: 85.
- Exact McNemar p-value: 1.0.
- Paired accuracy conclusion: inconclusive at α=0.05.
- Both correct: 805; Random-Forest-only correct: 42; Extra-Trees-only correct:
  43; both wrong: 110.

The two models therefore have practically identical aggregate performance on
this sample. The one extra correct prediction from Extra Trees is not evidence
of a reliable accuracy improvement.

## Explanation comparison

Tree SHAP maps were generated for the first 500 test images. Among all 500 maps,
mean cosine similarity was 0.585 and median similarity was 0.630. The models
made the same prediction for 449 of these samples; within that subset, mean
similarity was 0.654 and median similarity was 0.648.

At the current 0.80 inspection threshold:

- 455 of 500 explanation comparisons were flagged;
- 404 flags occurred while the predicted class stayed the same.

This does **not** show that 404 explanations are wrong. It shows that the 0.80
threshold developed for earlier controlled experiments does not transfer
directly to this MNIST cross-model-family comparison. Random Forest and Extra
Trees can have nearly identical accuracy while producing substantially
different normalized attribution patterns. MNIST therefore requires a
dataset- and comparison-specific calibration baseline before these flags can be
interpreted as failures.

## Reproduction

Download the Keras-style MNIST archive, then run:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input mnist.npz \
  --output-dir artifacts/mnist_model_demo \
  --limit 1000
```

Upload the four model files, `samples.csv`, `images.zip`, and `manifest.json`.
For the detailed comparison, select `model_3_random_forest.joblib` as the
baseline and `model_4_extra_trees.joblib` as the candidate.

## Limitations

- Only 1,000 training and 1,000 test images were used for this lightweight run.
- The models are conventional tree ensembles on flattened pixels, not modern
  image architectures.
- Tree SHAP processing was capped at the first 500 test images.
- Explanation maps are normalized separately and cosine similarity does not
  measure causal correctness.
- A benign-retraining calibration experiment is needed before setting an MNIST
  explanation-review threshold.

The follow-up benign-retraining calibration and revised review workload are
documented in
[`MNIST_EXPLANATION_CALIBRATION_RESULT.md`](MNIST_EXPLANATION_CALIBRATION_RESULT.md).
