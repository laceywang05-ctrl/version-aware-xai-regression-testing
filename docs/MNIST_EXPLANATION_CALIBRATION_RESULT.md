# MNIST explanation calibration

## Objective

The initial MNIST comparison reused a cosine-similarity threshold of 0.80 from
earlier experiments and flagged most same-prediction explanations. This
experiment estimates a MNIST-specific review boundary from benign retraining:
the model configuration and data stay fixed while only the Random Forest
training seed changes.

Four 60-tree Random Forest models were trained on the first 1,000 official
training images. All six model pairs were compared on the same first 200
official test images. Calibration uses only samples where both models predicted
the same class.

## Calibration result

| Quantity | Result |
|---|---:|
| Benign model pairs | 6 |
| Same-prediction explanation comparisons | 1,057 |
| Mean cosine similarity | 0.657 |
| Median cosine similarity | 0.672 |
| Fifth-percentile similarity | 0.486 |
| Benign comparisons below 0.80 | 97.35% |

The earlier 0.80 threshold is inappropriate for this MNIST setup: it would flag
almost every benign retraining comparison. Using the pooled fifth percentile
gives a provisional MNIST inspection threshold of 0.486. This is designed to
place approximately 5% of benign same-prediction comparisons into review.

## Applying the calibrated threshold

The Random Forest versus Extra Trees upload comparison contained 449
same-prediction samples with explanation maps. The original 0.80 threshold
flagged 404 of them. At the calibrated 0.486 threshold, 26 samples are flagged,
or 5.79%.

This reduction supports the interpretation that most original warnings were
caused by threshold transfer rather than automatically harmful explanation
drift. The remaining 26 samples are a focused review queue; they are not proof
that Extra Trees is defective.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression mnist-explanation-calibration \
  --input mnist.npz \
  --output-dir artifacts/mnist_explanation_calibration \
  --train-limit 1000 \
  --sample-limit 200
```

## Limitations

- Calibration uses four training seeds and one Random Forest configuration.
- The 1,057 comparisons are correlated because model pairs reuse fitted models
  and test images.
- The fifth percentile controls review workload under observed benign
  retraining; it does not estimate the probability of harmful drift.
- Broader calibration should include more seeds, training samples and model
  configurations.
