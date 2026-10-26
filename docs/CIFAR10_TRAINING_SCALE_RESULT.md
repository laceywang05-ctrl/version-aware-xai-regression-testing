# CIFAR-10 training-scale result

## Question

The first CIFAR-10 upload demonstration trained lightweight classifiers on only
1,000 images and reached 33.4% best accuracy. This follow-up asks how much of
that result is explained by the deliberately small training sample.

## Design

- Dataset: official CIFAR-10 Python batches
- Training sizes: 1,000, 2,500 and 5,000 images
- Test sample: the same first 1,000 official test images in every run
- Models: most-frequent dummy, depth-5 decision tree, 60-tree random forest and
  60-tree extra-trees classifier
- Inputs: flattened 32×32 RGB pixels, without learned image features

Holding the test sample and model settings fixed means the rows isolate the
effect of adding training data within this lightweight demonstration design.

## Results

| Training images | Best model | Accuracy | Macro F1 | Fit time |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | Random forest | 33.4% | 32.5% | 1.56 s |
| 2,500 | Extra trees | 35.2% | 33.9% | 1.65 s |
| 5,000 | Extra trees | 37.7% | 36.9% | 3.85 s |

The complete four-model results were:

| Training images | Dummy | Decision tree | Random forest | Extra trees |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 8.9% | 21.2% | 33.4% | 32.1% |
| 2,500 | 9.0% | 25.9% | 34.6% | 35.2% |
| 5,000 | 10.6% | 24.7% | 37.6% | 37.7% |

## Interpretation

Increasing the training sample from 1,000 to 5,000 improved the best accuracy
by 4.3 percentage points and the best macro F1 by about 4.4 points. Extra trees
also trained faster than the random forest at the larger sizes while reaching
similar or slightly better accuracy.

The gain is useful but does not make the classifiers competitive image models.
The main limitation is architectural: the demonstration feeds raw flattened
pixels into shallow tree ensembles. It is designed to validate the upload,
multi-model ranking, error inspection and explanation workflow—not to claim
state-of-the-art CIFAR-10 recognition. A later production-oriented extension
could accept pretrained convolutional or embedding-based models while keeping
the same comparison interface.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression cifar10-training-scale \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_training_scale \
  --test-limit 1000
```

The machine-readable result is written to
`artifacts/cifar10_training_scale/summary.json`.
