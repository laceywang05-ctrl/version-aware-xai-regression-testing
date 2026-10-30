# CIFAR-10 full-training experiment: paused after epoch 50

This is the historical pause record. Training subsequently completed; see
[`CIFAR10_FULL_CNN_RESULT.md`](CIFAR10_FULL_CNN_RESULT.md) for the final result.

This is an interim **validation** result. The user requested a pause after
epoch 50 on 2026-10-30. The official test set has **not** been evaluated for
this experiment; final model selection and test results remain pending.

## Protocol

- Source: official CIFAR-10 Python archive, SHA-256 recorded in the run's
  `config.json`.
- Official data: 50,000 training images and 10,000 test images; 32 × 32 RGB,
  10 classes.
- Stratified split of the official training pool: 45,000 training and 5,000
  validation images, seed 42. Normalization statistics come only from the
  45,000 training images.
- CNN: six convolutional layers, three pooling stages, batch normalization,
  dropout and global average pooling; **288,746 trainable parameters**.
- AdamW, cosine learning-rate schedule, random crop and horizontal flip;
  batch size 256. V1 is the fixed epoch-5 checkpoint. V2 will be selected by
  highest validation accuracy after training finishes.

## Validation progress

| Epoch | Augmented training accuracy | Validation accuracy | Validation loss |
| ---: | ---: | ---: | ---: |
| 5 | 66.45% | 71.64% | 0.8091 |
| 30 | 83.49% | 86.50% | 0.3865 |
| 50 | 86.66% | **88.80%** | 0.3376 |

Epoch 50 is the current best checkpoint. The validation accuracy increased by
17.16 percentage points from the fixed epoch-5 checkpoint. This is **not** a
final test-set improvement claim. The augmented training accuracy and
unaugmented validation accuracy are measured under different image
transformations, so their ordering is not an overfitting test by itself.

## Pause and resume

The run folder is `artifacts/cifar10_cnn_full_2026-10-29/`. It contains
`last.pt` (epoch 50 and optimizer/scheduler state), `best.pt`, `baseline.pt`,
`history.json`, `progress.json`, `split_indices.npz`, `config.json`, and a
preview `training_curves.png`. It is Git-ignored. To continue from epoch 51:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_full_2026-10-29 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --device mps --resume
```

The current implementation restores weights, optimizer and scheduler state,
but augmentation randomness after a restart is not bitwise identical to an
uninterrupted run. The split, data, hyperparameters and validation policy stay
fixed. The complete 10,000-image test set is evaluated only after the resumed
training selects the final checkpoint.
