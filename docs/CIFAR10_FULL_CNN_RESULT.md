# Fully trained CIFAR-10 CNN: validation and final test result

## Question and design

The earlier CIFAR-10 website demonstration used flattened RGB pixels and
shallow tree classifiers trained on at most 5,000 images. This experiment
tests a more suitable convolutional model on the **complete official training
pool**. It also supplies a concrete V1/V2 pair to the upload page.

- Data: official CIFAR-10 Python archive (`cifar-10-python.tar.gz`), 50,000
  training images and 10,000 official test images, each 32 × 32 RGB, across
  10 classes. The archive SHA-256, seed and split indices are saved in the run
  folder.
- Training/validation: a stratified, fixed seed-42 split of the official
  training pool into 45,000 training and 5,000 validation images. Channel
  normalization statistics use only the training subset.
- Model: a six-convolution-layer CNN with batch normalization, three pooling
  stages, dropout and global average pooling; **288,746 trainable
  parameters**. Training uses AdamW, a cosine learning-rate schedule,
  reflection-padded random crops and horizontal flips, batch size 256.
- Training: 80 epochs maximum. The run was paused after epoch 50 at the
  user's request and resumed from saved weights, optimizer and scheduler
  state. Because augmentation RNG state was not stored at that pause, the
  continuation is not bitwise equivalent to an uninterrupted run.
- Versions: **V1** is the epoch-5 checkpoint fixed before final evaluation.
  **V2** is selected by highest validation accuracy, with lower validation
  loss resolving a tie. The official test set was evaluated only after V2
  selection.

## Results

| Checkpoint | Validation accuracy | Official 10,000-image test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| V1, epoch 5 | 71.64% | 70.50% | 69.78% |
| V2, best epoch 74 | 89.50% | **88.54%** | **88.46%** |

Training finished at epoch 80; epoch 74 was retained because it had the best
validation accuracy. On the same 10,000 official test images, V2 accuracy was
**18.04 percentage points** higher than V1. For the paired correctness
comparison, 2,004 images changed from wrong to correct and 200 from correct
to wrong; 6,850 were correct under both and 946 were wrong under both.
Prediction agreement was 74.50%.

A paired multinomial bootstrap of the observed correctness transitions (20,000
draws, seed 20261030) gave a 95% percentile interval of **17.19 to 18.90
percentage points** for the accuracy change. The two-sided exact McNemar
probability is below 10^-300. These statistics describe this fixed test set
and the early-versus-later checkpoint comparison; they do not establish that
one explanation method is better or that the gain will repeat across new
training seeds.

## Website demonstration and verification

`artifacts/cifar10_cnn_full_2026-10-29/upload_bundle/` contains:

| File | Role |
| --- | --- |
| `cifar10_cnn_v1_epoch05.joblib` | Trusted, already trained V1 CNN |
| `cifar10_cnn_v2_best.joblib` | Trusted, validation-selected V2 CNN |
| `samples.csv` | First 1,000 official test images: ID, image filename, 3,072 RGB pixel features, true class |
| `images.zip` | Matching 1,000 RGB images for individual inspection |
| `manifest.json` | Column roles, image shape, class names and comparison metadata |

The full 10,000-image evaluation is in `summary.json` and
`test_predictions.csv` in the parent run folder. The upload sample is the
first 1,000 images, so its accuracy can differ from the full-test accuracy.
On this upload sample, V1 scored 70.4% and V2 88.7%.

The actual upload preflight accepted both serialized models, 1,000 rows and
3,072 features. The complete local comparison also ran successfully with the
image ZIP; it generated Grad-CAM maps for the first 100 rows per model.
Grad-CAM similarity uses an **exploratory, uncalibrated** threshold on this
CNN pair. Its drift flags invite visual review and are not automatic evidence
of a model regression. Both exported models separately passed a one-image
Integrated Gradients and Grad-CAM smoke test.

## Reproduction and files

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_full_2026-10-29 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --device mps
```

The completed run has `config.json`, `split_indices.npz`, `history.json`,
`training_curves.png`, epoch checkpoints, `summary.json`,
`test_predictions.csv`, `paired_statistics.json` and the `upload_bundle/`.
Generated files are Git-ignored; this document records the measured results.
The [epoch-50 interim record](CIFAR10_FULL_CNN_INTERIM.md) documents the pause.
