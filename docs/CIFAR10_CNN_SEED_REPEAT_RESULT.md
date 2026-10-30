# CIFAR-10 CNN second-seed repeat

## Purpose and protocol

The first full CIFAR-10 CNN run used one random seed. This repeat checks
whether its large early-to-later checkpoint gain persists when the
training/validation split, weight initialization, minibatch order and data
augmentation use seed 43 instead of seed 42. The architecture, 45,000/5,000
stratified split sizes, preprocessing, optimizer and maximum 80 epochs are
unchanged. Both runs use the same untouched 10,000-image official test set.
V1 is fixed at epoch 5; V2 is chosen by validation accuracy before test
evaluation. Seed 42 was paused and resumed after epoch 50; seed 43 ran
uninterrupted. See [the first run](CIFAR10_FULL_CNN_RESULT.md) for full model
and dataset details.

## Full official-test results

| Seed | V1 epoch | V1 test accuracy | V2 epoch | V2 validation accuracy | V2 test accuracy | V2 macro F1 | V1→V2 accuracy gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 5 | 70.50% | 74 | 89.50% | 88.54% | 88.46% | +18.04 percentage points |
| 43 | 5 | 71.29% | 69 | 89.34% | 88.83% | 88.77% | +17.54 percentage points |

For seed 43, 1,980 of 10,000 test images changed from wrong under V1 to
correct under V2, while 226 changed from correct to wrong. The remaining
6,903 were correct under both and 891 wrong under both. A paired multinomial
bootstrap of the correctness transitions (20,000 draws, seed 20261031) gave
a 95% percentile interval of **+16.69 to +18.39 percentage points** for the
accuracy gain. The exact two-sided McNemar p-value is below 10^-300.

The two validation-selected V2 models had test accuracies differing by only
**0.29 percentage points** in favor of seed 43. On the same 10,000 images,
8,499 were correct under both, 762 wrong under both, 384 correct only under
seed 43 and 355 correct only under seed 42. Their class predictions agreed
on **90.92%** of images. A paired bootstrap for seed 43 minus seed 42
accuracy gave a 95% interval of **-0.24 to +0.82 percentage points**;
the exact paired McNemar p-value was **0.303**. This does not establish
equivalence or a seed-independent accuracy estimate.

## Reproduction and upload check

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_full_seed43 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --seed 43 --device mps
```

The local output includes `config.json`, `split_indices.npz`, `history.json`,
`summary.json`, `test_predictions.csv`, `paired_statistics.json`,
`cross_seed_comparison.json`, checkpoints, training curves, and
`upload_bundle/`. The new bundle passed the actual upload preflight: both
trusted `.joblib` models were compatible with its 1,000-row, 3,072-feature
CSV, and the ZIP contained 1,000 matching images. Generated artifacts are
Git-ignored and remain local.

For a concrete dashboard demonstration, the bundle's first 1,000 official
test images have V1 accuracy **71.6%** and V2 accuracy **90.1%**. Among these
images, **201** changed from wrong to correct and **16** from correct to
wrong. For `image-00003` (true class **airplane**), V1 predicted **ship**
with 74.2% confidence; V2 predicted **airplane** with 98.0% confidence.
This is a clear individual improvement. For `image-00032` (true class
**deer**), V1 predicted **deer** with 52.9% confidence, while V2 predicted
**bird** with 47.3% confidence. This counterexample shows why an aggregate
gain does not remove the need to inspect regressions. The dashboard can show
both images from `images.zip` alongside their predictions and explanations.

## Interpretation

The substantial V1-to-V2 predictive gain appeared in both seeds, while their
final V2 test accuracies were close. Two seeds are only a limited repeat;
the paired intervals above quantify sampling uncertainty for these fixed
models on the official test set, not variation across possible training
runs. The experiment also does not calibrate a Grad-CAM review threshold or
show that one explanation method is better than another.
