# Third matched CIFAR-10 CNN width comparison (seed 44)

## Question and protocol

This is a third matched run of the [CNN width-change experiment](CIFAR10_WIDTH_REPEAT_RESULT.md). Within seed 44, V1 uses 32/64/128 convolution-stage channels (288,746 trainable parameters) and V2 uses 40/80/160 (450,210 parameters). Both use the same official CIFAR-10 archive, 45,000/5,000 stratified train/validation split, optimizer, augmentation, batch size, learning-rate schedule, and untouched 10,000-image official test set. The train and validation index arrays were checked for exact equality across the pair. Each model trained continuously for 80 epochs; its final checkpoint was selected by validation accuracy, with validation loss breaking ties. Neither run was resumed.

The training command now saves CPU, NumPy, and active-device RNG states in each epoch checkpoint so an interrupted *new* run can restore its augmentation and dropout streams. This change does not retroactively repair the seed 42 run, and it does not establish bitwise reproducibility across PyTorch versions or hardware. Width is the only intentional model-design change, although the different tensor shapes also change initialization RNG consumption.

## Official-test results

| Seed | V1 best epoch | V2 best epoch | V1 accuracy | V2 accuracy | Gain | V1 macro F1 | V2 macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 74 | 60 | 88.54% | 90.11% | +1.57 pp | 88.46% | 90.04% |
| 43 | 69 | 69 | 88.83% | 89.94% | +1.11 pp | 88.77% | 89.89% |
| **44** | **73** | **73** | **88.85%** | **90.08%** | **+1.23 pp** | **88.78%** | **90.02%** |

For seed 44, best validation accuracy was 88.88% for V1 and 89.52% for V2. Official-test cross-entropy loss was 0.3368 versus 0.3039. Across the three observed pairs, the accuracy gain ranged from 1.11 to 1.57 percentage points (descriptive mean 1.30 pp). Three runs are too few to estimate training-run uncertainty precisely; seed 42 also has the previously documented interrupted-run caveat.

| Seed | Changed predicted class | Wrong → correct | Correct → wrong | Both wrong, different class | Paired 95% CI for accuracy gain | Exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 891 (8.91%) | 442 | 285 | 164 | +1.03 to +2.11 pp | 6.34 × 10^-9 |
| 43 | 880 (8.80%) | 417 | 306 | 157 | +0.58 to +1.64 pp | 4.16 × 10^-5 |
| **44** | **855 (8.55%)** | **411** | **288** | **156** | **+0.71 to +1.75 pp** | **3.73 × 10^-6** |

For seed 44, 8,597 images were correct under both models and 704 wrong under both. The 411 fixes minus 288 regressions account for the net 123 additional correct images. The paired interval and McNemar test describe sampling uncertainty for **this fixed pair** on the 10,000 test images. The three pairs share the same official test set, so their p-values must not be combined as independent replications. All ten classes had positive observed recall changes in seed 44, from +0.1 pp for airplane, ship and truck to +3.7 pp for cat; these class-level changes are descriptive.

Frequent seed 44 prediction paths show why the actual label matters: `dog → cat` occurred 75 times (47 fixes, 23 regressions, 5 still wrong), while `deer → bird` occurred 23 times (10 fixes, 12 regressions, 1 still wrong). A direction of class change is not intrinsically good or bad.

## Dashboard subset and explanations

The generated upload bundle contains the first 1,000 official test images. In this subset, **85** predictions changed class: **38** fixes, **26** regressions and **21** changes between two wrong classes. These subset counts must not be confused with the 10,000-image results above. Example test image 33 is a dog: V1 predicted cat and V2 dog. Image 924 is a dog: V1 predicted dog and V2 airplane.

For each of the 1,000 images, the offline audit compared Grad-CAM maps for the **same true-class logit**. The mean cosine similarity was 0.854 for the 85 changed predictions and 0.934 for the 915 unchanged predictions (medians 0.877 and 0.951). Image 547 remained correctly classified as frog by both versions but its same-target map similarity was 0.608. These are exploratory map comparisons, not a calibrated automatic quality gate or evidence that a particular explanation is better. The website's interactive viewer instead explains each model's predicted class and therefore answers a different question when predictions differ.

## Reproduce and inspect

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_full_seed44 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --seed 44 --widths 32 64 128 --device mps

MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_wide_seed44 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --seed 44 --widths 40 80 160 --device mps

MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python scripts/compare_cifar10_versions.py \
  --baseline-run artifacts/cifar10_cnn_full_seed44 \
  --candidate-run artifacts/cifar10_cnn_wide_seed44 \
  --output-dir artifacts/cifar10_directional_version_comparison_seed44

MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression demo-check \
  --bundle-dir artifacts/cifar10_directional_version_comparison_seed44/upload_bundle
```

The local preflight passed: 2 models, 1,000 labelled images, 3,072 pixel features and 1,000 display images. The exact exported model SHA-256 values are `82068892c55a789ebfc902b0db923cb90ac9c218d1a0dcbd240d495edcf0ae1d` (V1) and `f244b581781738fc8ff3413c2cec3767b333ffb64309def3bcd70c1be5035a60` (V2). The source archive SHA-256 is `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`. Generated checkpoints, models, images and full JSON summaries stay under Git-ignored `artifacts/`.
