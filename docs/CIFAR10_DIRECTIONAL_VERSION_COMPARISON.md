# CIFAR-10 model-version comparison: wider CNN versus original CNN

## Question and model change

Can a model update change **which classes are predicted**, even when the overall
test loss and accuracy move only moderately? This experiment compares two
separately trained, validation-selected CNN checkpoints on the same official
CIFAR-10 test images. V1 has convolution-stage widths **32/64/128** and
**288,746 trainable parameters**. V2 increases those widths to **40/80/160**
and has **450,210 parameters** (+55.9%). Both have six convolutional layers,
the same input/output contract, AdamW training settings, the same nominal seed
42, and the same 45,000/5,000 stratified train/validation split. The official
10,000-image test set was used only after validation-based checkpoint selection.

Width is the only intentional design change. This is an example of a genuine
model-version change rather than a comparison of two random seeds. It is not
a perfectly isolated causal experiment: changing tensor shapes changes the
initialization random-number stream, and the original run resumed after epoch
50 without restoring augmentation RNG state. These factors may contribute to
the observed differences. A repeated, controlled training study would be
needed to estimate an architecture-only effect.

## Full official-test results

| Metric | V1 original | V2 wider | Difference |
| --- | ---: | ---: | ---: |
| Validation-selected epoch | 74 | 60 | — |
| Best validation accuracy | 89.50% | 90.56% | +1.06 percentage points |
| Test accuracy | 88.54% | 90.11% | **+1.57 percentage points** |
| Test loss | 0.3420 | 0.3117 | -0.0303 |
| Test macro F1 | 88.46% | 90.04% | +1.58 percentage points |

The paired correctness counts on all 10,000 images were **8,569 both
correct**, **442 wrong → correct**, **285 correct → wrong**, and **704 both
wrong**. Altogether **891 images (8.91%) changed predicted class**; for
**164** of these, both versions were wrong but chose different wrong classes.
Thus the net accuracy gain is 442 - 285 = **157 additional correct images**,
while the two versions still disagree on many individual predictions. A paired
multinomial bootstrap of correctness transitions (20,000 draws, fixed seed)
gave a 95% percentile interval of **+1.03 to +2.11 percentage points** for
V2 - V1 accuracy; the exact paired McNemar p-value was **6.34 × 10^-9**.
This quantifies sampling uncertainty for these two fixed models on this test
set; it does not quantify training-run variability.

## Direction of the prediction changes

The website's new **Where predictions moved** panel groups uploaded images
by **V1 predicted class → V2 predicted class**. Each path reports how many
images were fixed, regressed, or changed between two wrong classes. Clicking
a path filters the image table to the actual examples. Selected frequent
paths on the full 10,000-image test set are:

| V1 → V2 | Images | Fixed | Regressed | Both wrong, different class |
| --- | ---: | ---: | ---: | ---: |
| dog → cat | 79 | 47 | 27 | 5 |
| cat → dog | 64 | 32 | 21 | 11 |
| truck → automobile | 32 | 20 | 11 | 1 |
| deer → horse | 30 | 19 | 7 | 4 |
| frog → bird | 28 | 15 | 7 | 6 |
| airplane → ship | 26 | 11 | 15 | 0 |

For example, official test image **20** is a horse: V1 predicted dog with
53.8% confidence and V2 predicted horse with 96.8% confidence. Image **74**
is an airplane: V1 predicted airplane with 54.0% confidence but V2 predicted
truck with 80.7% confidence. The update clearly helps some images and hurts
others. Bird recall rose from **79.0% to 83.2%**; cat recall rose from
**74.2% to 76.9%**. All ten classes had a positive observed recall change,
but the size varied from +0.1 to +4.2 percentage points.

The ready-to-upload website bundle uses the **first 1,000 official test
images**, with the same labels, CSV pixel features and ZIP images for both
models. In that display subset, **98** images changed predicted class:
**46** were fixed, **33** regressed, and **19** changed between wrong
classes. These 1,000-image figures are for the dashboard view; the main
experiment figures above use all 10,000 test images.

## Explanation changes

For an apples-to-apples exploratory comparison, the analysis script computes
Grad-CAM maps for the **same true-class logit** in both models on **all 1,000
dashboard images**. For the **98** changed predictions, cosine similarity of
normalized maps averaged **0.841** (median **0.865**; middle 50% **0.783 to
0.909**). For the **902** unchanged predictions, it averaged **0.932**
(median **0.950**; middle 50% **0.915 to 0.969**). This is a descriptive
association, not evidence that explanation change caused the prediction
change. Test index **128** changed from cat to frog for a dog image and had
similarity **0.518**; both versions were wrong. Test index **162** changed
from cat to the correct frog prediction and had similarity **0.712**. A
different pattern appears at index **551**: both versions still predicted cat
for a dog, but same-target map similarity was only **0.501**. A stable
prediction can therefore conceal a changed explanation. These values are
**not** a calibrated drift threshold or a claim that one model's explanation
is better. The website's map viewer targets each model's **predicted** class
and should be interpreted differently when the two predicted classes differ.

## Reproduce and inspect

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_wide_seed42 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --seed 42 --widths 40 80 160 --device mps

MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python scripts/compare_cifar10_versions.py \
  --baseline-run artifacts/cifar10_cnn_full_2026-10-29 \
  --candidate-run artifacts/cifar10_cnn_wide_seed42 \
  --output-dir artifacts/cifar10_directional_version_comparison
```

The ignored local output contains `directional_comparison.json` with the
10,000-image counts, class flows, paired interval and 1,000-map explanation
audit. To reproduce the website display, upload both `.joblib` files,
`samples.csv`, `manifest.json`, and `images.zip` from
`artifacts/cifar10_directional_version_comparison/upload_bundle/`. The
dashboard preflight confirmed that both models accept the bundle's 1,000
rows and 3,072 pixel features. The actual upload comparison completed with
Grad-CAM support; it computes displayed predicted-class maps for the first
100 rows. Generated model/data files stay local and are Git-ignored.
