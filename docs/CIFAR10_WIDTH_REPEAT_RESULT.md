# CIFAR-10 CNN width-change repeat across two matched seeds

## Purpose and protocol

The [first directional comparison](CIFAR10_DIRECTIONAL_VERSION_COMPARISON.md)
found that widening a six-convolution-layer CNN changed individual predictions
and improved overall test accuracy. This repeat asks whether the **same
architecture change** produces a similar direction under another training
seed. It does not present a seed change as the V1-to-V2 update: within each
pair, V1 has 32/64/128 convolution-stage channels and 288,746 parameters;
V2 has 40/80/160 channels and 450,210 parameters. Seed 42 and seed 43 are
two separate matched pairs.

Both pairs use the same optimizer, augmentation specification, 45,000/5,000
stratified training/validation split sizes and untouched official 10,000-image
test set. Within each pair, the training/validation indices, nominal seed,
test order and labels were checked for exact alignment before comparison.
Every final checkpoint was selected by validation accuracy before test
evaluation. Seed 43 V1 and V2 ran uninterrupted. Seed 42 V1 was paused and
resumed after epoch 50 without restoring augmentation RNG state. Also,
architecture-dependent initialization can alter random-number streams even
with the same nominal seed. Consequently, this is a two-run robustness check,
not an isolated estimate of the width effect.

## Results on all 10,000 official test images

| Seed | V1 selected epoch | V2 selected epoch | V1 accuracy | V2 accuracy | Gain | V1 loss | V2 loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 74 | 60 | 88.54% | 90.11% | **+1.57 pp** | 0.3420 | 0.3117 |
| 43 | 69 | 69 | 88.83% | 89.94% | **+1.11 pp** | 0.3355 | 0.3124 |

Macro F1 rose from **88.46% to 90.04%** for seed 42 and from **88.77% to
89.89%** for seed 43. Best validation accuracy was 89.50% → 90.56% for seed
42 and 89.34% → 89.98% for seed 43.

| Seed | Changed predicted class | Wrong → correct | Correct → wrong | Both wrong, different class | Paired 95% CI for accuracy gain | Exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 891 (8.91%) | 442 | 285 | 164 | +1.03 to +2.11 pp | 6.34 × 10^-9 |
| 43 | 880 (8.80%) | 417 | 306 | 157 | +0.58 to +1.64 pp | 4.16 × 10^-5 |

The intervals are paired multinomial-bootstrap percentile intervals (20,000
draws) for the **fixed model pairs** on the same test images. The tests also
compare paired correctness for each fixed pair. They do not measure
uncertainty across training seeds or prove that widening will help every run.
Because both pairs share the same official test set, their p-values should
not be treated as independent replications or combined naively.

## Which predictions moved?

The two runs changed approximately 9% of class predictions, while the net
accuracy gain was around 1 percentage point. The changes are therefore more
substantial at the sample level than the aggregate score alone suggests.
The `dog → cat` path occurred **79** times under seed 42 (47 fixes, 27
regressions, 5 still wrong) and **77** times under seed 43 (31 fixes, 36
regressions, 10 still wrong). The same class-transition direction can help in
one run but hurt in another; the actual label is essential. In seed 43,
`deer → bird` occurred 30 times (19 fixes, 7 regressions, 4 still wrong).

All ten classes had a positive observed recall change in each matched pair,
but some changes were tiny. Bird recall rose by **4.2 pp** in seed 42 and
**3.3 pp** in seed 43. These class-specific values are descriptive; no
multiple-comparison or across-training-run claim is made.

The dashboard upload bundle displays the first **1,000** official test
images. Its **Where predictions moved** panel groups images by V1 prediction
→ V2 prediction, breaks each path into fixed/regressed/still-wrong counts,
and filters the image table when a path is selected. In that subset, seed 42
had **98** changed predictions (46 fixes, 33 regressions, 19 still wrong)
and seed 43 had **77** (35 fixes, 32 regressions, 10 still wrong). The main
tables above use all **10,000** images.

## Explanation audit on the 1,000 dashboard images

For each image, Grad-CAM was computed for the **same true-class logit** in
both models and the normalized maps were compared by cosine similarity.
This keeps the explained target fixed even when V1 and V2 predict different
classes. Results are exploratory:

| Seed | Prediction changed: count, mean similarity, median | Prediction unchanged: count, mean similarity, median |
| ---: | --- | --- |
| 42 | 98; 0.841; 0.865 | 902; 0.932; 0.950 |
| 43 | 77; 0.835; 0.880 | 923; 0.927; 0.946 |

Lower similarity among changed predictions appeared in both runs, but this
does not establish causation or make one model's explanation better. Some
unchanged predictions also had markedly changed maps. No automatic
explanation-drift acceptance threshold was calibrated from this audit. The
website's interactive map viewer uses each model's **predicted class** and
currently renders maps for the first 100 uploaded rows; those maps answer a
different question from this same-target offline audit.

## Reproduction and local artifacts

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_wide_seed43 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256 \
  --seed 43 --widths 40 80 160 --device mps

MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python scripts/compare_cifar10_versions.py \
  --baseline-run artifacts/cifar10_cnn_full_seed43 \
  --candidate-run artifacts/cifar10_cnn_wide_seed43 \
  --output-dir artifacts/cifar10_directional_version_comparison_seed43
```

The ignored local comparison folder contains `directional_comparison.json`
and `upload_bundle/` with two `.joblib` models, `samples.csv`, `images.zip`
and `manifest.json`. Preflight confirmed that both models accept all 1,000
rows and 3,072 pixel features. The actual upload comparison completed and
produced predictions, transitions and Grad-CAM explanations. These generated
model and dataset files are local artifacts, not files committed to Git.
