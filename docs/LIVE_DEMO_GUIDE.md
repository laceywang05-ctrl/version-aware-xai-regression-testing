# Live dashboard demonstration guide

## 1. Start and preflight

From the repository root:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression demo-check \
  --bundle-dir artifacts/cifar10_directional_version_comparison_seed43/upload_bundle

MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression serve
```

Open <http://127.0.0.1:8000>, select **Compare uploaded models**, and use
only bundles whose preflight ends with `READY FOR DEMO`.

## 2. Recommended main demonstration: real CIFAR-10 model update

Directory: `artifacts/cifar10_directional_version_comparison_seed43/upload_bundle/`

| Page field | File or setting |
| --- | --- |
| Prediction task | Classification |
| Model file 1 (V1) | `cifar10_v1_original_widths.joblib` |
| Model file 2 (V2) | `cifar10_v2_wider_widths.joblib` |
| Labelled dataset | `samples.csv` |
| Manifest | `manifest.json` |
| Sample images | `images.zip` |
| Additional models | Leave empty |

The models are six-convolution-layer CIFAR-10 classifiers. V1 has stage widths
32/64/128 and 288,746 parameters; V2 has 40/80/160 and 450,210 parameters.
This is an actual architecture change. The uploaded files contain the first
1,000 official test images, 3,072 ordered RGB pixel features per image and
ten classes. Confirm that the model files are trusted, select V1 as baseline
and V2 as candidate, then click **Run local comparison**.

In this **1,000-image website subset**, V1 accuracy is **90.1%** and V2
accuracy is **90.4%**: a small net improvement of just three images. Yet
**77 images changed predicted class**: 35 changed from wrong to correct, 32
from correct to wrong, and 10 changed between two wrong classes. This is the
clearest demonstration of why the platform inspects prediction direction in
addition to aggregate performance. In **all 10,000 official test images**,
the same pair scored 88.83% and 89.94%, with 880 changed class predictions;
do not confuse these full-test figures with the uploaded subset.

Suggested live sequence:

1. Show the V1/V2 filenames and the two close accuracy cards.
2. In **Where predictions moved**, explain that each arrow is a V1 predicted
   class → V2 predicted class. Green, orange and grey portions count images
   fixed, regressed and still wrong. Click a path to filter the image table.
3. Use the transition filter to show a concrete improvement: `image-00032`
   is a deer that V1 called bird and V2 called deer. Then show a regression:
   `image-00070` is a bird that V1 got right and V2 called airplane.
4. Open **Compare maps** for a row among the first 100 images. The displayed
   Grad-CAM maps explain each model's predicted class and are exploratory;
   they are not a calibrated pass/fail decision.
5. Show the JSON, readable summary and sample CSV export buttons. The files
   are created locally and contain the selected model identities and results.

This seed 43 pair is one repeat of the same width change. The seed 42 pair
also improved on the full test set, from 88.54% to 90.11%, with 891 changed
class predictions. See
[`CIFAR10_WIDTH_REPEAT_RESULT.md`](CIFAR10_WIDTH_REPEAT_RESULT.md) for both
matched experiments and their limitations.

## 3. Close MNIST tree-model demonstration

Directory: `artifacts/mnist_model_demo/` (about 27 MB)

| Page field | File |
| --- | --- |
| Prediction task | Classification |
| Model file 1 | `model_3_random_forest.joblib` |
| Model file 2 | `model_4_extra_trees.joblib` |
| Labelled dataset | `samples.csv` |
| Manifest | `manifest.json` |
| Sample images | `images.zip` |
| Additional models | `model_1_dummy.joblib`, `model_2_tree.joblib` |

The bundle contains 1,000 labelled MNIST test images, 784 pixel features and
10 classes (digits 0–9). The four expected accuracies are 9.9%, 60.7%, 84.7%
and 84.8%. Random Forest and Extra Trees are the strongest live pair because
their aggregate accuracy differs by only one image while the dashboard still
shows their individual disagreements and explanation-map changes.

After upload, point out:

1. the four-model ranking and pairwise prediction agreement;
2. the selected baseline and candidate names;
3. the paired McNemar result and transition counts;
4. class-level accuracy, confusion matrices and error paths;
5. individual images that improved or regressed;
6. Tree SHAP method labels, calibrated threshold source and heatmap comparison;
7. the JSON, readable summary and sample CSV downloads.

## 4. Neural explanation demonstration

Directory: `artifacts/mnist_neural_demo/` (about 2.0 MB)

The bundle contains 500 labelled MNIST test images and 784 pixel features.

| Model | Architecture summary | Parameters | Method | Accuracy |
| --- | --- | ---: | --- | ---: |
| `model_1_mlp_small.joblib` | 784→64→10 MLP | 50,890 | Integrated Gradients | 82.4% |
| `model_2_mlp_large.joblib` | 784→128→64→10 MLP | 109,386 | Integrated Gradients | 80.0% |
| `model_3_cnn_shallow.joblib` | 1 conv layer + linear head | 15,770 | Grad-CAM | 83.4% |
| `model_4_cnn_deep.joblib` | 2 conv layers + pooled head | 3,818 | Grad-CAM | 68.6% |

Use the two MLP files to demonstrate a method-compatible Integrated Gradients
comparison. The active MNIST threshold is `0.896604`, estimated from benign
retraining, with a 7.0% held-out review rate.

Use the two CNN files to demonstrate Grad-CAM. The maps and similarities are
shown, but the page marks them exploratory: expanded calibration produced 100%
informative coverage but a fifth percentile of `0.009905`, too close to zero
for a useful automatic drift rule.

## 5. Lightweight CIFAR-10 tree-model demonstration

Directory: `artifacts/cifar10_model_demo/` (about 26 MB)

Upload the same file roles as the MNIST tree bundle. It contains 1,000 labelled
32×32 RGB images, 3,072 ordered pixel-channel features and 10 object classes.
The four expected accuracies are 8.9%, 21.2%, 33.4% and 32.1%. This validates
the RGB workflow and image display, but the flattened-pixel models are
deliberately lightweight and are not competitive CIFAR-10 architectures.

## 6. Regression demonstration

Directory: `artifacts/demo_regression_upload/` (about 2.2 MB)

Select **Regression** and upload `linear_regression.joblib` and
`gradient_boosting.joblib` as the main pair, plus `decision_tree.joblib` and
`random_forest.joblib` as additional models. Use `samples.csv` and
`manifest.json`; no image ZIP is needed.

The bundle contains 133 held-out scikit-learn diabetes samples with 10 numeric
features. Expected MAE values are 41.9194 (linear), 47.0346 (decision tree),
43.2944 (random forest), and 41.7280 (gradient boosting). Show the MAE/RMSE/R²
ranking, paired per-sample error changes, parity plot, permutation importance,
individual patient rows and regression exports.

## 7. Fast fallback bundle

If time is short, use `artifacts/demo_image_upload/` (about 3.5 MB). It contains
two models, 60 labelled 8×8 digit images and 64 features. It exercises the same
classification upload path with much smaller files.

## 8. Demonstration boundaries

- Upload only trusted `.joblib` files because deserialization can execute code.
- A review flag requests human inspection; it is not an automatic deployment
  rejection or proof of model harm.
- Statistical conclusions describe the uploaded labelled sample.
- Explanation methods have different semantics and thresholds must not be
  transferred across datasets or methods without calibration.
- Uploaded files are processed locally in memory and are not retained by the
  server.
