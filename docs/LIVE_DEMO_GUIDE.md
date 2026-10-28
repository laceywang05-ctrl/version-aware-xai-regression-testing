# Live dashboard demonstration guide

## 1. Start and preflight

From the repository root:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression demo-check \
  --bundle-dir artifacts/mnist_model_demo

MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression serve
```

Open <http://127.0.0.1:8000>, select **Compare uploaded models**, and use
only bundles whose preflight ends with `READY FOR DEMO`.

## 2. Recommended main demonstration: close MNIST tree models

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

## 3. Neural explanation demonstration

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

## 4. CIFAR-10 demonstration

Directory: `artifacts/cifar10_model_demo/` (about 26 MB)

Upload the same file roles as the MNIST tree bundle. It contains 1,000 labelled
32×32 RGB images, 3,072 ordered pixel-channel features and 10 object classes.
The four expected accuracies are 8.9%, 21.2%, 33.4% and 32.1%. This validates
the RGB workflow and image display, but the flattened-pixel models are
deliberately lightweight and are not competitive CIFAR-10 architectures.

## 5. Regression demonstration

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

## 6. Fast fallback bundle

If time is short, use `artifacts/demo_image_upload/` (about 3.5 MB). It contains
two models, 60 labelled 8×8 digit images and 64 features. It exercises the same
classification upload path with much smaller files.

## 7. Demonstration boundaries

- Upload only trusted `.joblib` files because deserialization can execute code.
- A review flag requests human inspection; it is not an automatic deployment
  rejection or proof of model harm.
- Statistical conclusions describe the uploaded labelled sample.
- Explanation methods have different semantics and thresholds must not be
  transferred across datasets or methods without calibration.
- Uploaded files are processed locally in memory and are not retained by the
  server.
