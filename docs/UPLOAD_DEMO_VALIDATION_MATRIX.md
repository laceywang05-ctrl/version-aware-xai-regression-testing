# Upload demo validation matrix

All bundles below passed the dashboard `demo-check` preflight on 19 November
2026. The check loads every trusted model, validates the manifest and labelled
data, and runs the same model-summary code used by the local upload service.

| Bundle | Task | Test rows | Features | Classes | Models | Best result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `wine_multiclass_model_demo` | tabular classification | 54 | 13 | 3 | 4 | random forest, 100.0% accuracy |
| `mnist_model_demo` | grayscale image classification | 1,000 | 784 | 10 | 4 | extra trees, 84.8% accuracy |
| `cifar10_model_demo` | RGB image classification | 1,000 | 3,072 | 10 | 4 | random forest, 33.4% accuracy |
| `demo_regression_upload` | tabular regression | 133 | 10 | n/a | 4 | gradient boosting, MAE 41.728 |

The MNIST and CIFAR-10 bundles also contain 1,000 display images. Their image
models were trained on 1,000 official training samples and evaluated on 1,000
official test samples. They are deliberately small interaction demos, not
state-of-the-art image classifiers.

## Wine multiclass upload

Generate the bundle:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression wine-demo
```

Upload from `artifacts/wine_multiclass_model_demo/`:

1. Choose **Classification**.
2. Select `model_1_dummy.joblib` and `model_2_tree.joblib` as the first pair.
3. Select `model_3_random_forest.joblib` and
   `model_4_logistic_regression.joblib` under **Additional models**.
4. Upload `samples.csv` and `manifest.json`; no image ZIP is required.

Expected held-out results are 38.9%, 96.3%, 100.0%, and 98.1% accuracy in the
model-file order above. The dashboard should expose three class rows and
multi-model tabular explanation summaries.

## Re-run preflight

Use the same command for any classification or regression bundle:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression demo-check \
  --bundle-dir artifacts/wine_multiclass_model_demo
```

A usable bundle ends with `READY FOR DEMO` and lists every model as `READY`.
