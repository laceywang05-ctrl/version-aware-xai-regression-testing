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
| `mnist_neural_demo` | grayscale neural classification | 500 | 784 | 10 | 4 | shallow CNN, 83.4% accuracy |
| `cifar10_cnn_full_2026-10-29/upload_bundle` | RGB CNN classification | 1,000 | 3,072 | 10 | 2 | validation-selected CNN pair |

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

For image bundles, append `--check-explanations`. This executes the real upload
explanation path for every model rather than inferring the method from its
filename. The MNIST neural bundle passed with Integrated Gradients for both MLP
models and Grad-CAM for both CNN models. The CIFAR-10 CNN pair passed with
Grad-CAM for both versions.

The same flag also validates tabular classification explanations. In the Wine
bundle, the dummy and logistic-regression models passed with LIME, while the
decision tree and random forest passed with both Tree SHAP and LIME. This
selection is performed by the service after upload; users do not train separate
"SHAP models" or "LIME models".

## Iris model-family upload

Generate the five-model bundle with:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression iris-demo
```

The bundle adds logistic regression, K-nearest neighbours, RBF SVM, Gaussian
naive Bayes and Extra Trees on the same 45-row, four-feature, three-class test
split. This checks model-agnostic LIME routing alongside Tree SHAP plus LIME for
the Extra Trees model.

For regression bundles, the same flag generates model-agnostic permutation
importance for every uploaded regressor and confirms that numeric feature
importance can be displayed before a live demonstration.

## One-command matrix

Use `demo-matrix` with repeated `--bundle-dir` options to validate several
representative bundles in one run. The command reports the task, model count,
sample count and explanation methods for each bundle, continues after an
individual failure, and succeeds only when the complete matrix is ready.

The `--standard` preset checks Breast Cancer, Wine, Iris, the four-model MNIST
neural bundle and the four-model diabetes regression bundle. It therefore covers
tabular classification, neural image classification and numeric regression,
including Tree SHAP, LIME, Integrated Gradients, Grad-CAM and permutation
importance.

Add `--output-dir artifacts/standard_demo_validation` to save both
`demo_validation_matrix.json` for automation and
`demo_validation_matrix.md` as a readable validation record. The Markdown
report also lists every model file, estimator type, serialized file size,
compatibility status and explanation method. It records the UTC generation
time, Python runtime, operating-system platform and relevant library versions
so the validation can be reproduced in another environment.
