# Multi-model upload validation

This check covers two practical risks in the user upload workflow: a repeated model file being counted as independent evidence, and feature rankings that change when LIME is run on only a few rows. It also exercises the same upload functions on a three-class dataset.

## Reproduce

From the repository root, with the project environment installed:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
  .venv/bin/python scripts/validate_multimodel_uploads.py
```

The script reads only the local, trusted Breast Cancer demo models. It writes detailed run data to the ignored local file `artifacts/multimodel_upload_validation.json`. The Wine models are trained deterministically in memory; no Wine model artifact is needed.

## Breast Cancer LIME sensitivity

The held-out Breast Cancer CSV has 171 rows and 30 numeric features. The check uses the tree, random forest, and logistic regression models. For each of three fixed row orders (seeds 17, 42, 73), it computes LIME global rankings from the first 6, 20, and 50 rows. Each local explanation uses 500 perturbations. The reference is the 50-row run with seed 17. The table counts shared top-five features, so 5 means the same set of five features was selected.

| Rows used | Seed 17 | Seed 42 | Seed 73 |
| --- | ---: | ---: | ---: |
| 6 | 4/5 | 4/5 | 3/5 |
| 20 | 4/5 | 4/5 | 5/5 |
| 50 | 5/5 | 4/5 | 5/5 |

The six-row view can change materially with sample order. The website therefore uses 20 rows for the optional multi-model LIME overview and labels it **exploratory**. This is an empirical sensitivity check on one held-out split, not a confidence interval or a claim that the ranking will be stable on new datasets. The selected pair's SHAP/LIME analysis remains a separate calculation.

Identical uploaded `.joblib` bytes are now counted once in the multi-model feature vote. They remain visible as separate uploaded models in the performance ranking and pairwise comparisons. The UI reports how many duplicate files were omitted from the vote. This detects byte-identical files; it cannot detect semantically identical models serialized differently.

## Three-class Wine upload

The script uses scikit-learn Wine: 178 samples, three classes, and 13 numeric features. A stratified split with seed 42 gives 124 training and 54 held-out rows. It trains a most-frequent dummy classifier, a depth-four tree, a 40-tree random forest, and a scaled logistic regression model. All four pass the upload preflight on the same 54-row labelled CSV. The overview has three class rows and a four-model LIME consensus. The selected tree-to-forest comparison also completes.

| Model | Held-out accuracy |
| --- | ---: |
| Dummy | 38.9% |
| Decision tree | 96.3% |
| Random forest | 100.0% |
| Scaled logistic regression | 98.1% |

These are workflow checks on a small, fixed split, not a model benchmark. For real user uploads, feature names, order, preprocessing, and label mapping must match what each trained model expects. Preflight checks prediction compatibility before the full comparison.
