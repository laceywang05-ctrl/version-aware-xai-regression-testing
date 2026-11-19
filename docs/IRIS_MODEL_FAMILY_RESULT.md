# Iris Multi-Model Dashboard Validation

## Purpose

This validation checks whether the upload dashboard can compare five distinct
classifier families on the same labelled multiclass dataset. It exercises the
full overview path rather than only checking that serialized models can load.

## Data and models

- Dataset: scikit-learn Iris
- Split: fixed stratified 70/30 split (`random_state=42`)
- Training rows: 105
- Uploaded test rows: 45
- Numeric features: 4
- Classes: 3 (`setosa`, `versicolor`, `virginica`)
- Models: scaled logistic regression, scaled KNN, calibrated RBF SVM,
  Gaussian naive Bayes and Extra Trees

## Observed performance

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic regression | 91.1% | 91.1% |
| KNN | 91.1% | 91.0% |
| Calibrated RBF SVM | **93.3%** | **93.3%** |
| Gaussian naive Bayes | 91.1% | 91.1% |
| Extra Trees | 91.1% | 91.1% |

The dashboard selected the calibrated RBF SVM as the best model. It produced
five ranked model cards, all ten pairwise agreement rows, three per-class rows
and all 45 sample-level comparison results. Pairwise prediction agreement ranged
from 93.3% to 100.0%.

## Explanation coverage

All five models supported LIME. Extra Trees additionally supported Tree SHAP.
The multi-model LIME consensus used 20 deterministic rows per model and agreed
that petal width and petal length were the two highest-ranked features for all
five models. Sepal width and sepal length followed with slightly different
average ranks.

## Interpretation

This is a compatibility and interaction demonstration, not evidence that one
classifier family is universally superior. The test set contains only 45 rows,
so the small observed accuracy differences should be treated as descriptive.
The important result is that heterogeneous probabilistic classifiers can enter
the same upload, ranking, pairwise, class-level, sample-level and explanation
workflow without model-specific user preparation.
