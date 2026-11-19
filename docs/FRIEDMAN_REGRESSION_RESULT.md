# Friedman Nonlinear Regression Dashboard Validation

## Purpose

This validation checks whether the upload dashboard can compare five distinct
regression model families on the same nonlinear numeric problem. It exercises
the complete regression overview path, including ranking, pairwise error
changes, sample-level numeric predictions and model-agnostic explanations.

## Data and models

- Dataset: Friedman #1 synthetic nonlinear regression
- Generation: 1,000 rows, 10 numeric features, noise `1.0`, random state `42`
- Split: fixed 70/30 train-test split (`random_state=42`)
- Training rows: 700
- Uploaded test rows: 300
- Models: linear regression, scaled KNN, scaled RBF SVR, random forest and
  gradient boosting

## Observed performance

| Rank | Model | MAE | RMSE | R² |
| ---: | --- | ---: | ---: | ---: |
| 1 | RBF SVR | **1.100** | **1.402** | **0.912** |
| 2 | Gradient boosting | 1.305 | 1.609 | 0.884 |
| 3 | Random forest | 1.572 | 1.948 | 0.829 |
| 4 | Linear regression | 1.892 | 2.339 | 0.754 |
| 5 | KNN regressor | 2.187 | 2.693 | 0.674 |

The dashboard produced five ranked model summaries and all ten possible
pairwise error-change comparisons. For example, the SVR improved the absolute
error of 214 of 300 rows relative to linear regression and worsened 86 rows.
Gradient boosting improved 198 rows relative to linear regression and worsened
102. These paired counts make clear that a better aggregate MAE does not mean
that every individual prediction improved.

## Explanation coverage

All five models passed the regression upload contract and produced permutation
feature importance. This is important because linear regression, KNN, SVR and
tree ensembles do not share one native explanation interface. The dashboard
therefore uses the same model-agnostic perturbation procedure and labelled
evaluation rows for a comparable feature-level view.

## Interpretation

The nonlinear data-generating process favours the RBF SVR and ensemble models
over the unregularized linear model in this fixed split. The result is a
demonstration of dashboard compatibility rather than evidence that one family
is universally superior. The main validation outcome is that five
heterogeneous regressors can be uploaded together and compared through the
same ranking, pairwise, sample-level and explanation workflow.
