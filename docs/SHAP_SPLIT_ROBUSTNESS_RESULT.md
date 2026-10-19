# Cross-Model Tree SHAP Split Robustness

## Purpose

The cross-model-family summary found that the combined depth-and-leaf change
separated from benign retraining on the original split (`split_seed=42`) for
both datasets and both model families. This experiment retests that preselected
scenario on two additional data splits, 17 and 73.

Only the previously selected combined-change scenario is tested against its
seed-only control. This avoids choosing a new best-performing scenario after
seeing the additional-split results.

## Design

- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine.
- Models: Random Forest and Gradient Boosting.
- Additional split seeds: 17 and 73.
- Calibration: 30 benign model seeds for each cell.
- Evaluation: 30 disjoint held-out model seeds for the control and candidate.
- Candidate: combined maximum-depth and minimum-leaf-size constraints.
- Explainer: Tree SHAP.
- Test: exact paired McNemar test of control versus candidate review decisions.

This creates eight new dataset-model-split comparisons.

## Results

| Dataset | Model | Split | Control review | Candidate review | Prediction agreement | p-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer | Random Forest | 17 | 13.3% | 66.7% | 99.3% | `3.05e-5` |
| Breast Cancer | Random Forest | 73 | 13.3% | 46.7% | 98.9% | `0.00195` |
| Wine | Random Forest | 17 | 33.3% | 90.0% | 98.0% | `1.53e-5` |
| Wine | Random Forest | 73 | 13.3% | 53.3% | 100.0% | `0.00183` |
| Breast Cancer | Gradient Boosting | 17 | 20.0% | 100.0% | 97.0% | `1.19e-7` |
| Breast Cancer | Gradient Boosting | 73 | 10.0% | 100.0% | 98.0% | `1.49e-8` |
| Wine | Gradient Boosting | 17 | 0.0% | 100.0% | 100.0% | `1.86e-9` |
| Wine | Gradient Boosting | 73 | 13.3% | 100.0% | 100.0% | `2.98e-8` |

All eight comparisons separated the combined candidate from the seed-only
control at `p < 0.05`. The overall status is therefore `ROBUST` for the tested
additional splits.

## Interpretation

The cross-model result is not an artefact of the original split 42. The same
combined structural change is detected on splits 17 and 73 in every tested
dataset-model cell, while prediction agreement remains between 97% and 100%.

This strengthens the thesis evidence but still does not establish universal
generalisability. The experiment covers two small tabular binary datasets, two
tree-based model families and three total split seeds. Control review rates also
range from 0% to 33.3%, so a review flag remains a triage signal rather than an
automatic failure decision.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression shap-split-robustness
```

The full reports and per-run artifacts are written under
`artifacts/experiment_31_shap_split_robustness/`.
