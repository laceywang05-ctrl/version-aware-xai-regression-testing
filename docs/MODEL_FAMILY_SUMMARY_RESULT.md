# Cross-Model-Family SHAP Summary

## Question

Does the held-out explanation-regression policy respond consistently when the
same structural changes are applied to Random Forest and Gradient Boosting
models on both completed datasets?

## Evidence matrix

The summary combines 12 paired comparisons: three structural-change scenarios
across two datasets and two model families. Every comparison uses 30 held-out
seeds and an exact paired McNemar test with Bonferroni adjustment.

| Scenario | Significant cells | Total cells | Interpretation |
| --- | ---: | ---: | --- |
| Shallow depth | 2 | 4 | Detected for both Gradient Boosting tasks, but not for either Random Forest task |
| Larger leaves | 3 | 4 | Detected except on Breast Cancer Random Forest |
| Combined constraints | 4 | 4 | Detected on both datasets and both model families |

The combined depth-and-leaf change is the first controlled scenario that
separates from benign retraining in every dataset-model cell. Its prediction
agreement remains between 98.0% and 100%, so the evidence is not explained only
by changed predicted labels.

## Interpretation

This supports a limited cross-model-family sensitivity claim: the method can
identify at least one consistent structural explanation change in Random Forest
and Gradient Boosting models. It does not imply that every parameter change is
detectable. The shallow-depth result, for example, depends strongly on model
family, and the larger-leaf result still depends on the dataset/model pairing.

The correct thesis claim is therefore:

> Under the tested tabular binary-classification settings, the calibrated Tree
> SHAP policy detected a combined structural change across two datasets and two
> tree-based model families while prediction agreement remained high.

It should not be described as universal generalisation to all algorithms,
datasets or kinds of model update.

## Reproduction

After generating the four benchmark summaries, run:

```bash
PYTHONPATH=src python -m xai_regression model-family-summary
```

The machine-readable JSON, comparison CSV and Markdown report are written to
`artifacts/experiment_30_model_family_summary/reports/`.
