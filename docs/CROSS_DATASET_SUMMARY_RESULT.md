# Cross-Dataset SHAP and LIME Evidence Summary

## Material Passport

- Material ID: `cross-dataset-xai-summary-v1`
- Type: derived experiment summary
- Verification status: `VERIFIED`
- Source scope: four local, previously generated experiment summaries
- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine
- Explanation methods: Tree SHAP and LIME
- External data upload: none
- Generated artifact: `artifacts/experiment_14_cross_dataset_summary/`

## Purpose

This summary closes the dataset-by-method matrix for the current prototype. It
places SHAP and LIME evidence from both datasets in one machine-readable report
and one dashboard view without treating their policy statistics as
interchangeable.

## Evidence matrix

| Dataset | Method | Evidence design | Observations | Prediction agreement | Cosine similarity | Headline result | Status |
|---|---|---|---:|---:|---:|---:|---|
| Breast Cancer Wisconsin | SHAP | Held-out repeated benchmark | 120 candidate runs | 0.9967 | 0.9746 | Strongest candidate review rate: 50.0% | SEPARATED |
| Breast Cancer Wisconsin | LIME | Noise-calibrated matched comparison | 200 matched explanations | 0.9500 | 0.8072 | Any-metric breach rate: 12.5% | REVIEW |
| Wine (binary) | SHAP | Held-out repeated benchmark | 120 candidate runs | 0.9978 | 0.9820 | Strongest candidate review rate: 76.7% | SEPARATED |
| Wine (binary) | LIME | Noise-calibrated matched comparison | 200 matched explanations | 1.0000 | 0.9684 | Any-metric breach rate: 22.5% | REVIEW |

For SHAP, `SEPARATED` means that at least one configured model-change scenario
was separated from the seed-only control after Bonferroni correction. It is not
a pass/fail judgement about the model. For LIME, `REVIEW` is an advisory signal
that at least one explanation metric exceeded its calibrated run-level limit.

## Descriptive interpretation

Prediction agreement remained between 95.0% and 100% in the four displayed
cells. Explanation behaviour nevertheless produced method- and
dataset-dependent signals. This is consistent with the project's motivation:
prediction stability alone does not fully describe explanation stability.

The SHAP and LIME headline rates answer different questions. SHAP's strongest
candidate review rate comes from repeated held-out model-version scenarios.
LIME's breach rate comes from matched, noise-calibrated explanation comparisons.
The values are therefore displayed together to document coverage, not to rank
the two explanation methods.

## Reproducibility check

The summary was regenerated from the four existing local JSON reports using:

```bash
PYTHONPATH=src python -m xai_regression portfolio
```

All 51 automated tests passed after adding the aggregator, server route and
dashboard view. No model retraining was required because this stage only
normalizes completed experiment artifacts.

## Output files

- `artifacts/experiment_14_cross_dataset_summary/reports/summary.json`
- `artifacts/experiment_14_cross_dataset_summary/reports/cross_dataset_comparison.csv`
- `artifacts/experiment_14_cross_dataset_summary/reports/report.md`

## Limitations

- The SHAP and LIME evidence designs are intentionally different and cannot be
  collapsed into a single universal regression score.
- The two datasets are both tabular binary-classification tasks and do not
  establish generality to text, images or multiclass production systems.
- `REVIEW` and `SEPARATED` remain investigation labels, not evidence that a
  candidate model is defective or less accurate.
