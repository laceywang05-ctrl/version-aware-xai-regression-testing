# Handwritten-digit explanation-threshold sensitivity

## Objective

The explanation experiments use cosine similarity below 0.80 as a pragmatic
inspection rule. This analysis checks how strongly the hidden-drift conclusion
and review workload depend on that choice.

The same V1–V3 comparisons are repeated across five stratified data splits,
with 80 test images per split. Only cases where the two models predict the same
class are eligible for hidden-drift review.

## Results

| Threshold | Hidden-drift flags | Flag rate |
|---:|---:|---:|
| 0.70 | 0 | 0.00% |
| 0.75 | 0 | 0.00% |
| 0.80 | 1 | 0.09% |
| 0.85 | 11 | 0.94% |
| 0.90 | 82 | 7.01% |

There are 1,170 same-prediction comparisons. Raising the threshold from 0.80
to 0.90 increases the number of reviewed cases from 1 to 82. The result is
therefore highly sensitive to strict thresholds near the upper end of the
observed similarity distribution.

## Interpretation

The 0.80 threshold produces a small, focused review queue for this controlled
experiment. A 0.90 threshold is much more conservative and would ask a reviewer
to inspect about 7% of matching predictions. Neither setting can be called
optimal without expert labels indicating which explanation changes are
acceptable or harmful.

This analysis should be presented as a workload trade-off, not an accuracy
comparison. Lower thresholds reduce review volume but may miss meaningful
drift; higher thresholds increase sensitivity but may create many unnecessary
reviews.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression \
  digits-explanation-threshold-sensitivity
```

The machine-readable report is saved to
`artifacts/digits_explanation_threshold_sensitivity/summary.json`.

## Limitations

- There is no expert-labelled ground truth for harmful explanation drift.
- Thresholds are evaluated on related Random Forest versions and one dataset.
- The five data splits reuse observations and are not independent datasets.
- Cosine similarity ignores absolute attribution magnitude.
