# Handwritten-digit explanation split robustness

## Objective

The fixed-split explanation experiment found no hidden explanation drift. This
follow-up checks whether that result depends on the seed-42 data partition. V1,
V2 and V3 are retrained on five stratified train-test splits. On every split,
their Tree SHAP predicted-class maps are compared on the same 80 held-out
images.

Hidden drift means that two versions predict the same class but their
explanation cosine similarity falls below the 0.80 inspection threshold.

## Aggregate results

| Comparison | Mean cosine across splits | Lowest split mean | Comparisons | Below 0.80 | Hidden drift |
|---|---:|---:|---:|---:|---:|
| V1 vs V2 | 0.906 | 0.857 | 400 | 12 | 1 |
| V1 vs V3 | 0.915 | 0.882 | 400 | 11 | 0 |
| V2 vs V3 | 0.936 | 0.901 | 400 | 8 | 0 |

Across 1,200 pairwise sample comparisons, one case met the hidden-drift rule.
It occurred for V1 versus V2 on split seed 11. The other 30 below-threshold
comparisons coincided with a changed predicted class, so they are not counted
as hidden drift.

## Interpretation

The explanations are broadly stable but not perfectly invariant to the data
split. All three pairs retain high mean similarity, and only one of the 1,200
comparisons is a hidden-drift flag. This supports a more careful conclusion
than the fixed-split experiment: close versions usually explain matching
predictions similarly, while rare split-sensitive cases still justify
sample-level inspection.

The 0.80 value remains a pragmatic review threshold rather than a calibrated
failure boundary. The five splits are sensitivity checks on one dataset, not
independent population samples.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression \
  digits-explanation-split-robustness
```

The aggregate report is saved to
`artifacts/digits_explanation_split_robustness/summary.json`. Per-split reports
are saved in the corresponding `split_<seed>` folders.

## Limitations

- Only five split seeds and the first 80 test images per split are evaluated.
- The versions are related Random Forest models, not different model families.
- Cosine similarity measures attribution direction rather than scale.
- The same source dataset is reused across splits, so results are correlated.
- Expert-labelled explanation changes are still needed to calibrate the review
  threshold.
