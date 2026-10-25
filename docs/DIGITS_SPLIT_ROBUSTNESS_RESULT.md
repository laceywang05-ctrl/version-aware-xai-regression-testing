# Handwritten-digit split robustness

## Objective

This experiment tests whether the ordering of four closely related model
versions is stable across data splits. Each model was trained and evaluated on
10 deterministic stratified 70/30 splits of the scikit-learn handwritten-digit
dataset. Confidence intervals were estimated by 2,000 deterministic bootstrap
resamples of the 10 split-level results.

## Mean performance across 10 splits

| Version | Mean accuracy | 95% bootstrap CI | Mean macro F1 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| V1 | 96.52% | 96.22%–96.83% | 96.50% | 96.20%–96.80% |
| V2 | 97.30% | 96.85%–97.76% | 97.28% | 96.86%–97.75% |
| V3 | 97.69% | 97.39%–97.98% | 97.67% | 97.39%–97.97% |
| V4 | 95.72% | 95.41%–96.09% | 95.71% | 95.37%–96.07% |

V3 had the highest average accuracy and macro F1. Its accuracy ranged from
97.04% to 98.33% across individual splits, showing that the exact result still
depends on which images enter the held-out set.

## Paired split-level changes

The mean changes below are calculated as the right-hand version minus the
left-hand version on the same split.

| Comparison | Mean accuracy change | 95% bootstrap CI | Right wins / ties / left wins |
|---|---:|---:|---:|
| V1 → V2 | +0.78 pp | +0.35 to +1.20 pp | 9 / 1 / 0 |
| V1 → V3 | +1.17 pp | +0.81 to +1.57 pp | 10 / 0 / 0 |
| V1 → V4 | −0.80 pp | −1.20 to −0.37 pp | 1 / 1 / 8 |
| V2 → V3 | +0.39 pp | +0.15 to +0.67 pp | 7 / 2 / 1 |
| V2 → V4 | −1.57 pp | −2.11 to −1.02 pp | 0 / 1 / 9 |
| V3 → V4 | −1.96 pp | −2.41 to −1.50 pp | 0 / 0 / 10 |

All six bootstrap intervals exclude zero. Across these selected splits, adding
trees and relaxing the depth constraint improved mean accuracy, while the
shallower V4 consistently performed worse than V2 and V3.

## Relationship to the single-split experiment

The earlier seed-42 experiment found no conclusive paired correctness
difference among V1, V2 and V3 on one held-out set. The repeated-split analysis
answers a different question: it estimates the average performance change over
variation in the train-test partition. The repeated result suggests small but
consistent average improvements for V2 and V3 even though one individual split
did not provide strong McNemar evidence.

These findings are complementary rather than contradictory. McNemar evaluates
paired predictions within one fixed test set; the bootstrap intervals here
describe variability across data splits.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression digits-split-robustness
```

The machine-readable report is saved to
`artifacts/digits_split_robustness/summary.json`.

## Limitations

- Ten splits provide useful sensitivity evidence but are not an exhaustive
  characterization of all possible partitions.
- The bootstrap treats the selected split results as the empirical population;
  its interval depends on this predefined seed set.
- Model random states are held fixed, so the experiment isolates split
  variation rather than combining data-split and training randomness.
- Results apply to the small 8×8 digits dataset and should be repeated on MNIST
  and CIFAR-10 before making broader image-model claims.
