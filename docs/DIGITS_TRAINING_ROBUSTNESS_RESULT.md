# Handwritten-digit training-randomness robustness

## Objective

This experiment isolates random-forest training randomness. The train-test split
is fixed at seed 42, while each of the four model configurations is trained
with 10 different random seeds. This complements the split-robustness experiment,
which varied the data partition while holding model seeds fixed.

## Mean performance across 10 training seeds

| Version | Mean accuracy | 95% bootstrap CI | Mean macro F1 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| V1 | 95.69% | 95.43%–95.98% | 95.67% | 95.41%–95.96% |
| V2 | 96.74% | 96.57%–96.91% | 96.72% | 96.57%–96.87% |
| V3 | 96.96% | 96.76%–97.17% | 96.94% | 96.73%–97.14% |
| V4 | 94.37% | 94.06%–94.67% | 94.36% | 94.04%–94.67% |

V3 had the highest mean performance, while the more strongly depth-constrained
V4 had the lowest. Accuracy and macro F1 remain close for every version, so the
ordering is not explained by one dominant digit class.

## Paired training-seed changes

| Comparison | Mean accuracy change | 95% bootstrap CI | Right wins / ties / left wins |
|---|---:|---:|---:|
| V1 → V2 | +1.06 pp | +0.78 to +1.33 pp | 10 / 0 / 0 |
| V1 → V3 | +1.28 pp | +0.91 to +1.65 pp | 10 / 0 / 0 |
| V1 → V4 | −1.31 pp | −1.74 to −0.93 pp | 0 / 0 / 10 |
| V2 → V3 | +0.22 pp | approximately 0.00 to +0.44 pp | 6 / 2 / 2 |
| V2 → V4 | −2.37 pp | −2.72 to −2.04 pp | 0 / 0 / 10 |
| V3 → V4 | −2.59 pp | −2.93 to −2.28 pp | 0 / 0 / 10 |

V2 and V3 consistently outperform V1 across the selected training seeds, and
all three outperform V4. However, the V2–V3 interval touches zero and their
win/tie/loss record is mixed. The evidence is therefore insufficient to claim
that V3 reliably improves on V2 under training randomness alone.

## Comparison with split robustness

The training-seed intervals are generally narrower than the data-split
intervals. In this experiment, changing which observations enter the test set
creates more uncertainty than changing the random construction of the forest
on one fixed split. The conclusion that V4 is worse and that V2 improves over
V1 is stable under both sources of variation.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression digits-training-robustness
```

The machine-readable report is saved to
`artifacts/digits_training_robustness/summary.json`.

## Limitations

- Ten training seeds are a finite sensitivity sample.
- The bootstrap interval is conditional on the fixed seed-42 train-test split.
- Model configurations differ in both capacity and tree count; this experiment
  evaluates versions rather than attributing change to one causal parameter.
- A larger image dataset and neural image models remain necessary for broader
  conclusions.
