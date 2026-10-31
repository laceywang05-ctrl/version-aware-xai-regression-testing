# CIFAR-10 width-change consistency at the image level

## Question and scope

The three matched-seed width experiments each improved official-test accuracy.
This follow-up asks whether the *same* official test images changed, improved,
or regressed under all three training seeds. It compares the validation-selected
32/64/128-channel V1 and 40/80/160-channel V2 predictions from seeds 42, 43
and 44. The inputs are the existing 10,000-row official-test prediction files;
no model was retrained or selected using the test set for this analysis.

The analysis checks seed, architecture widths, source archive hash, ordered
official-test indices and ground-truth labels before aligning rows by image
index. “Changed” means the predicted class differs between V1 and V2 within
one seed. “Fixed” means V1 was wrong and V2 correct; “regressed” means V1 was
correct and V2 wrong. A class change between two wrong answers counts only as
“changed.”

## Results

| Outcome | Seed 42 | Seed 43 | Seed 44 | Same images in all 3 | Images in at least 2 | Images in any seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Predicted class changed | 891 | 880 | 855 | 185 | 740 | 1,701 |
| Wrong → correct | 442 | 417 | 411 | 23 | 228 | 1,019 |
| Correct → wrong | 285 | 306 | 288 | 10 | 119 | 750 |

For each pair of seeds, the Jaccard overlap of changed-image sets was
**0.262–0.276**. It was **0.113–0.127** for fixed-image sets and
**0.082–0.089** for regressed-image sets. Jaccard overlap is the number of
shared image indices divided by the number present in either set; these are
descriptive comparisons, not hypothesis tests.

Across all three runs, **255 distinct images** appeared as a fix under at
least one seed and a regression under another. Thus “V2 improved this image”
is a statement about a *specific trained model pair*, not a stable property
of the wider architecture. The aggregate gains of +1.57, +1.11 and +1.23
percentage points remain consistent in direction across the observed pairs,
while the affected images vary substantially.

The three runs share the same official test set, so their image-level outcomes
and test-set p-values are not independent replications. Seed 42 also has the
previously documented interrupted-training RNG caveat; seeds 43 and 44 trained
continuously. Three seeds do not establish a population-level probability of
improvement or regression. The dashboard should present a single pair's
sample-level findings with its seed and artifact identity, and should avoid
calling one image an architectural failure without further replication.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m xai_regression.cifar10_width_consistency \
  --pair 42 artifacts/cifar10_cnn_full_2026-10-29 artifacts/cifar10_cnn_wide_seed42 \
  --pair 43 artifacts/cifar10_cnn_full_seed43 artifacts/cifar10_cnn_wide_seed43 \
  --pair 44 artifacts/cifar10_cnn_full_seed44 artifacts/cifar10_cnn_wide_seed44 \
  --output artifacts/cifar10_width_instance_consistency.json
```

The machine-readable report is generated under Git-ignored `artifacts/`.
Its values are summarized here so that the evidence is reviewable without
checking large model or prediction files into Git.
