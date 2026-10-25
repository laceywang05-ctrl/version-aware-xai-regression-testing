# Close-version handwritten-digit benchmark

## Research question

Can the regression-testing workflow distinguish meaningful model changes when
four versions have similar aggregate performance, rather than comparing an
obviously weak baseline with a strong candidate?

## Method

Four random-forest versions were trained on the same fixed stratified split of
the scikit-learn handwritten-digit dataset. The training set contained 1,257
images and the independent test set contained 540 images. Versions varied in
tree count, maximum depth, and random seed.

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression digits-close-benchmark
```

## Results

| Version | Trees | Maximum depth | Accuracy | Macro F1 | Correct / 540 |
|---|---:|---:|---:|---:|---:|
| V1 | 40 | 8 | 96.48% | 96.46% | 521 |
| V2 | 80 | 10 | 96.48% | 96.44% | 521 |
| V3 | 120 | Unlimited | 96.67% | 96.65% | 522 |
| V4 | 80 | 6 | 94.81% | 94.84% | 512 |

V1 and V2 had identical accuracy but did not make exactly the same predictions:
each version uniquely classified two test images correctly, and their overall
prediction agreement was 98.89%. Their exact paired McNemar p-value was 1.0,
so this test provides no evidence that either version has better correctness.

V3 obtained one additional correct prediction relative to V1. The models had
98.15% prediction agreement, with three V1-only correct samples and four
V3-only correct samples. The exact McNemar p-value was again 1.0. The small
observed gain is therefore not statistically conclusive on this test set.

The depth-constrained V4 showed a clearer regression. Compared with V1, 12
samples were correct only under V1 and 3 only under V4. The exact McNemar
p-value was 0.0352. Similar comparisons against V2 and V3 produced p-values of
0.0117 and 0.0129 respectively.

## Interpretation

This experiment demonstrates why aggregate accuracy alone is insufficient for
version-aware testing. V1 and V2 have identical accuracy, yet four samples
exchange correctness status. Those samples remain important for inspection,
even though the paired test does not favor either version.

The results also show that non-significance is not evidence of equivalence.
For V1–V3, the number of discordant correctness outcomes is only seven, so the
test has limited information about a very small difference. A larger held-out
set or repeated evaluation would be required for a stronger conclusion.

## Conclusion

The workflow can represent both subtle and clear version changes. Closely
matched V1–V3 models require sample-level review and cautious statistical
language, while the shallower V4 produces a detectable paired regression. This
is a more realistic validation of the system than comparing only an extremely
weak baseline with a high-performing candidate.
