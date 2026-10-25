# Four-model handwritten-digit evaluation

## Objective

This experiment checks whether the upload workflow can compare four image
classifiers, rank them on a shared labelled test set, and expose concrete
sample-level improvements and regressions. It uses scikit-learn's bundled 8×8
handwritten-digit dataset so that the experiment is reproducible offline.

## Protocol

- Dataset: scikit-learn handwritten digits, 1,797 images and 10 classes.
- Split: fixed stratified 70/30 train-test split with random seed 42.
- Training set: 1,257 images.
- Independent test set: 540 images.
- Models: depth-limited decision tree, random forest, extra trees, and
  five-nearest-neighbours.
- Primary metrics: test accuracy and macro F1.
- Dashboard inspection subset: 60 test images selected to include correct,
  changed, and incorrect predictions. This subset is diagnostic rather than a
  representative performance sample.

The experiment can be reproduced with:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-demo \
  --output-dir artifacts/digits_four_model_evaluation
```

## Full held-out test results

| Rank | Model | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 1 | K-nearest neighbours | 98.70% | 98.68% |
| 2 | Extra trees | 98.33% | 98.31% |
| 3 | Random forest | 96.85% | 96.83% |
| 4 | Decision tree (`max_depth=3`) | 47.96% | 39.82% |

The close agreement between accuracy and macro F1 for the three stronger
models indicates that their overall performance is not being created by one
dominant digit class. The deliberately constrained decision tree performs much
worse and provides a useful baseline version for demonstrating improvements
and regressions.

## Dashboard diagnostic subset

On the curated 60-image inspection subset, the model ordering remained the
same: decision tree 41.67%, random forest 75.00%, extra trees 86.67%, and KNN
90.00% accuracy. These values must not be reported as unbiased test accuracy
because the subset was deliberately enriched for changed and incorrect
predictions.

For the illustrative decision-tree V1 versus KNN V2 comparison:

- accuracy increased from 41.67% to 90.00%;
- macro F1 increased from 32.02% to 91.07%;
- 32 samples changed from wrong to correct;
- 3 samples changed from correct to wrong;
- 22 samples were correct under both versions;
- 3 samples were wrong under both versions; and
- the exact paired McNemar p-value was approximately 4.18 × 10⁻⁷.

The candidate therefore shows strong paired evidence of improved predictive
correctness on this diagnostic subset. The remaining regressions are still
visible in the sample table and should be reviewed individually rather than
hidden by the aggregate improvement.

## Interpretation limits

- The dataset contains small 8×8 grayscale images and is not a substitute for
  a full MNIST or CIFAR-10 evaluation.
- The four models are lightweight demonstration baselines, not optimized image
  architectures.
- The 60-image dashboard subset is intended for error inspection, not unbiased
  model ranking; the 540-image held-out results provide the performance claim.
- Explanation similarity should not be interpreted across unlike explanation
  methods because such comparisons mix explainer behavior with model behavior.
- The very weak V1 is intentional and makes the end-to-end version comparison
  easy to demonstrate; future experiments should include closer model versions.

## Conclusion

The workflow successfully handled four uploaded image classifiers, preserved
the independent test split, ranked the models using accuracy and macro F1, and
identified exact images that improved or regressed. KNN achieved the strongest
held-out performance in this lightweight experiment, while the constrained
decision tree served as an intentionally weak baseline for validating the
regression-testing interface.
