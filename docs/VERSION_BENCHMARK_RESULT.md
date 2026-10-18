# Controlled Model-Version Benchmark

## Purpose

The earlier experiment evaluated only one reference/candidate pair. This
benchmark asks a stronger question: does the same calibrated SHAP policy behave
differently for ordinary retraining noise and for controlled model-configuration
changes?

The benchmark reuses the fixed Breast Cancer Wisconsin probe set, Tree SHAP
configuration, explanation metrics and benign-retraining policy from the main
experiment. It evaluates four candidate versions against the same reference.

## Preliminary results

| Scenario | Change | Prediction agreement | Flagged metric | Status |
|---|---|---:|---|---|
| `seed_only_control` | Random seed only | 1.000 | None | Within baseline |
| `shallow_depth` | `max_depth=4` | 1.000 | Direction-flip rate | Review |
| `larger_leaves` | `min_samples_leaf=5` | 1.000 | None | Within baseline |
| `constrained_candidate` | `max_depth=4`, `min_samples_leaf=5` | 0.990 | Cosine similarity | Review |

The seed-only control remained within the benign-retraining baseline. Two of
the three controlled model changes triggered a review signal. Most importantly,
the shallow-depth candidate retained identical probe predictions while still
triggering an explanation warning. This is an example of the gap that ordinary
prediction regression testing cannot observe.

The larger-leaf candidate did not trigger the current policy. That result is
also informative: the tool does not label every implementation change as an
explanation regression. It reports only changes that exceed the empirically
calibrated run-level limits.

## Interpretation limits

These are preliminary case-study results, not estimates of general false-positive
or detection rates. The benchmark currently uses one dataset, one reference
model, one seed-only control candidate and three hand-selected configuration
changes. A review flag means that an explanation metric exceeded its calibrated
limit; it does not establish that the candidate is defective or unsafe.

The next research step is to repeat each scenario across multiple candidate
seeds and, later, a second tabular dataset. That will support confidence
intervals and a more defensible evaluation of policy sensitivity and false
positives.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark
```

Machine-readable results and the comparison figure are generated under
`artifacts/experiment_09_version_benchmark/`.
