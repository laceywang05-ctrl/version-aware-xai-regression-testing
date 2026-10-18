# Controlled Model-Version Benchmark

## Purpose

The earlier experiment evaluated only one reference/candidate pair. This
benchmark asks a stronger question: does the same calibrated SHAP policy behave
differently for ordinary retraining noise and for controlled model-configuration
changes?

The benchmark reuses the fixed Breast Cancer Wisconsin probe set, Tree SHAP
configuration, explanation metrics and benign-retraining policy from the main
experiment. It evaluates four candidate-version scenarios against the same
reference. Each scenario is repeated with five candidate seeds that are not
used to calibrate the benign baseline.

## Preliminary results

| Scenario | Change | Mean prediction agreement | Review runs | Review rate |
|---|---|---:|---:|---:|
| `seed_only_control` | Random seed only | 0.996 | 1/5 | 20% |
| `shallow_depth` | `max_depth=4` | 1.000 | 4/5 | 80% |
| `larger_leaves` | `min_samples_leaf=5` | 0.996 | 2/5 | 40% |
| `constrained_candidate` | `max_depth=4`, `min_samples_leaf=5` | 0.998 | 4/5 | 80% |

The seed-only control produced one warning in five runs, so the current policy
does have observable false-positive behaviour. However, the shallow-depth and
combined-constraint scenarios were flagged in four of five runs, compared with
one of five control runs. This preliminary separation suggests that the policy
is responding to model-configuration changes more often than to seed-only
retraining noise.

Most importantly, the shallow-depth candidates retained 100% mean prediction
agreement while 80% of their runs triggered an explanation warning. This is an
example of the gap that ordinary prediction regression testing cannot observe.

The larger-leaf scenario was flagged in only two of five runs. That result is
also informative: the tool does not label every implementation change as an
explanation regression, and different configuration changes have different
detectability under the current metrics and thresholds.

## Interpretation limits

These are preliminary case-study results, not estimates of general false-positive
or detection rates. The benchmark currently uses one dataset, one reference
model, five runs per scenario and three hand-selected configuration changes. A
review flag means that an explanation metric exceeded its calibrated limit; it
does not establish that the candidate is defective or unsafe.

The next research step is to expand the number of independent control and change
runs, report uncertainty intervals for review rates, and later repeat the study
on a second tabular dataset. This will support a more defensible evaluation of
policy sensitivity and false positives.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark
```

Machine-readable results and the comparison figure are generated under
`artifacts/experiment_09_version_benchmark/`.
