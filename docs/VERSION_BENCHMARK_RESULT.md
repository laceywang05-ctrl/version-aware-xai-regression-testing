## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-18
- Verification Status: VERIFIED
- Version Label: version_benchmark_30_runs_v1

# Controlled Model-Version Benchmark

## Purpose

The earlier experiment evaluated only one reference/candidate pair. This
benchmark asks a stronger question: does the same calibrated SHAP policy behave
differently for ordinary retraining noise and for controlled model-configuration
changes?

The benchmark reuses the fixed Breast Cancer Wisconsin probe set, Tree SHAP
configuration, explanation metrics and benign-retraining policy from the main
experiment. It evaluates four candidate-version scenarios against the same
reference. Each scenario is repeated with 30 candidate seeds that are not
used to calibrate the benign baseline.

## Expanded results

| Scenario | Change | Mean prediction agreement | Review runs | Review rate | Wilson 95% CI |
|---|---|---:|---:|---:|---:|
| `seed_only_control` | Random seed only | 0.997 | 11/30 | 36.7% | 21.9%–54.5% |
| `shallow_depth` | `max_depth=4` | 1.000 | 20/30 | 66.7% | 48.8%–80.8% |
| `larger_leaves` | `min_samples_leaf=5` | 0.996 | 11/30 | 36.7% | 21.9%–54.5% |
| `constrained_candidate` | `max_depth=4`, `min_samples_leaf=5` | 0.996 | 20/30 | 66.7% | 48.8%–80.8% |

The expanded seed-only control produced 11 warnings in 30 runs. This is a much
stronger indication of false-positive behaviour than the earlier five-run
result suggested. The shallow-depth and combined-constraint scenarios were
flagged in 20 of 30 runs. Their point estimates are 30 percentage points above
the control, but the confidence intervals overlap, so the current evidence does
not establish reliable separation.

The shallow-depth candidates retained approximately 100% mean prediction
agreement while 66.7% of their runs triggered an explanation warning. This
continues to demonstrate that prediction-only checks can miss explanation
changes, but the warning must be interpreted alongside the high control review
rate.

The larger-leaf scenario matched the control review rate at 36.7%. Under the
current metrics and thresholds, this configuration change is not separated
from seed-only retraining.

The direction-flip metric produced most of the warnings: 9 control warnings,
18 shallow-depth warnings, 9 larger-leaf warnings and 10 combined-constraint
warnings. This concentration indicates that the direction-flip policy needs
specific calibration analysis before the overall review flag can be treated as
reliable.

## Interpretation limits

These are preliminary case-study results, not estimates of general false-positive
or detection rates. The benchmark currently uses one dataset, one reference
model, 30 runs per scenario and three hand-selected configuration changes. A
review flag means that an explanation metric exceeded its calibrated limit; it
does not establish that the candidate is defective or unsafe.

The next research step is to diagnose metric-level false positives, especially
for direction flips, and evaluate a pre-specified revised policy against held-out
control seeds. The study should then be repeated on a second tabular dataset.
This avoids tuning and evaluating the policy on the same benchmark runs.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark
```

Machine-readable results and the comparison figure are generated under
`artifacts/experiment_09_version_benchmark/`.

The verified run completed successfully with 36 automated tests passing.
