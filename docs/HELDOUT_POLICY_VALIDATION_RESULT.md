## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-18
- Verification Status: VERIFIED
- Version Label: heldout_policy_validation_v1

# Held-Out Policy Validation

## Purpose

The 30-run benchmark found a 36.7% review rate in the seed-only control. This
follow-up tests whether that behaviour was partly caused by calibrating the
policy with only ten benign retrainings.

The quantile policy is unchanged. The benign baseline is expanded to 30 random
seeds, and the four candidate scenarios are evaluated on a separate set of 30
seeds. The calibration and evaluation seed sets are disjoint and neither
contains the reference seed.

## Design

- Dataset: scikit-learn Breast Cancer Wisconsin Diagnostic
- Reference model: 200-tree balanced random forest, seed 42
- Probe set: the same fixed 100 samples used by the earlier benchmark
- Calibration: 30 benign retrainings with unchanged model settings
- Evaluation: 30 new seeds per scenario, 120 candidate runs in total
- Interval: Wilson 95% confidence interval for each review rate
- Paired comparison: exact McNemar test against the seed-only control, with
  Bonferroni adjustment across three scenario comparisons

## Results

| Scenario | Mean prediction agreement | Review runs | Review rate | Wilson 95% CI |
|---|---:|---:|---:|---:|
| `seed_only_control` | 0.994 | 5/30 | 16.7% | 7.3%–33.6% |
| `shallow_depth` | 0.999 | 13/30 | 43.3% | 27.4%–60.8% |
| `larger_leaves` | 0.995 | 11/30 | 36.7% | 21.9%–54.5% |
| `constrained_candidate` | 0.997 | 15/30 | 50.0% | 33.2%–66.8% |

Expanding the benign calibration set reduced the held-out control review rate
from the earlier descriptive estimate of 36.7% to 16.7%. The model-configuration
scenarios remained more frequently flagged than the control.

## Paired control comparisons

| Candidate scenario | Review-rate difference | Control clear / candidate review | Control review / candidate clear | Exact p | Adjusted p |
|---|---:|---:|---:|---:|---:|
| `shallow_depth` | +26.7 pp | 11 | 3 | 0.057 | 0.172 |
| `larger_leaves` | +20.0 pp | 6 | 0 | 0.031 | 0.094 |
| `constrained_candidate` | +33.3 pp | 12 | 2 | 0.013 | 0.039 |

Only the combined-constraint comparison remains below 0.05 after adjustment
for the three paired comparisons. This is evidence that the current policy can
separate this controlled configuration change from seed-only retraining on the
held-out seeds. It is not evidence that every flagged change is harmful.

The shallow-depth candidates retain 99.9% mean prediction agreement while
43.3% of runs trigger explanation review. This continues to show that
prediction-only testing and explanation monitoring answer different questions.

## Threshold changes

| Metric | Earlier run-level limit | Expanded-calibration limit |
|---|---:|---:|
| Cosine similarity | 9.20% | 7.55% |
| Rank correlation | 7.10% | 7.55% |
| Top-k overlap | 5.10% | 6.00% |
| Direction flip rate | 3.00% | 4.55% |

The direction-flip limit increased from 3.00% to 4.55%, which reduced control
warnings attributable to that metric from nine in the earlier evaluation to
three in the new held-out evaluation. This supports the diagnosis that the
ten-seed calibration produced an unstable run-level threshold.

## Interpretation limits

The validation holds out random seeds, not datasets, model families or probe
sets. The results therefore remain a single-dataset case study. The follow-up
was also motivated by the earlier benchmark result, so it should be treated as
a structured robustness check rather than a final confirmatory study.

The next step is to repeat the fixed policy on a second tabular dataset before
making broader claims. A later policy revision should be specified before its
evaluation seeds are generated.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/benchmark_calibration.yaml \
  --scenarios config/version_benchmark_heldout.yaml
```

Reports and the comparison figure are generated under
`artifacts/experiment_10_heldout_benchmark/`.
