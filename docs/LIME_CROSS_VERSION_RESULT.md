# LIME Cross-Version Comparison Result

## Purpose

This experiment compares LIME explanations from a reference and candidate
random forest, then evaluates whether the observed changes exceed LIME's
fixed-model random-seed variation.

## Design

- Dataset: Breast Cancer Wisconsin Diagnostic
- Probe set: 20 fixed samples
- Repeated LIME seeds: 10
- Perturbed samples per explanation: 1,000
- Matched cross-version comparisons: 200
- Policy calibration: 5th/95th percentile fixed-model limits
- Status semantics: advisory review signal, not an automatic failure

Two comparisons are reported:

1. **Noise-matched comparison:** the reference baseline seed is compared with
   each candidate seed, matching the independent-seed structure used by the
   fixed-model baseline. This comparison is used for policy evaluation.
2. **Paired-seed comparison:** reference and candidate use the same LIME seed.
   This controls the random perturbation sequence and is reported
   descriptively, but is not evaluated against the independent-seed policy.

## Predictive results

| Measure | Reference | Candidate |
|---|---:|---:|
| Accuracy | 0.9720 | 0.9580 |
| Balanced accuracy | 0.9623 | 0.9512 |
| F1, malignant class | 0.9608 | 0.9423 |
| ROC AUC | 0.9976 | 0.9973 |

Prediction agreement on the 20 fixed probe samples was 0.950.

## Explanation results

| Metric | Fixed-model baseline | Noise-matched cross-version | Paired-seed cross-version |
|---|---:|---:|---:|
| Cosine similarity | 0.8494 | 0.8072 | 0.9513 |
| Absolute rank correlation | 0.5843 | 0.5540 | 0.8467 |
| Top-5 feature overlap | 0.7680 | 0.7210 | 0.8130 |
| Direction-flip rate | 0.2537 | 0.2697 | 0.0977 |

## Advisory policy result

Overall status: **REVIEW**.

Cosine similarity was the only flagged metric. Its cross-version sample breach
rate was 0.115, compared with a fixed-model run-level limit of 0.0775. Rank
correlation, top-5 overlap and direction-flip rate remained within their
provisional fixed-model limits. Across all four sample-level checks, 12.5% of
matched comparisons breached at least one limit.

## Interpretation and limitations

The result indicates that some cross-version LIME attribution vectors changed
more than expected from the measured fixed-model random-seed baseline. It does
not prove that the candidate model is defective or that real-world concept
drift occurred. The strong paired-seed similarities also show that conclusions
depend on how explainer randomness is controlled.

This is a small initial experiment with 20 probe samples, one dataset, one model
pair and provisional quantile thresholds. The `REVIEW` status should therefore
be used to select cases for inspection, not to block deployment automatically.
