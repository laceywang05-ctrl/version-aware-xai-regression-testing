# Tree SHAP Multiplicity-Aware Policy

## Purpose

The original Tree SHAP run decision reviews a model seed when any one of four
metrics exceeds its 95th-percentile benign-retraining limit. Testing several
metrics in parallel can increase the probability that a benign run is reviewed
by chance. This experiment evaluates whether a Bonferroni-adjusted run rule can
reduce those control warnings without losing the combined-change signal.

## Compared rules

- **Unadjusted:** each metric uses the 95th percentile of its calibration run
  distribution.
- **Bonferroni:** with four metrics and family alpha 0.05, each metric uses the
  `1 - 0.05 / 4 = 0.9875` calibration quantile.

Both rules reuse the same 30 calibration seeds and the same 30 paired control
and candidate seeds from each of the eight additional-split robustness cells.

## Aggregate results

| Rule | Control reviews | Candidate reviews | Significantly separated cells |
| --- | ---: | ---: | ---: |
| Unadjusted | 35/240 (14.6%) | 197/240 (82.1%) | 8/8 |
| Bonferroni | 16/240 (6.7%) | 173/240 (72.1%) | 7/8 |

The adjustment removes 19 control warnings, a relative reduction of about 54%,
but also removes 24 candidate warnings. Seven cells retain statistically
significant paired separation. The exception is Breast Cancer Random Forest on
split 73, where the Bonferroni rule produces a 6.7% control review rate and a
23.3% candidate review rate with exact McNemar `p=0.0625`.

Gradient Boosting remains strongly separated in every cell: candidate review
rates stay at 100%, while control review rates under the adjusted rule range
from 0% to 10%.

## Decision

The Bonferroni rule should not replace the existing rule as a single universal
pass/fail threshold because it loses one of eight held-out separations. A more
defensible presentation is a two-level advisory policy:

1. **REVIEW:** any metric exceeds its ordinary calibrated limit. This prioritises
   sensitivity and identifies cases for inspection.
2. **HIGH-CONFIDENCE REVIEW:** a metric exceeds the multiplicity-adjusted limit.
   This produces fewer benign-control warnings but is less sensitive.

Neither level proves that a model is defective. The two levels communicate the
strength of the signal and preserve the observed sensitivity-specificity
trade-off instead of hiding it behind one threshold.

## Reproduction

First generate the split-robustness artifacts, including saved benign samples:

```bash
PYTHONPATH=src python -m xai_regression shap-split-robustness
```

Then compare the two run policies:

```bash
PYTHONPATH=src python -m xai_regression shap-multiplicity
```

Reports are written under
`artifacts/experiment_32_shap_multiplicity_policy/reports/`.
