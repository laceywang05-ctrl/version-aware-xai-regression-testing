# Independent-Seed Confirmation of the LIME Policy

## Material Passport

- Material ID: `lime-confirmatory-policy-v1`
- Type: repeated code experiment with independent evaluation seeds
- Verification status: `VERIFIED`
- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine
- Explanation method: LIME
- Calibration seeds: 1001–1030
- Earlier diagnostic seeds: 2001–2030
- Confirmatory evaluation seeds: 3001–3030
- Fixed probes: 20 samples per dataset
- LIME perturbation samples: 1,000 per explanation
- Preselected run rule: Bonferroni-adjusted 98.75th-percentile metric limits
- External data upload: none

## Research purpose

The multiplicity-aware rule was introduced after examining the first held-out
evaluation set. This experiment therefore applies the already selected rule to
a second evaluation-seed set that was not used for calibration or for the
earlier policy comparison. It tests whether the lower control review rates and
the separation between fixed-model control and cross-version candidate runs
repeat without changing the model, probe, explainer settings or thresholds.

## Results

| Dataset | Policy | Control review rate | Candidate review rate | Exact paired p | Status |
|---|---|---:|---:|---:|---|
| Breast Cancer Wisconsin | Unadjusted | 13.3% (4/30), 95% CI [5.3%, 29.7%] | 53.3% (16/30), 95% CI [36.1%, 69.8%] | 0.001831 | SEPARATED |
| Breast Cancer Wisconsin | Bonferroni | 6.7% (2/30), 95% CI [1.8%, 21.3%] | 53.3% (16/30), 95% CI [36.1%, 69.8%] | 0.000519 | SEPARATED |
| Wine (binary) | Unadjusted | 13.3% (4/30), 95% CI [5.3%, 29.7%] | 63.3% (19/30), 95% CI [45.5%, 78.1%] | 0.000275 | SEPARATED |
| Wine (binary) | Bonferroni | 3.3% (1/30), 95% CI [0.6%, 16.7%] | 43.3% (13/30), 95% CI [27.4%, 60.8%] | 0.001831 | SEPARATED |

Under the adjusted rule, the Breast Cancer paired comparison contained 15
control-clear/candidate-review pairs and one pair in the opposite direction.
Wine contained 13 control-clear/candidate-review pairs and one pair in the
opposite direction.

## Confirmation against the earlier held-out set

| Dataset | Earlier adjusted control | New adjusted control | Earlier adjusted candidate | New adjusted candidate |
|---|---:|---:|---:|---:|
| Breast Cancer Wisconsin | 10.0% | 6.7% | 63.3% | 53.3% |
| Wine (binary) | 3.3% | 3.3% | 43.3% | 43.3% |

The Wine result reproduced exactly at the review-count level: one control run
and 13 candidate runs were selected in both independent evaluation sets. The
Breast Cancer rates changed moderately but retained the same qualitative
pattern: fewer control selections than candidate selections and a significant
paired separation.

## Metric-level diagnosis

For Breast Cancer, the two adjusted control selections were caused by
`top_k_overlap`, while most candidate selections were caused by cosine
similarity. For Wine, one control selection was caused by direction-flip rate;
all 13 adjusted candidate selections were caused by cosine similarity.

## Descriptive interpretation

The independent-seed results support the repeatability of the central finding:
the cross-version candidate is selected more often than fixed-model LIME noise
under the multiplicity-aware policy. The adjustment reduced control selections
relative to the unadjusted rule on both datasets in this new evaluation set.

This evidence supports using the adjusted policy as the stronger of the two
current research prototypes. It does not establish a universal threshold, a
production error guarantee or proof that a selected candidate model is harmful.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/lime_confirmatory.yaml
PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/wine_lime_confirmatory.yaml
PYTHONPATH=src python -m xai_regression lime-multiplicity \
  --breast-report-dir artifacts/experiment_18_breast_lime_confirmatory/reports \
  --wine-report-dir artifacts/experiment_19_wine_lime_confirmatory/reports \
  --output-dir artifacts/experiment_20_lime_confirmatory_policy
```

## Output directories

- `artifacts/experiment_18_breast_lime_confirmatory/`
- `artifacts/experiment_19_wine_lime_confirmatory/`
- `artifacts/experiment_20_lime_confirmatory_policy/`

The complete two-dataset workflow was rerun after implementation. All three
summary reports matched their first-run SHA-256 hashes exactly:

- Breast Cancer: `81460afe19d8863f5ce173f5e3dcdf429fd2ce1e0960dd04b0a504c4f3f3ae6e`
- Wine: `60c3f079fd465360a051463cd88e840b2b74a8178a0c9284cb57d05ecd129ae0`
- Combined policy: `1cc6bbe622579c12cd10afae29891d00b950caa61e816c4faa750d7086ab19a0`

## Limitations

- The confirmation changes only explainer random seeds; it retains the same
  train/test split, fixed probe and candidate model configuration.
- Thirty runs still produce wide confidence intervals.
- Bonferroni is conservative and does not model dependence among metrics.
- Both datasets are small tabular binary-classification tasks.
- A final policy decision should also consider operational review cost and a
  larger collection of datasets and candidate-change types.
