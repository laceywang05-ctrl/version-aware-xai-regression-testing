# LIME Robustness Across Data Splits and Probe Subsets

## Material Passport

- Material ID: `lime-split-probe-robustness-v1`
- Type: repeated robustness experiment
- Verification status: `VERIFIED`
- Datasets: Breast Cancer Wisconsin Diagnostic and binary Wine
- Explanation method: LIME
- Data split seeds: 17, 42 and 73
- Calibration seeds per split: 30
- Evaluation seeds per split: 30, disjoint from calibration
- Fixed probes per split: 20
- LIME perturbation samples: 1,000 per explanation
- Run policy: Bonferroni-adjusted 98.75th-percentile metric limits
- External data upload: none

## Research purpose

All earlier experiments used one train/test split and one fixed probe subset.
This final experiment tests whether the multiplicity-aware LIME result persists
when the training data, test data and explained probe samples change.

Split 42 reuses the completed independent-seed confirmation. Splits 17 and 73
are newly executed. Every split independently trains the reference and candidate
models, selects its probe from its own test partition, calibrates metric limits
and evaluates held-out explainer seeds. Results are reported by split because
the partitions overlap and should not be treated as independent studies.

## Probe variation

The new probes were nearly disjoint from the split-42 probe:

| Dataset | Split | Shared probe rows with split 42 |
|---|---:|---:|
| Breast Cancer Wisconsin | 17 | 0/20 |
| Breast Cancer Wisconsin | 73 | 2/20 |
| Wine (binary) | 17 | 2/20 |
| Wine (binary) | 73 | 4/20 |

## Bonferroni-adjusted results

| Dataset | Split | Prediction agreement | Control review rate | Candidate review rate | Exact paired p | Status |
|---|---:|---:|---:|---:|---:|---|
| Breast Cancer Wisconsin | 17 | 100% | 10.0% (3/30) | 30.0% (9/30) | 0.031250 | SEPARATED |
| Breast Cancer Wisconsin | 42 | 100% | 6.7% (2/30) | 53.3% (16/30) | 0.000519 | SEPARATED |
| Breast Cancer Wisconsin | 73 | 100% | 6.7% (2/30) | 33.3% (10/30) | 0.007812 | SEPARATED |
| Wine (binary) | 17 | 95% | 30.0% (9/30) | 100.0% (30/30) | 0.000001 | SEPARATED |
| Wine (binary) | 42 | 100% | 3.3% (1/30) | 43.3% (13/30) | 0.001831 | SEPARATED |
| Wine (binary) | 73 | 100% | 6.7% (2/30) | 0.0% (0/30) | 0.500000 | NO_SEPARATION |

Overall status: `SPLIT_SENSITIVE`.

## Diagnosis

Breast Cancer retained paired separation on all three splits. Candidate review
rates ranged from 30.0% to 53.3%, while control rates ranged from 6.7% to 10.0%.
The magnitude varied, but the direction of the result was consistent.

Wine was strongly split dependent. Split 17 selected every candidate run but
also had a high 30.0% control review rate. Split 42 reproduced the earlier
low-control result. On split 73, neither the unadjusted nor adjusted policy
separated the candidate: the adjusted rule selected zero candidate runs and two
control runs.

For Wine split 73, reference and candidate predictive metrics were both perfect
on the test set, probe prediction agreement was 100%, and the mean paired-seed
cosine similarity was 0.9903. The controlled candidate configuration therefore
produced little detectable explanation change for this particular fitted-model
and probe combination. This is not evidence that the detector failed; it shows
that the chosen model change does not induce the same explanation difference on
every data split.

## Descriptive interpretation

The experiment supports a narrower conclusion than the fixed-split studies:
the framework can distinguish cross-version explanation changes from LIME noise,
but detection strength depends on the fitted models and probe population. The
current thresholds and one candidate configuration must not be described as
universally robust across data splits.

This negative result is important for the final dissertation. It demonstrates
why version-aware explanation testing must version the training split and probe
set alongside the model and explainer settings, and why a warning is an
investigation trigger rather than an automatic declaration of regression.

## Reproduction

The complete workflow was rerun from the saved configuration after the initial
experiment. All six split-level count, p-value and decision results were
reproduced exactly. The regenerated summary file had the same SHA-256 digest:

`3bfd8c3976a4a90e68e865ff7451589d6f3b03bc3d000644a871aedf8a59d246`

```bash
PYTHONPATH=src python -m xai_regression lime-split-robustness
```

Outputs are written under:

- `artifacts/experiment_27_lime_split_probe_robustness/runs/`
- `artifacts/experiment_27_lime_split_probe_robustness/policies/`
- `artifacts/experiment_27_lime_split_probe_robustness/reports/`
- `artifacts/experiment_27_lime_split_probe_robustness/figures/`

## Limitations

- Three splits are enough to expose sensitivity but not to estimate a population
  distribution over possible splits.
- Train/test partitions overlap, so split-level results are not independent.
- Each split uses only 20 probe samples and 30 calibration/evaluation seeds.
- Only one random-forest candidate configuration is tested per dataset.
- The study covers two small tabular binary-classification datasets.
- No universal production threshold or error-rate guarantee is established.
