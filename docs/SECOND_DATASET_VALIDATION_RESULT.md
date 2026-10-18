## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-18
- Verification Status: VERIFIED
- Version Label: wine_binary_validation_v1

# Second-Dataset SHAP Policy Validation

## Purpose

This experiment tests whether the version-aware SHAP workflow produces the same
pattern on a second tabular dataset. It uses the scikit-learn Wine dataset and
retains classes 0 and 1 as a binary classification task, with class 1 treated as
the positive class.

The experiment is a generalisation check, not a direct replication. The wine
task has fewer samples and features than the Breast Cancer Wisconsin task, and
the same model configuration changes can affect its trees differently.

## Design

- Dataset: scikit-learn Wine, classes 0 and 1 only
- Samples and features: 130 samples, 13 numeric features
- Split: stratified 70/30 train-test split
- Probe set: 30 fixed test samples
- Model: 200-tree balanced random forest
- Calibration: 30 benign retraining seeds
- Evaluation: 30 disjoint held-out seeds per scenario
- Candidate runs: 120
- Explanation: positive-class Tree SHAP
- Statistical summary: Wilson 95% confidence intervals and paired exact
  McNemar tests with Bonferroni adjustment

## Results

| Scenario | Mean accuracy | Mean prediction agreement | Review runs | Review rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| `seed_only_control` | 0.993 | 1.000 | 1/30 | 3.3% | 0.6%–16.7% |
| `shallow_depth` | 0.992 | 1.000 | 2/30 | 6.7% | 1.8%–21.3% |
| `larger_leaves` | 0.981 | 0.998 | 23/30 | 76.7% | 59.1%–88.2% |
| `constrained_candidate` | 0.981 | 0.998 | 23/30 | 76.7% | 59.1%–88.2% |

The control review rate is low at 3.3%. Shallow trees remain close to the
control, while larger leaves and combined constraints are flagged in 76.7% of
runs despite 99.8% mean prediction agreement.

## Paired control comparisons

| Candidate scenario | Review-rate difference | Control clear / candidate review | Control review / candidate clear | Exact p | Adjusted p |
|---|---:|---:|---:|---:|---:|
| `shallow_depth` | +3.3 pp | 1 | 0 | 1.000 | 1.000 |
| `larger_leaves` | +73.3 pp | 22 | 0 | <0.001 | <0.001 |
| `constrained_candidate` | +73.3 pp | 22 | 0 | <0.001 | <0.001 |

Both leaf-constrained scenarios remain clearly separated from the control after
adjustment. Shallow depth does not separate from the control on this task.

## Dataset-specific interpretation

The reference forest already contains relatively shallow trees: the mean tree
depth is 3.35 and the maximum is 6. Limiting depth to 4 therefore changes the
model only modestly. Increasing `min_samples_leaf` to 5 has a stronger effect on
the fitted tree structures and on the SHAP explanations.

This result differs from the breast-cancer benchmark, where combined constraints
provided the clearest held-out separation. The difference supports a central
design requirement: explanation-regression thresholds and detectable changes
must be validated for each dataset and model context rather than treated as
universal constants.

## Interpretation limits

The binary wine task contains only 130 samples and excludes the original third
class. The result therefore demonstrates cross-dataset feasibility but does not
establish broad external validity. Review signals indicate explanation changes
beyond the calibrated baseline, not harmful or unsafe model behaviour.

## Reproduction

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/wine_benchmark_calibration.yaml \
  --scenarios config/wine_benchmark_heldout.yaml
```

Reports and the comparison figure are generated under
`artifacts/experiment_11_wine_heldout_benchmark/`.
