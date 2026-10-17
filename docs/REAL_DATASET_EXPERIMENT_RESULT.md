## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-12
- Verification Status: VERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: real-dataset-validation-v1
- **Type**: analysis
- **Status**: completed
- **Command**: `python experiments/07_real_dataset_validation.py`
- **Working Directory**: repository root
- **Duration**: approximately 11 seconds
- **Exit Code**: 0

### Design

The experiment uses the scikit-learn Breast Cancer Wisconsin Diagnostic dataset:
569 samples and 30 numeric features. The target is recoded so malignant cases are
the positive class. A stratified 75/25 train-test split and a fixed 100-sample
probe set are used throughout.

The accepted reference is a 200-tree balanced random forest. Ten retrainings with
the same hyperparameters and different random seeds estimate normal explanation
variation. The controlled candidate keeps 200 trees but introduces a maximum
depth of 4 and a minimum leaf size of 5. Positive-class Tree SHAP explanations
are compared on exactly matched probe samples.

### Output Files

| File | Purpose |
|---|---|
| `artifacts/experiment_05_real_dataset/reports/summary.json` | Machine-readable overall result |
| `artifacts/experiment_05_real_dataset/reports/policy_evaluation.csv` | Metric limits, breach rates and flags |
| `artifacts/experiment_05_real_dataset/reports/global_feature_comparison.csv` | Versioned global SHAP importance |
| `artifacts/experiment_05_real_dataset/reports/candidate_per_sample.csv` | Matched local explanation comparisons |
| `artifacts/experiment_05_real_dataset/figures/real_dataset_validation.png` | Main result figure |

### Output Summary

| Result | Reference | Candidate |
|---|---:|---:|
| Accuracy | 0.972 | 0.958 |
| Balanced accuracy | 0.962 | 0.951 |
| Malignant-class F1 | 0.961 | 0.942 |
| ROC AUC | 0.998 | 0.997 |

The two versions agree on 99% of fixed-probe predictions. Their mean explanation
cosine similarity is 0.972, mean absolute-rank correlation is 0.907, mean top-5
overlap is 0.962, and mean direction-flip rate is 0.100.

The candidate cosine breach rate is 0.100, slightly above the provisional benign
run-level limit of 0.092. Rank correlation, top-5 overlap and direction-flip rate
remain within their benign baselines. The overall status is therefore **REVIEW**,
not automatic failure.

The largest global ranking change among the reference model's top features is
`worst radius`, which moves from rank 5 to rank 1 in the candidate. `worst
perimeter` moves from rank 1 to rank 3. These changes provide concrete cases for
investigation even though predictive agreement remains high.

### Anomalies Detected

- Cosine-similarity breach rate is marginally above the provisional run-level
  baseline.
- The calibration uses only ten benign retrainings, so the 95th-percentile
  run-level limit is coarse.
- The result concerns one dataset, one model family and one explainer; it must not
  be interpreted as a universal threshold.

