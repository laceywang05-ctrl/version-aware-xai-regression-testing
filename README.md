# Version-Aware XAI Regression Testing

An FYP project for detecting unexpected changes in model explanations across machine-learning model versions.

## Research question

When a model changes from version `v1` to `v2`, prediction accuracy may remain acceptable while the model changes which features it relies on. This project investigates how to identify those changes reliably and distinguish them from normal variation in the explanation method.

## Initial scope

- Tabular classification
- Two model versions: an accepted baseline and a candidate
- A fixed probe dataset and preprocessing pipeline
- SHAP and LIME explanations
- Same-model repeated runs to estimate natural explainer variance
- Cross-version comparison using explanation-similarity metrics

Integrated Gradients and Grad-CAM are included in the literature review and remain possible later extensions for differentiable and image models.

## Proposed workflow

1. Register the model, dataset, preprocessing and explainer versions.
2. Run a fixed probe dataset through the accepted and candidate models.
3. Generate explanations with controlled settings.
4. Estimate same-model explanation variance through repeated runs.
5. Compare versions using metrics appropriate to the explanation artifact.
6. Apply a drift policy and report warnings for investigation.

```text
Register versions
      |
Run fixed probe dataset
      |
Generate controlled explanations
      |
Compare with accepted baseline
      |
Apply policy and report warning
```

## Candidate comparison metrics

The literature review identifies several candidate metrics for later evaluation:

- cosine similarity of attribution vectors;
- Spearman rank correlation;
- top-k feature overlap;
- contribution-direction flip rate.

Possible later metrics include distribution divergence for aggregate explanations and heatmap similarity for Grad-CAM.

The final metric set and thresholds will be selected after the initial scope is confirmed with the supervisor.

## Repository structure

```text
.
├── config/                 # Reproducible experiment settings
├── docs/                   # Research scope, architecture and literature map
├── src/xai_regression/     # Placeholder for the future implementation
├── CONTRIBUTING.md
└── pyproject.toml
```

## Current status

The repository now contains a working Tree SHAP proof of concept. It logs and
compares two model versions on a fixed probe dataset, verifies deterministic
same-model repeatability, estimates normal explanation variation across ten
benign retraining seeds, and applies a provisional review policy. Initial results
are summarised in [`docs/INITIAL_EXPERIMENT_RESULTS.md`](docs/INITIAL_EXPERIMENT_RESULTS.md).

The same workflow has also been validated on the real Breast Cancer Wisconsin
Diagnostic dataset. The reference and candidate agree on 99% of probe
predictions, while the explanation policy still identifies a small excess of
cosine-similarity breaches for review. See
[`docs/REAL_DATASET_EXPERIMENT_RESULT.md`](docs/REAL_DATASET_EXPERIMENT_RESULT.md).

Run the current experiment sequence from the repository root:

```bash
python experiments/02_generate_two_versions.py
python experiments/03_compare_explanations.py
python experiments/04_same_model_stability.py
python experiments/05_retraining_baseline.py
python experiments/06_apply_provisional_policy.py
python experiments/07_real_dataset_validation.py --config config/default.yaml
```

The YAML configuration records the dataset split, fixed-probe size, reference
and candidate model settings, benign-retraining seeds, explainer, comparison
metrics and provisional policy quantiles used by the real-dataset experiment.

The same real-dataset workflow can be launched through the package entry point:

```bash
PYTHONPATH=src python -m xai_regression run --config config/default.yaml
```

Measure LIME's own stochastic variation while keeping the trained model,
training data and fixed probe samples unchanged:

```bash
PYTHONPATH=src python -m xai_regression lime-stability --config config/lime_stability.yaml
```

This experiment compares one baseline LIME seed with ten repeated seeds on 20
probe samples. It produces per-sample metrics, per-seed summaries and a metric
distribution figure under `artifacts/experiment_06_lime_stability/`. These
differences estimate explainer variance; they are not evidence of model drift.
The observed baseline is summarised in
[`docs/LIME_STABILITY_RESULT.md`](docs/LIME_STABILITY_RESULT.md).

Compare reference and candidate model versions using identical LIME seeds, then
evaluate the changes against the fixed-model LIME baseline:

```bash
PYTHONPATH=src python -m xai_regression lime-compare --config config/lime_cross_version.yaml
```

The resulting policy status is advisory. A `REVIEW` result identifies changes
larger than the measured fixed-model variation; it does not by itself prove a
software defect, harmful regression or real-world concept drift.
The first cross-version result and its limitations are documented in
[`docs/LIME_CROSS_VERSION_RESULT.md`](docs/LIME_CROSS_VERSION_RESULT.md).

Combine the completed SHAP and LIME artifacts into one method-aware report:

```bash
PYTHONPATH=src python -m xai_regression report
```

The unified report preserves each explainer's baseline definition and writes
CSV, JSON, Markdown and figure outputs under
`artifacts/experiment_08_unified_report/`.
The current combined result is summarised in
[`docs/UNIFIED_REPORT_RESULT.md`](docs/UNIFIED_REPORT_RESULT.md).

## Important interpretation rule

An explanation change is evidence for investigation, not automatic proof of model regression or concept drift. Changes can also come from sampling, background data, preprocessing, explainer settings or model retraining.
