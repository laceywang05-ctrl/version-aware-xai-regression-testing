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
├── dashboard/              # Local report dashboard
├── src/xai_regression/     # Experiment, policy and reporting package
├── tests/                  # Unit tests
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

## Local dashboard

The repository includes a dependency-free dashboard prototype for presenting
the unified report. Start it from the repository root:

```bash
python -m http.server 8000 --directory dashboard
```

Open `http://localhost:8000` and optionally load a newly generated unified
`summary.json`. The browser processes the selected file locally; it is not
uploaded to an external service.

## End-to-end workflow

Run the SHAP validation, LIME cross-version comparison and unified report in
one command:

```bash
PYTHONPATH=src python -m xai_regression workflow
```

To start the dashboard immediately after the workflow completes:

```bash
PYTHONPATH=src python -m xai_regression workflow --serve
```

The dashboard then loads the newly generated report automatically. Press
`Control+C` to stop the local server. To reopen an existing report without
rerunning the experiments, use:

```bash
PYTHONPATH=src python -m xai_regression serve
```

The server-backed dashboard now includes a **Compare uploaded models** MVP.
It accepts two to five trusted scikit-learn-compatible `.joblib` classifiers,
one labelled CSV, and a JSON manifest. Select any two uploaded models as the
baseline and candidate, then inspect overall accuracy and the exact samples
that changed from correct to wrong or wrong to correct. The selected pair is
processed locally in memory and files are not persisted; changing the pair
reuses the dataset and images already selected in the browser. Because loading
joblib can execute Python, only trusted model files should be used. The current
upload workflow is intentionally limited to classification; regression remains
a future extension.

An optional image ZIP can be matched to a CSV `image_column`. When present,
the sample-level table displays each image beside its true label, both model
predictions, confidence values, and correctness transition. This supports the
planned MNIST/CIFAR-style error inspection while retaining the tabular upload
path.

Generate a complete offline handwritten-digit demonstration bundle with:

```bash
PYTHONPATH=src python -m xai_regression image-demo
```

The command trains a deliberately limited decision-tree V1 and a stronger
random-forest V2, then writes both trusted model files, a labelled sample CSV,
an image ZIP, and the matching manifest under
`artifacts/demo_image_upload/`. Upload those five files on **Compare uploaded
models** to demonstrate concrete image-level improvements and regressions.
This uses scikit-learn's bundled 8×8 handwritten digits so the demo is fully
offline; true MNIST and CIFAR-10 remain the next dataset integrations.

When the manifest supplies `image_shape`, each sample row can open a
contribution comparison. Compatible tree estimators use Tree SHAP. Other
classifiers exposing `predict_proba()` and `classes_` automatically use pixel
occlusion, which measures the predicted-class probability change after setting
one pixel feature to zero. The detail view names the method used by each model.
Red supports the displayed class and blue opposes it. Each map is normalized
independently, so spatial patterns can be compared but color strength cannot be
compared directly across models. These values explain fitted model outputs;
they are not causal evidence or human-like attention.

When the held-out breast-cancer and wine benchmark summaries are present, the
same command also exposes a **Benchmark validation** view. It compares four
model-version scenarios across 30 held-out seeds, shows Wilson 95% confidence
intervals, and reports Bonferroni-adjusted paired-control results. A benchmark
summary can also be supplied explicitly by repeating `--benchmark-report`.

Build a single cross-dataset evidence matrix from the completed SHAP and LIME
experiments:

```bash
PYTHONPATH=src python -m xai_regression portfolio
```

This writes JSON, CSV and Markdown reports under
`artifacts/experiment_14_cross_dataset_summary/`. After it has been generated,
`PYTHONPATH=src python -m xai_regression serve` automatically adds a
**Cross-dataset overview** page to the dashboard. The page preserves the
different meanings of SHAP review rates and LIME breach rates instead of
combining them into one score. Results and limitations are documented in
[`docs/CROSS_DATASET_SUMMARY_RESULT.md`](docs/CROSS_DATASET_SUMMARY_RESULT.md).

When the final split-and-probe robustness summary is present, the same command
also adds a **Split robustness** page. It shows all six held-out dataset/split
comparisons and makes the `SPLIT_SENSITIVE` limitation visible instead of
presenting only the successful fixed-split results. Supply a different summary
with `--robustness-report PATH` when needed. The final experiment is documented
in
[`docs/LIME_SPLIT_PROBE_ROBUSTNESS_RESULT.md`](docs/LIME_SPLIT_PROBE_ROBUSTNESS_RESULT.md).

## Second model-family validation

Repeat the held-out Tree SHAP benchmark with gradient boosting on both datasets:

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/gradient_boosting_breast_calibration.yaml \
  --scenarios config/gradient_boosting_breast_scenarios.yaml

PYTHONPATH=src python -m xai_regression benchmark \
  --config config/gradient_boosting_wine_calibration.yaml \
  --scenarios config/gradient_boosting_wine_scenarios.yaml
```

This uses 30 calibration seeds and 30 disjoint evaluation seeds per scenario.
The observed cross-family result and its limitations are documented in
[`docs/GRADIENT_BOOSTING_VALIDATION_RESULT.md`](docs/GRADIENT_BOOSTING_VALIDATION_RESULT.md).

Combine the two Random Forest and two Gradient Boosting benchmark reports into
one scenario-level generalisation matrix:

```bash
PYTHONPATH=src python -m xai_regression model-family-summary
```

The combined result identifies which controlled changes separate from benign
retraining in every dataset-model cell. See
[`docs/MODEL_FAMILY_SUMMARY_RESULT.md`](docs/MODEL_FAMILY_SUMMARY_RESULT.md).

Retest the preselected combined-change scenario on data splits 17 and 73 for
both datasets and both model families:

```bash
PYTHONPATH=src python -m xai_regression shap-split-robustness
```

This produces eight additional held-out comparisons and records whether the
cross-model result depends on the original split 42. Results are documented in
[`docs/SHAP_SPLIT_ROBUSTNESS_RESULT.md`](docs/SHAP_SPLIT_ROBUSTNESS_RESULT.md).

Compare the ordinary Tree SHAP run rule with a Bonferroni-adjusted rule across
the eight split-robustness cells:

```bash
PYTHONPATH=src python -m xai_regression shap-multiplicity
```

The adjusted rule reduces benign-control warnings but also loses one held-out
separation, motivating separate advisory and high-confidence review levels.
See
[`docs/SHAP_MULTIPLICITY_POLICY_RESULT.md`](docs/SHAP_MULTIPLICITY_POLICY_RESULT.md).

## Important interpretation rule

An explanation change is evidence for investigation, not automatic proof of model regression or concept drift. Changes can also come from sampling, background data, preprocessing, explainer settings or model retraining.

Run the LIME stability and cross-version checks on the binary Wine dataset:

```bash
PYTHONPATH=src python -m xai_regression lime-stability --config config/wine_lime_stability.yaml
PYTHONPATH=src python -m xai_regression lime-compare --config config/wine_lime_cross_version.yaml
```

The design and observed results are documented in
[`docs/WINE_LIME_VALIDATION_RESULT.md`](docs/WINE_LIME_VALIDATION_RESULT.md).

Validate the LIME policy using 30 calibration seeds and a disjoint set of 30
held-out evaluation seeds on both datasets:

```bash
PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/lime_heldout.yaml
PYTHONPATH=src python -m xai_regression lime-heldout \
  --config config/wine_lime_heldout.yaml
```

The experiment reports held-out fixed-model false-positive behaviour,
candidate review rates, Wilson 95% confidence intervals and an exact paired
McNemar comparison. Results and limitations are documented in
[`docs/LIME_HELDOUT_VALIDATION_RESULT.md`](docs/LIME_HELDOUT_VALIDATION_RESULT.md).

Compare the original per-metric run limits with a Bonferroni-adjusted policy
that accounts for simultaneously monitoring four explanation metrics:

```bash
PYTHONPATH=src python -m xai_regression lime-multiplicity
```

This analysis reuses the same disjoint calibration and held-out samples so the
two policy rules are compared without rerunning LIME or changing the model.
Results and limitations are documented in
[`docs/LIME_MULTIPLICITY_POLICY_RESULT.md`](docs/LIME_MULTIPLICITY_POLICY_RESULT.md).

Confirm the multiplicity-aware rule on a new evaluation-seed set that is
disjoint from both calibration and the earlier held-out analysis:

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

The independent-seed confirmation is documented in
[`docs/LIME_CONFIRMATORY_POLICY_RESULT.md`](docs/LIME_CONFIRMATORY_POLICY_RESULT.md).

Run the final robustness check across three train/test splits and nearly
disjoint fixed-probe subsets:

```bash
PYTHONPATH=src python -m xai_regression lime-split-robustness
```

Each split recalibrates and evaluates the same Bonferroni-adjusted policy. The
experiment reports per-split evidence rather than treating overlapping dataset
splits as independent replicates. Results are documented in
[`docs/LIME_SPLIT_PROBE_ROBUSTNESS_RESULT.md`](docs/LIME_SPLIT_PROBE_ROBUSTNESS_RESULT.md).

## Controlled version benchmark

Evaluate the calibrated SHAP policy across several controlled candidate-model
configurations:

```bash
PYTHONPATH=src python -m xai_regression benchmark
```

The scenario file includes a seed-only control and three model-configuration
changes, each repeated across 30 candidate seeds. The scenario summary reports
Wilson 95% confidence intervals for the review rate. Results are written under
`artifacts/experiment_09_version_benchmark/`.
Warnings on the control scenario estimate false-positive behaviour; warnings on
the other scenarios measure sensitivity to model changes, not whether those
changes are harmful. The preliminary findings and limitations are documented in
[`docs/VERSION_BENCHMARK_RESULT.md`](docs/VERSION_BENCHMARK_RESULT.md).

Run the follow-up benchmark with 30 calibration seeds and a disjoint set of 30
held-out candidate seeds:

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/benchmark_calibration.yaml \
  --scenarios config/version_benchmark_heldout.yaml
```

This diagnostic keeps the quantile policy unchanged while testing whether the
earlier control warnings were caused by a small calibration set.
The verified results and interpretation are documented in
[`docs/HELDOUT_POLICY_VALIDATION_RESULT.md`](docs/HELDOUT_POLICY_VALIDATION_RESULT.md).

## Second-dataset validation

Run the same held-out benchmark design on a binary subset of the scikit-learn
Wine dataset:

```bash
PYTHONPATH=src python -m xai_regression benchmark \
  --config config/wine_benchmark_calibration.yaml \
  --scenarios config/wine_benchmark_heldout.yaml
```

The results are documented in
[`docs/SECOND_DATASET_VALIDATION_RESULT.md`](docs/SECOND_DATASET_VALIDATION_RESULT.md).
