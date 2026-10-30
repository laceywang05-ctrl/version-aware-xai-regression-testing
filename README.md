# Version-Aware XAI Regression Testing

The consolidated research questions, evidence, limitations and final prototype
assessment are available in
[`docs/FINAL_EVALUATION_SUMMARY.md`](docs/FINAL_EVALUATION_SUMMARY.md).
Exact files, expected results and click order for a supervisor demonstration
are recorded in [`docs/LIVE_DEMO_GUIDE.md`](docs/LIVE_DEMO_GUIDE.md).

An FYP project for detecting unexpected changes in model explanations across machine-learning model versions.

Every pull request is automatically checked on Python 3.11 and 3.13 using the
complete unit-test suite. The same verification can be run locally with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m unittest discover -s tests -q
```

## Research question

When a model changes from version `v1` to `v2`, prediction accuracy may remain acceptable while the model changes which features it relies on. This project investigates how to identify those changes reliably and distinguish them from normal variation in the explanation method.

## Initial scope and implemented extensions

- Tabular classification
- Two model versions: an accepted baseline and a candidate
- A fixed probe dataset and preprocessing pipeline
- SHAP and LIME explanations
- Same-model repeated runs to estimate natural explainer variance
- Cross-version comparison using explanation-similarity metrics
- Image classification with Integrated Gradients and Grad-CAM
- Uploaded collections of two to five classification or regression models
- Sample-level prediction, confidence and explanation diagnostics

Integrated Gradients and Grad-CAM are now implemented for trusted PyTorch image
models. MNIST Integrated Gradients uses a held-out benign-retraining threshold;
Grad-CAM remains explicitly exploratory because its benign calibration did not
produce a useful confirmatory lower-tail boundary.

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

The image workflow also reports cosine similarity between spatial heatmaps.
Its interpretation remains method- and dataset-specific rather than sharing a
universal threshold with tabular attribution vectors.

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

The repository now contains an end-to-end local comparison system rather than
only the original Tree SHAP proof of concept. Users can upload two to five
trusted models with labelled tabular or image data, rank the full collection,
select a baseline and candidate, inspect aggregate and sample-level changes,
compare explanation maps, and export an auditable report. The automated suite
currently contains 137 passing tests across Python 3.11 and 3.13 CI.

The original proof of concept verifies deterministic same-model repeatability,
estimates normal explanation variation across benign retraining seeds, and
applies a review policy. Initial results are summarised in
[`docs/INITIAL_EXPERIMENT_RESULTS.md`](docs/INITIAL_EXPERIMENT_RESULTS.md).

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

The local server now provides the first regression-upload foundation at
`POST /api/regression/compare`. It accepts two trusted `.joblib` regressors, a
numeric labelled CSV, and a manifest whose `task` is `regression`. The response
contains MAE, RMSE, R², paired error-direction evidence, and per-sample absolute
error changes. Select **Regression** in the dashboard upload form to use a
dedicated metric summary and filterable per-sample error table. Classification
remains the more feature-complete workflow.

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

The server-backed dashboard now includes a **Compare uploaded models** workflow.
It accepts two to five trusted scikit-learn-compatible `.joblib` classifiers or
regressors, one labelled CSV, and an uploaded or browser-generated JSON
manifest. Select any two uploaded models as the baseline and candidate, then
inspect aggregate metrics and the exact samples that improved or regressed.
The selected pair is processed locally in memory and files are not persisted;
changing the pair reuses the dataset and optional images already selected in
the browser. Because loading joblib can execute Python, only trusted model
files should be used.

Before execution, the selection summary shows the task, dataset, model count,
manifest source, and exact V1/V2 filenames. A preflight check then runs every
uploaded model against the shared feature columns and blocks comparison with an
actionable per-file message if a model cannot produce compatible predictions.

The results first rank every uploaded model by accuracy on the shared labelled
sample, display macro F1 for every model, and report pairwise prediction
agreement. Accuracy breaks ranking ties with macro F1 so minority-class
performance remains visible. The detailed section then
compares the selected baseline and candidate at sample level, including image
transitions and compatible explanation maps.

Before scoring, a preflight check tests each model against the manifest-selected
feature columns. The dashboard reports the failing filename and distinguishes
load, prediction-shape, feature/preprocessing, and optional confidence support.

The selected pair also receives per-class diagnostics: sample counts, baseline
and candidate accuracy, and the signed accuracy change for every true class.
This makes class-specific improvements and regressions visible for multiclass
image datasets such as handwritten digits and CIFAR-style labels.

The comparison also reports macro F1 for both selected models and per-class
precision, recall and F1. Macro F1 weights each class equally, making changes
on smaller classes visible even when overall accuracy is dominated by a larger
class.

Misclassification-path diagnostics then show the concrete true-class to
predicted-class error pairs for both versions, including whether each error
count increased or decreased in the candidate model.

Completed comparisons can be exported locally as a JSON audit artifact or a
sample-level CSV. The JSON bundles compatibility checks, the multi-model
overview, and the detailed selected-pair report; the CSV provides predictions,
confidence values, and transition labels for downstream analysis.

The overview and exported JSON also record SHA-256 content fingerprints for
every uploaded model plus the labelled dataset and manifest. These fingerprints
identify the exact compared artifacts independently of their filenames, making
the audit output suitable for version-aware reproduction.

The manifest can either be uploaded as JSON or generated in the dashboard.
After a CSV is selected, the browser detects its header and lets the user choose
the label, optional sample-ID and image columns, plus an optional image shape;
all remaining columns become ordered model features.

An optional image ZIP can be matched to a CSV `image_column`. When present,
the sample-level table displays each image beside its true label, both model
predictions, confidence values, and correctness transition. This supports the
planned MNIST/CIFAR-style error inspection while retaining the tabular upload
path.

Generate a complete offline handwritten-digit demonstration bundle with:

```bash
PYTHONPATH=src python -m xai_regression image-demo
```

The command trains a deliberately limited decision-tree V1, random forest,
extra-trees classifier, and k-nearest-neighbours classifier. It writes all four
trusted model files, a labelled sample CSV, an image ZIP, and the matching
manifest under `artifacts/demo_image_upload/`. Upload the first two model files,
select the other two together under **Additional models**, and use **Compare
uploaded models** to demonstrate model ranking and concrete image-level
improvements and regressions.
This uses scikit-learn's bundled 8×8 handwritten digits so the demo is fully
offline; true MNIST and CIFAR-10 remain the next dataset integrations.

Run the complementary close-version benchmark with:

```bash
PYTHONPATH=src python -m xai_regression digits-close-benchmark
```

This evaluates four related random-forest versions on the same 540-image
held-out test set and reports accuracy, macro F1, pairwise prediction agreement,
discordant correctness counts, and exact McNemar p-values. It is designed to
show subtle model-version changes rather than an intentionally large V1–V2 gap.

Repeat the same versions across 10 deterministic data splits with:

```bash
PYTHONPATH=src python -m xai_regression digits-split-robustness
```

This reports mean accuracy and macro F1 with deterministic bootstrap intervals,
plus paired split-level accuracy changes and win/tie/loss counts. It separates
split sensitivity from the paired-sample evidence on one fixed test set.

Hold the seed-42 data split fixed and vary random-forest training randomness
with:

```bash
PYTHONPATH=src python -m xai_regression digits-training-robustness
```

This second robustness experiment uses 10 training seeds and reports the same
bootstrap intervals and paired win/tie/loss summaries. Comparing both reports
separates uncertainty from data partitioning and stochastic model fitting.

To prepare MNIST, CIFAR-10, or another labelled image dataset supplied as an
NPZ file, use generic `images` and `labels` arrays or the common Keras-style
`x_train`/`y_train` and `x_test`/`y_test` keys, then run:

```bash
PYTHONPATH=src python -m xai_regression prepare-image-data \
  --input path/to/dataset.npz \
  --output-dir artifacts/prepared_image_upload \
  --limit 500
```

The command prefers `x_test`/`y_test` when both Keras-style splits are present,
accepts flat labels or CIFAR-style `(N,1)` labels, and supports grayscale
`(N,H,W)` or channel-last `(N,H,W,C)` arrays. It creates `samples.csv`,
`images.zip`, and `manifest.json`. Upload those files
with two to five trusted `.joblib` classifiers trained on the same flattened
pixel order. RGB images are displayed in the sample table; pixel heatmaps remain
limited to grayscale inputs with an `image_shape` manifest entry.
An optional one-dimensional `class_names` array maps numeric labels to readable
names. If it is absent, filenames containing `cifar10` receive the standard ten
CIFAR-10 names, while MNIST/digits filenames receive readable digit labels.

For a complete four-model demonstration from the same NPZ format, run:

```bash
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input path/to/dataset.npz \
  --output-dir artifacts/image_model_demo \
  --limit 1000
```

This performs a fixed stratified train/test split and trains a dummy baseline,
decision tree, random forest, and extra-trees classifier. The output directory
contains all four trusted `.joblib` files plus the held-out CSV, image ZIP,
manifest, and a JSON summary. These are intentionally lightweight demo
baselines rather than recommended MNIST or CIFAR-10 architectures.
When all four Keras-style split arrays are present, the command instead trains
on `x_train`/`y_train` and evaluates only on `x_test`/`y_test`, preserving the
dataset's official split. The generated upload bundle contains test samples.

The same command also accepts the official CIFAR-10 Python archive without
extracting it first:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_model_demo \
  --limit 1000
```

It reads `data_batch_1` through `data_batch_5`, `test_batch`, and the class
names in `batches.meta`. Only the requested number of training and test rows is
kept in memory, and the archive is never extracted to the project directory.
Use only the official trusted CIFAR-10 archive because the Python distribution
stores its batches as pickle files.

Generate a ready-to-upload four-model regression demonstration with:

```bash
PYTHONPATH=src python -m xai_regression regression-demo
```

This trains linear regression, decision-tree, random-forest and
gradient-boosting models on a fixed split of scikit-learn's bundled diabetes
dataset. The command writes four trusted `.joblib` files, `samples.csv` and
`manifest.json` under `artifacts/demo_regression_upload/`. Select Regression
on the upload page, use two model files as the initial pair, and select the
remaining two under Additional models.

Before a live demonstration, validate any generated classification or
regression bundle with one command:

```bash
PYTHONPATH=src python -m xai_regression demo-check \
  --bundle-dir artifacts/mnist_model_demo
```

The preflight checks that `manifest.json`, `samples.csv`, two to five model
files and any required `images.zip` are present. It then loads every trusted
model, verifies its prediction shape against the manifest-selected feature
columns, reports the dataset dimensions and prints `READY FOR DEMO` only when
the complete bundle is compatible.

When the manifest supplies `image_shape`, each sample row can open a
contribution comparison. Compatible tree estimators use Tree SHAP. Other
classifiers exposing `predict_proba()` and `classes_` automatically use pixel
occlusion, which measures the predicted-class probability change after setting
one pixel feature to zero. The detail view names the method used by each model.
Red supports the displayed class and blue opposes it. Each map is normalized
independently, so spatial patterns can be compared but color strength cannot be
compared directly across models. These values explain fitted model outputs;
they are not causal evidence or human-like attention.

The results page defaults to counting candidate predictions that are wrong at
80% confidence or above. The user can change this inspection threshold directly
on the results page. The sample table can isolate those errors or the narrower
set of high-confidence regressions where V1 was correct and V2 became wrong.
This is a transparent inspection threshold, not a statistically calibrated
release boundary.

The multi-model overview reports top-label expected calibration error (ECE)
for classifiers with probability output, alongside mean confidence and the
confidence-minus-accuracy gap. Lower ECE indicates closer agreement between
stated confidence and observed correctness on the uploaded sample.
Compact reliability charts expose the underlying ten bins, comparing mean
confidence with observed accuracy and showing the number of samples in each
populated range.

For a selected baseline–candidate pair, the dashboard applies a two-sided exact
McNemar test to their discordant correctness outcomes. It reports accuracy
change, discordant sample count, the unadjusted p-value, and whether the result
reaches the displayed 0.05 evidence threshold.

When both versions produce explanation maps, the dashboard calculates their
cosine similarity and initially flags values below 0.80. This boundary is also
adjustable without rerunning inference. A dedicated filter isolates
hidden explanation drift: the predicted class stayed the same even though the
feature-contribution pattern changed substantially.

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

## Handwritten-digit explanation stability

Compare Tree SHAP predicted-class contribution maps for the closely performing
V1, V2 and V3 handwritten-digit model versions:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression digits-explanation-stability
```

The experiment measures cosine similarity on the same 120 test images and
separately flags hidden explanation drift, where predictions agree but
explanations fall below the inspection threshold. Results and limitations are
documented in
[`docs/DIGITS_EXPLANATION_STABILITY_RESULT.md`](docs/DIGITS_EXPLANATION_STABILITY_RESULT.md).

Repeat the explanation comparison across five stratified train-test splits:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression \
  digits-explanation-split-robustness
```

This sensitivity analysis aggregates 1,200 pairwise sample comparisons and
reports whether hidden explanation drift depends on the data partition. Results
are documented in
[`docs/DIGITS_EXPLANATION_SPLIT_ROBUSTNESS_RESULT.md`](docs/DIGITS_EXPLANATION_SPLIT_ROBUSTNESS_RESULT.md).

Measure the hidden-drift review workload at similarity thresholds from 0.70 to
0.90:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression \
  digits-explanation-threshold-sensitivity
```

This analysis treats the threshold as a review-workload choice rather than a
calibrated failure boundary. Results are documented in
[`docs/DIGITS_EXPLANATION_THRESHOLD_SENSITIVITY_RESULT.md`](docs/DIGITS_EXPLANATION_THRESHOLD_SENSITIVITY_RESULT.md).

## MNIST multi-model validation

Run the upload workflow on a real Keras-style MNIST archive:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input mnist.npz \
  --output-dir artifacts/mnist_model_demo \
  --limit 1000
```

The validated run ranks four uploaded classifiers and compares the two closest
tree ensembles on 1,000 official test images. Results, explanation findings and
limitations are documented in
[`docs/MNIST_MULTIMODEL_RESULT.md`](docs/MNIST_MULTIMODEL_RESULT.md).

Calibrate the MNIST explanation threshold using same-configuration Random
Forest retraining:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression mnist-explanation-calibration \
  --input mnist.npz
```

The experiment estimates the lower tail of benign explanation similarity,
checks it on disjoint training seeds and documents the revised review workload in
[`docs/MNIST_EXPLANATION_CALIBRATION_RESULT.md`](docs/MNIST_EXPLANATION_CALIBRATION_RESULT.md).

## CIFAR-10 multi-model validation

Run the same four-model upload workflow directly from the official CIFAR-10
Python archive:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression image-model-demo \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_model_demo \
  --limit 1000
```

The validated run uses 1,000 official training and test images, ranks four
models, compares the two closest ensemble methods, and generates spatial Tree
SHAP maps by aggregating RGB-channel contributions at each pixel. Results,
class-level changes and limitations are documented in
[`docs/CIFAR10_MULTIMODEL_RESULT.md`](docs/CIFAR10_MULTIMODEL_RESULT.md).

Calibrate the RGB explanation threshold with disjoint benign-retraining seeds:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression cifar10-explanation-calibration \
  --input cifar-10-python.tar.gz
```

The command estimates a fifth-percentile similarity boundary from three
same-configuration Random Forest retrainings and checks it on two unseen seeds.
The validated threshold and its effect on the cross-model review workload are
documented in
[`docs/CIFAR10_EXPLANATION_CALIBRATION_RESULT.md`](docs/CIFAR10_EXPLANATION_CALIBRATION_RESULT.md).

The upload comparison applies these calibrated explanation thresholds
automatically when its manifest identifies MNIST (`28×28`) or CIFAR-10 RGB
(`32×32×3`) inputs. The results page displays both the active threshold and its
calibration source; users can still edit it to perform sensitivity analysis.

### Neural-network explanations

Install the optional neural dependency with `python -m pip install -e '.[neural]'`.
Trusted PyTorch image networks can then be wrapped in
`xai_regression.neural_explanations.TorchImageClassifier` and saved as a
`.joblib` upload. The comparison selects Grad-CAM when the network contains a
convolutional layer and falls back to zero-baseline Integrated Gradients for
other differentiable networks. Existing tree and non-neural uploads continue
to use Tree SHAP and pixel occlusion respectively.

Create four upload-ready MNIST neural models—two MLPs explained with Integrated
Gradients and two CNNs explained with Grad-CAM—with:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression neural-image-demo \
  --input mnist.npz \
  --output-dir artifacts/mnist_neural_demo \
  --limit 1000 --test-limit 500 --epochs 3
```

The paired Integrated Gradients and Grad-CAM validation results, including the
method-specific threshold limitation, are documented in
[`docs/MNIST_NEURAL_EXPLANATION_RESULT.md`](docs/MNIST_NEURAL_EXPLANATION_RESULT.md).

Method-specific benign-retraining calibration is documented in
[`docs/MNIST_NEURAL_CALIBRATION_RESULT.md`](docs/MNIST_NEURAL_CALIBRATION_RESULT.md).
The resulting MNIST Integrated Gradients boundary is active; Grad-CAM remains
exploratory because its calibration distribution did not yield a useful
positive lower-tail boundary.

The expanded calibration and the decision to keep Grad-CAM visual-only are
documented in
[`docs/GRADCAM_CALIBRATION_REFINEMENT_RESULT.md`](docs/GRADCAM_CALIBRATION_REFINEMENT_RESULT.md).

Measure how the lightweight CIFAR-10 demo scales from 1,000 to 5,000 training
images while retaining the same official test sample:

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src python -m xai_regression cifar10-training-scale \
  --input cifar-10-python.tar.gz
```

The result separates the effect of additional training data from the known
limitation of using flattened pixels with shallow tree ensembles. See
[`docs/CIFAR10_TRAINING_SCALE_RESULT.md`](docs/CIFAR10_TRAINING_SCALE_RESULT.md).

### Full CIFAR-10 CNN training

Train a six-convolution-layer CNN on the official CIFAR-10 training pool. A
stratified 45,000/5,000 training/validation split selects the final checkpoint;
the complete 10,000-image official test set is evaluated only afterward. The
fixed epoch-5 checkpoint is V1 and the best validation checkpoint is V2. This
is one model family and one random seed, so the V1/V2 comparison measures the
effect of additional training within that run.

Install the optional neural dependency with `python -m pip install -e '.[neural]'`
if it is not already present in the project's virtual environment.

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression cifar10-cnn-train \
  --input cifar-10-python.tar.gz \
  --output-dir artifacts/cifar10_cnn_full_2026-10-29 \
  --epochs 80 --min-epochs 30 --patience 15 --batch-size 256
```

The command uses the Mac GPU when PyTorch MPS is available, or a CUDA GPU/CPU
elsewhere. Each epoch saves a resumable checkpoint, `history.json` and
`progress.json`. If interrupted, repeat the command with `--resume`. The
finished folder contains full-test metrics and predictions, training curves,
and `upload_bundle/` with two trusted `.joblib` models plus a 1,000-image
dashboard sample (`samples.csv`, `images.zip`, `manifest.json`). The website
can compare these two models with Grad-CAM maps; those maps remain exploratory.
The completed protocol, full-test results, paired comparison, and upload-bundle
verification are recorded in
[`docs/CIFAR10_FULL_CNN_RESULT.md`](docs/CIFAR10_FULL_CNN_RESULT.md).
A second full run with a different seed and a paired comparison of the two
validation-selected checkpoints are documented in
[`docs/CIFAR10_CNN_SEED_REPEAT_RESULT.md`](docs/CIFAR10_CNN_SEED_REPEAT_RESULT.md).
