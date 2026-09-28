# Proposed Architecture

## Components

### Version registry

Records identifiers and metadata for models, datasets, preprocessing pipelines, probe data and explainer configurations.

### Explanation engine

Loads a registered model and generates SHAP or LIME explanation artifacts for a fixed probe dataset.

### Regression-test engine

Compares baseline and candidate predictions, explanation vectors, rankings and contribution directions.

### Policy engine

Applies calibrated thresholds. A warning requests investigation rather than declaring that concept drift or model failure has occurred.

### Report layer

Stores per-sample metrics, aggregate summaries, failed policies and the version metadata required to reproduce the run.

## Reproducibility requirements

Each run should record:

- model and code revision;
- training-data and probe-data identifiers;
- preprocessing version;
- explainer name and version;
- background dataset for SHAP;
- sample count, kernel settings and random seed for LIME;
- metric definitions and policy thresholds.

## Evaluation sequence

1. Repeat explanations for the accepted model to estimate natural variation.
2. Run the accepted and candidate models on the same probe examples.
3. Separate prediction changes from explanation changes.
4. Compare explanation artifacts with method-appropriate metrics.
5. Inspect warnings and trace them back to samples, features and registered versions.

