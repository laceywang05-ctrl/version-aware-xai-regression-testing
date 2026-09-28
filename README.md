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

This is the initial project repository. It currently contains the literature-based research scope, proposed architecture and an experiment-configuration draft. Implementation will begin after the initial scope is confirmed with the supervisor.

## Important interpretation rule

An explanation change is evidence for investigation, not automatic proof of model regression or concept drift. Changes can also come from sampling, background data, preprocessing, explainer settings or model retraining.
