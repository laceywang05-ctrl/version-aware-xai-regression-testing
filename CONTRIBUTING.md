# Contributing

## Development workflow

1. Create a short branch from `main`, for example `feature/shap-runner`.
2. Make one focused change.
3. Add tests once implementation work begins.
4. Open a pull request that explains the research or implementation decision.

## Research records

When adding an experiment, record the model version, dataset version, preprocessing version, probe-data identifier, explainer settings, random seed, metric definition and threshold. Results without this metadata should not be used for cross-version conclusions.

## Interpretation

Do not label an explanation difference as concept drift or regression without supporting evidence. Treat warnings as prompts for investigation.
