# Contributing

## Development workflow

1. Create a short branch from `main`, for example `feature/shap-runner`.
2. Make one focused change.
3. Add tests once implementation work begins.
4. Run the complete local test suite:

   ```bash
   MPLBACKEND=Agg MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
   PYTHONPATH=src python -m unittest discover -s tests -q
   ```

5. Open a pull request that explains the research or implementation decision.
   GitHub Actions repeats the complete suite on Python 3.11 and 3.13.

## Research records

When adding an experiment, record the model version, dataset version, preprocessing version, probe-data identifier, explainer settings, random seed, metric definition and threshold. Results without this metadata should not be used for cross-version conclusions.

## Interpretation

Do not label an explanation difference as concept drift or regression without supporting evidence. Treat warnings as prompts for investigation.
