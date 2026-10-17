## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-12
- Verification Status: VERIFIED
- Version Label: exp_result_v1

# Initial Experiment Results

## Purpose

The first experiments test whether explanation changes across model versions can
be separated from normal variation. All results use a synthetic tabular binary
classification dataset, a fixed 100-sample probe set, random forests, and
positive-class Tree SHAP values.

## Experiment sequence

1. **Two-version comparison:** compare an accepted baseline model with a candidate
   whose tree count, maximum depth, and random seed changed.
2. **Same-model repeatability:** run the same deterministic explainer twice on the
   same model and samples.
3. **Benign retraining baseline:** retrain the unchanged random-forest
   configuration with ten different random seeds.
4. **Provisional policy:** compare the candidate model's sample-level breach rates
   with the distribution observed across benign retraining runs.

## Main findings

| Result | Value |
|---|---:|
| Same-model cosine similarity | 1.000 |
| Benign-retraining mean prediction agreement | 0.982 |
| Benign-retraining mean cosine similarity | 0.988 |
| Benign-retraining mean rank correlation | 0.885 |
| Benign-retraining mean top-3 overlap | 0.905 |
| Benign-retraining mean direction-flip rate | 0.087 |
| Candidate-version mean cosine similarity | 0.984 |
| Candidate-version mean rank correlation | 0.864 |
| Candidate-version mean top-3 overlap | 0.887 |
| Candidate-version mean direction-flip rate | 0.101 |

The candidate model is not treated as an automatic failure. Its breach rates are
higher than the provisional benign run-level limits for cosine similarity, rank
correlation, and direction-flip rate, so the appropriate status is **REVIEW**.
Top-3 overlap remains within the benign baseline.

## Interpretation

Tree SHAP is exactly repeatable here when the model, data, and explainer settings
are fixed. Stochastic model retraining still creates measurable explanation
variation, so comparing a candidate only against perfect same-model repeatability
would be too strict. The benign-retraining experiment supplies a more realistic
reference distribution.

This is an initial feasibility result, not a universal threshold. The experiment
uses one synthetic dataset, one model family, ten retraining seeds, and one SHAP
configuration. The next validation step should repeat the workflow on a real
tabular dataset and test deliberately controlled model or data changes.

That next step has now been completed and is documented in
[`REAL_DATASET_EXPERIMENT_RESULT.md`](REAL_DATASET_EXPERIMENT_RESULT.md).

