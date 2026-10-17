# Unified SHAP and LIME Report Result

## Overall status

**REVIEW** — both explanation methods flag cosine similarity for further
inspection. This is an advisory result, not proof of a software defect, harmful
model regression or real-world concept drift.

## Method summary

| Method | Natural-variation baseline | Probe samples | Prediction agreement | Flagged metric |
|---|---|---:|---:|---|
| SHAP | Benign model retraining | 100 | 0.990 | Cosine similarity |
| LIME | Fixed-model explainer randomness | 20 | 0.950 | Cosine similarity |

## Policy comparison

| Method | Metric | Baseline mean | Candidate mean | Breach rate | Run-level limit | Review |
|---|---|---:|---:|---:|---:|---|
| SHAP | Cosine similarity | 0.9779 | 0.9719 | 0.1000 | 0.0920 | Yes |
| SHAP | Rank correlation | 0.9158 | 0.9075 | 0.0500 | 0.0710 | No |
| SHAP | Top-5 overlap | 0.9464 | 0.9620 | 0.0300 | 0.0510 | No |
| SHAP | Direction-flip rate | 0.0920 | 0.1000 | 0.0100 | 0.0300 | No |
| LIME | Cosine similarity | 0.8494 | 0.8072 | 0.1150 | 0.0775 | Yes |
| LIME | Rank correlation | 0.5843 | 0.5540 | 0.0550 | 0.1000 | No |
| LIME | Top-5 overlap | 0.7680 | 0.7210 | 0.0500 | 0.0775 | No |
| LIME | Direction-flip rate | 0.2537 | 0.2697 | 0.0500 | 0.1000 | No |

## Interpretation boundary

The methods use different baselines: SHAP is calibrated against variation from
benign model retraining, while LIME is calibrated against stochastic variation
from repeated explanations of a fixed model. Their raw thresholds are therefore
not interchangeable and should not be used to rank one method as universally
more stable than the other.
