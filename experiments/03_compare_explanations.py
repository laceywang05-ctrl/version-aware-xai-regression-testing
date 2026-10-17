from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOG_DIR = Path("artifacts/experiment_01/shap_logs")
REPORT_DIR = Path("artifacts/experiment_01/reports")
FIGURE_DIR = Path("artifacts/experiment_01/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 3
EPSILON = 1e-12


# 1. Load every Parquet explanation log.
files = sorted(LOG_DIR.rglob("*.parquet"))

if len(files) < 2:
    raise RuntimeError("At least two explanation log files are required.")

frames = [pd.read_parquet(file) for file in files]
logs = pd.concat(frames, ignore_index=True)

print("Loaded model versions:")
print(logs["model_version"].value_counts())


# 2. Separate the two model versions.
baseline = (
    logs[logs["model_version"] == "baseline-v1"]
    .reset_index(drop=True)
)

candidate = (
    logs[logs["model_version"] == "candidate-v2"]
    .reset_index(drop=True)
)

if len(baseline) != len(candidate):
    raise RuntimeError("The two versions contain different sample counts.")


# 3. Identify SHAP and original-feature columns.
shap_columns = [
    column
    for column in logs.columns
    if column.startswith("shap_")
]

feature_columns = [
    column
    for column in logs.columns
    if column.startswith("feat_")
]

feature_names = [
    column.removeprefix("shap_")
    for column in shap_columns
]


# 4. Confirm that both versions used the same probe samples.
same_probe = np.allclose(
    baseline[feature_columns].to_numpy(),
    candidate[feature_columns].to_numpy(),
)

if not same_probe:
    raise RuntimeError(
        "The baseline and candidate logs do not contain the same probe data."
    )

print("Fixed probe verification: passed")


baseline_values = baseline[shap_columns].to_numpy()
candidate_values = candidate[shap_columns].to_numpy()


# 5. Define per-sample comparison metrics.
def cosine_similarity(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)

    if denominator == 0:
        return 1.0 if np.allclose(left, right) else 0.0

    return float(np.dot(left, right) / denominator)


def rank_correlation(left, right):
    left_ranks = pd.Series(np.abs(left)).rank(method="average")
    right_ranks = pd.Series(np.abs(right)).rank(method="average")

    correlation = left_ranks.corr(right_ranks)

    if pd.isna(correlation):
        return 1.0 if np.allclose(left, right) else 0.0

    return float(correlation)


def top_k_overlap(left, right, k):
    left_top = set(np.argsort(np.abs(left))[-k:])
    right_top = set(np.argsort(np.abs(right))[-k:])

    return len(left_top.intersection(right_top)) / k


def direction_flip_rate(left, right):
    active = (
        (np.abs(left) > EPSILON)
        | (np.abs(right) > EPSILON)
    )

    if not active.any():
        return 0.0

    flips = (
        np.sign(left[active])
        != np.sign(right[active])
    )

    return float(flips.mean())


# 6. Compare every matched probe sample.
per_sample_rows = []

for sample_id, (left, right) in enumerate(
    zip(baseline_values, candidate_values)
):
    per_sample_rows.append(
        {
            "sample_id": sample_id,
            "prediction_v1": baseline.loc[sample_id, "prediction"],
            "prediction_v2": candidate.loc[sample_id, "prediction"],
            "prediction_changed": (
                baseline.loc[sample_id, "prediction"]
                != candidate.loc[sample_id, "prediction"]
            ),
            "cosine_similarity": cosine_similarity(left, right),
            "rank_correlation": rank_correlation(left, right),
            "top_k_overlap": top_k_overlap(left, right, TOP_K),
            "direction_flip_rate": direction_flip_rate(left, right),
        }
    )

per_sample = pd.DataFrame(per_sample_rows)

per_sample.to_csv(
    REPORT_DIR / "per_sample_comparison.csv",
    index=False,
)


# 7. Create a global feature comparison.
mean_abs_v1 = np.abs(baseline_values).mean(axis=0)
mean_abs_v2 = np.abs(candidate_values).mean(axis=0)

mean_signed_v1 = baseline_values.mean(axis=0)
mean_signed_v2 = candidate_values.mean(axis=0)

global_comparison = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_v1": mean_abs_v1,
        "mean_abs_v2": mean_abs_v2,
        "delta_mean_abs": mean_abs_v2 - mean_abs_v1,
        "mean_signed_v1": mean_signed_v1,
        "mean_signed_v2": mean_signed_v2,
    }
)

global_comparison["rank_v1"] = (
    global_comparison["mean_abs_v1"]
    .rank(ascending=False, method="min")
)

global_comparison["rank_v2"] = (
    global_comparison["mean_abs_v2"]
    .rank(ascending=False, method="min")
)

global_comparison["rank_change"] = (
    global_comparison["rank_v2"]
    - global_comparison["rank_v1"]
)

global_comparison["sign_flip"] = (
    np.sign(global_comparison["mean_signed_v1"])
    != np.sign(global_comparison["mean_signed_v2"])
)

global_comparison = global_comparison.sort_values("rank_v1")

global_comparison.to_csv(
    REPORT_DIR / "global_feature_comparison.csv",
    index=False,
)


# 8. Produce a simple comparison figure.
x = np.arange(len(feature_names))
width = 0.4

plt.figure(figsize=(11, 5))
plt.bar(
    x - width / 2,
    mean_abs_v1,
    width,
    label="baseline-v1",
)
plt.bar(
    x + width / 2,
    mean_abs_v2,
    width,
    label="candidate-v2",
)

plt.xticks(x, feature_names, rotation=45, ha="right")
plt.ylabel("Mean absolute SHAP value")
plt.title("Global SHAP Feature Importance by Model Version")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "global_feature_importance.png",
    dpi=160,
)

plt.close()


# 9. Print the experiment summary.
global_rank_correlation = (
    global_comparison["rank_v1"]
    .corr(global_comparison["rank_v2"])
)

print("\nComparison summary")
print("------------------")
print(f"Matched samples: {len(per_sample)}")
print(
    "Prediction agreement: "
    f"{1 - per_sample['prediction_changed'].mean():.3f}"
)
print(
    "Mean cosine similarity: "
    f"{per_sample['cosine_similarity'].mean():.3f}"
)
print(
    "Mean rank correlation: "
    f"{per_sample['rank_correlation'].mean():.3f}"
)
print(
    f"Mean top-{TOP_K} overlap: "
    f"{per_sample['top_k_overlap'].mean():.3f}"
)
print(
    "Mean direction-flip rate: "
    f"{per_sample['direction_flip_rate'].mean():.3f}"
)
print(
    "Global feature-rank correlation: "
    f"{global_rank_correlation:.3f}"
)
print(
    "Global contribution sign flips: "
    f"{int(global_comparison['sign_flip'].sum())}"
)

print(f"\nReports saved to: {REPORT_DIR}")
print(f"Figure saved to: {FIGURE_DIR}")
