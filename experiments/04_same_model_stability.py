from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from shapmonitor import SHAPMonitor


MODEL_PATH = Path(
    "artifacts/experiment_01/models/baseline_v1.joblib"
)

PROBE_PATH = Path("data/fixed_probe.csv")

OUTPUT_DIR = Path(
    "artifacts/experiment_02_same_model"
)

REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "shap_logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load exactly the same model and probe dataset.
model = joblib.load(MODEL_PATH)

probe = pd.read_csv(PROBE_PATH)

X_probe = probe.drop(
    columns=["sample_id", "target"]
)

feature_names = X_probe.columns.tolist()


# 2. Compute one independent explanation run.
def compute_explanations():
    explainer = shap.TreeExplainer(model)

    monitor = SHAPMonitor(
        explainer=explainer,
        data_dir=LOG_DIR,
        sample_rate=1.0,
        model_version="baseline-v1",
        feature_names=feature_names,
        random_seed=42,
    )

    explanation = monitor.compute(X_probe)

    return np.asarray(explanation.values)


run_1 = compute_explanations()
run_2 = compute_explanations()

if run_1.shape != run_2.shape:
    raise RuntimeError(
        "The repeated runs produced different shapes."
    )


# Binary classifiers may return:
# samples × features × classes.
# We consistently explain the positive class, class 1.
if run_1.ndim == 3:
    print(f"Original explanation shape: {run_1.shape}")

    run_1 = run_1[:, :, 1]
    run_2 = run_2[:, :, 1]

    print(f"Selected class-1 shape: {run_1.shape}")

elif run_1.ndim != 2:
    raise RuntimeError(
        f"Unsupported explanation shape: {run_1.shape}."
    )


# 3. Compare the two explanation runs.
dot_products = np.sum(run_1 * run_2, axis=1)

norm_products = (
    np.linalg.norm(run_1, axis=1)
    * np.linalg.norm(run_2, axis=1)
)

cosine_similarities = np.divide(
    dot_products,
    norm_products,
    out=np.ones_like(dot_products),
    where=norm_products != 0,
)

absolute_differences = np.abs(run_1 - run_2)

report = pd.DataFrame(
    {
        "sample_id": probe["sample_id"],
        "cosine_similarity": cosine_similarities,
        "mean_absolute_difference": (
            absolute_differences.mean(axis=1)
        ),
        "maximum_absolute_difference": (
            absolute_differences.max(axis=1)
        ),
    }
)

report.to_csv(
    REPORT_DIR / "same_model_stability.csv",
    index=False,
)


# 4. Print the stability baseline.
print("Same-model stability results")
print("----------------------------")
print(f"Samples: {len(report)}")
print(
    "Mean cosine similarity: "
    f"{report['cosine_similarity'].mean():.12f}"
)
print(
    "Minimum cosine similarity: "
    f"{report['cosine_similarity'].min():.12f}"
)
print(
    "Mean absolute difference: "
    f"{report['mean_absolute_difference'].mean():.12f}"
)
print(
    "Maximum absolute difference: "
    f"{report['maximum_absolute_difference'].max():.12f}"
)
print(f"Report saved to: {REPORT_DIR}")
