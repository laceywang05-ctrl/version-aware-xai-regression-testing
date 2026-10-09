from pathlib import Path

import joblib
import pandas as pd
import shap
from shapmonitor import SHAPMonitor
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts/experiment_01")
MODEL_DIR = ARTIFACT_DIR / "models"
SHAP_LOG_DIR = ARTIFACT_DIR / "shap_logs"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHAP_LOG_DIR.mkdir(parents=True, exist_ok=True)


# 1. Generate one reproducible dataset.
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=RANDOM_SEED,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)


# 2. Create one fixed train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y,
)


# 3. Save the fixed probe dataset.
X_probe = X_test.iloc[:100].copy()
y_probe = y_test[:100]

probe_to_save = X_probe.copy()
probe_to_save.insert(0, "sample_id", range(len(probe_to_save)))
probe_to_save["target"] = y_probe
probe_to_save.to_csv(DATA_DIR / "fixed_probe.csv", index=False)


# 4. Train baseline model v1.
model_v1 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model_v1.fit(X_train, y_train)


# 5. Train candidate model v2 with a controlled configuration change.
model_v2 = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=99,
)

model_v2.fit(X_train, y_train)


# 6. Evaluate both model versions.
test_predictions_v1 = model_v1.predict(X_test)
test_predictions_v2 = model_v2.predict(X_test)

accuracy_v1 = accuracy_score(y_test, test_predictions_v1)
accuracy_v2 = accuracy_score(y_test, test_predictions_v2)

probe_predictions_v1 = model_v1.predict(X_probe)
probe_predictions_v2 = model_v2.predict(X_probe)

prediction_agreement = (
    probe_predictions_v1 == probe_predictions_v2
).mean()

print(f"Baseline v1 accuracy:  {accuracy_v1:.3f}")
print(f"Candidate v2 accuracy: {accuracy_v2:.3f}")
print(f"Probe prediction agreement: {prediction_agreement:.3f}")


# 7. Save both models.
joblib.dump(model_v1, MODEL_DIR / "baseline_v1.joblib")
joblib.dump(model_v2, MODEL_DIR / "candidate_v2.joblib")


# 8. Generate and store explanations for each version.
def log_model_explanations(model, version, predictions):
    explainer = shap.TreeExplainer(model)

    monitor = SHAPMonitor(
        explainer=explainer,
        data_dir=SHAP_LOG_DIR,
        sample_rate=1.0,
        model_version=version,
        feature_names=feature_names,
        random_seed=RANDOM_SEED,
    )

    monitor.log_batch(X_probe, predictions)
    print(f"Logged explanations for {version}")


log_model_explanations(
    model_v1,
    "baseline-v1",
    probe_predictions_v1,
)

log_model_explanations(
    model_v2,
    "candidate-v2",
    probe_predictions_v2,
)


print(f"Fixed probe saved to: {DATA_DIR / 'fixed_probe.csv'}")
print(f"Models saved to: {MODEL_DIR}")
print(f"SHAP logs saved to: {SHAP_LOG_DIR}")
print("Two-version experiment completed successfully.")
