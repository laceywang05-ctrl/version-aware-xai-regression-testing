from pathlib import Path

import pandas as pd
import shap
from shapmonitor import SHAPMonitor
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
OUTPUT_DIR = Path("artifacts/shap_logs")


# 1. Generate a small tabular classification dataset.
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=RANDOM_SEED,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)


# 2. Create fixed training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y,
)


# 3. Train the baseline model.
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Baseline accuracy: {accuracy:.3f}")


# 4. Select a fixed probe dataset.
X_probe = X_test.iloc[:100].copy()
probe_predictions = model.predict(X_probe)

print(f"Probe samples: {len(X_probe)}")


# 5. Create the SHAP explainer.
explainer = shap.TreeExplainer(model)


# 6. Configure shap-monitor.
monitor = SHAPMonitor(
    explainer=explainer,
    data_dir=OUTPUT_DIR,
    sample_rate=1.0,
    model_version="baseline-v1",
    feature_names=feature_names,
    random_seed=RANDOM_SEED,
)


# 7. Compute and store SHAP explanations.
monitor.log_batch(X_probe, probe_predictions)

print(f"Model version: {monitor.model_version}")
print(f"SHAP logs saved to: {OUTPUT_DIR}")
print("Smoke test completed successfully.")
