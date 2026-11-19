"""Build a nonlinear multi-family regression upload demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_friedman_regression_demo(output_dir: str | Path) -> dict[str, object]:
    """Train five regressor families on a reproducible nonlinear dataset."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    values, labels = make_friedman1(
        n_samples=1000, n_features=10, noise=1.0, random_state=42
    )
    feature_names = [f"x{index}" for index in range(1, 11)]
    features = pd.DataFrame(values, columns=feature_names)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.30, random_state=42
    )
    models = [
        ("model_1_linear_regression.joblib", LinearRegression()),
        (
            "model_2_knn_regressor.joblib",
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        ),
        (
            "model_3_svr.joblib",
            make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        ),
        (
            "model_4_random_forest.joblib",
            RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1),
        ),
        (
            "model_5_gradient_boosting.joblib",
            GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
        ),
    ]
    metrics = []
    for filename, model in models:
        model.fit(train_x, train_y)
        prediction = np.asarray(model.predict(test_x), dtype=float)
        residual = prediction - test_y
        metrics.append(
            {
                "filename": filename,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "r2": float(
                    1.0
                    - np.sum(np.square(residual))
                    / np.sum(np.square(test_y - np.mean(test_y)))
                ),
            }
        )
        joblib.dump(model, destination / filename)

    sample_frame = test_x.reset_index(drop=True).copy()
    sample_frame.insert(
        0, "sample_id", [f"friedman-{index:03d}" for index in range(len(sample_frame))]
    )
    sample_frame["target"] = test_y
    sample_frame.to_csv(destination / "samples.csv", index=False)
    manifest = {
        "task": "regression",
        "label_column": "target",
        "id_column": "sample_id",
        "dataset_note": (
            "Friedman #1 nonlinear synthetic regression benchmark: 1,000 rows, "
            "10 features, noise=1.0, and a fixed 70/30 train-test split."
        ),
        "feature_columns": feature_names,
        "trusted_model_files": True,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": "Friedman #1 nonlinear synthetic regression",
        "training_samples": len(train_x),
        "sample_count": len(sample_frame),
        "feature_count": len(feature_names),
        "models": metrics,
        "files": [name for name, _ in models] + ["samples.csv", "manifest.json"],
    }
    summary_path = destination / "README.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": destination,
        "training_samples": len(train_x),
        "sample_count": len(sample_frame),
        "feature_count": len(feature_names),
        "model_count": len(models),
        "best_mae": min(item["mae"] for item in metrics),
        "summary_path": summary_path,
    }
