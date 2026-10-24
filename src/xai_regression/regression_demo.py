"""Build a self-contained multi-model regression upload demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def build_regression_demo(output_dir: str | Path) -> dict[str, object]:
    """Train four regressors and write files ready for the local upload page."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    labels = dataset.target.to_numpy(dtype=float)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.30, random_state=42
    )
    models = [
        ("linear_regression.joblib", LinearRegression()),
        ("decision_tree.joblib", DecisionTreeRegressor(max_depth=4, random_state=11)),
        ("random_forest.joblib", RandomForestRegressor(n_estimators=80, random_state=23, n_jobs=1)),
        ("gradient_boosting.joblib", GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=37)),
    ]
    metrics = []
    for filename, model in models:
        model.fit(train_x, train_y)
        prediction = np.asarray(model.predict(test_x), dtype=float)
        mae = float(np.mean(np.abs(prediction - test_y)))
        joblib.dump(model, destination / filename)
        metrics.append({"filename": filename, "mae": mae})

    sample_frame = test_x.reset_index(drop=True).copy()
    sample_frame.insert(0, "sample_id", [f"patient-{index:03d}" for index in range(len(sample_frame))])
    sample_frame["target"] = test_y
    sample_frame.to_csv(destination / "samples.csv", index=False)
    manifest = {
        "task": "regression",
        "label_column": "target",
        "id_column": "sample_id",
        "dataset_note": "Scikit-learn diabetes regression demo using a fixed 70/30 train-test split.",
        "feature_columns": list(features.columns),
        "trusted_model_files": True,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": "scikit-learn diabetes",
        "sample_count": len(sample_frame),
        "models": metrics,
        "files": [name for name, _ in models] + ["samples.csv", "manifest.json"],
    }
    summary_path = destination / "README.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": destination,
        "sample_count": len(sample_frame),
        "model_count": len(models),
        "best_mae": min(item["mae"] for item in metrics),
        "summary_path": summary_path,
    }
