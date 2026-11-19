"""Build a self-contained four-model Wine classification upload demo."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_wine_classification_demo(output_dir: str | Path) -> dict[str, object]:
    """Train four classifiers and write an upload-ready multiclass bundle."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = load_wine(as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.30,
        stratify=dataset.target,
        random_state=42,
    )
    models = [
        ("model_1_dummy.joblib", DummyClassifier(strategy="most_frequent")),
        ("model_2_tree.joblib", DecisionTreeClassifier(max_depth=4, random_state=42)),
        (
            "model_3_random_forest.joblib",
            RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=1),
        ),
        (
            "model_4_logistic_regression.joblib",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
    ]
    metrics = []
    train_values = train_x.to_numpy(dtype=float)
    test_values = test_x.to_numpy(dtype=float)
    for filename, model in models:
        model.fit(train_values, train_y.to_numpy())
        prediction = np.asarray(model.predict(test_values))
        accuracy = float(np.mean(prediction == test_y.to_numpy()))
        macro_f1 = float(f1_score(test_y, prediction, average="macro"))
        joblib.dump(model, destination / filename)
        metrics.append(
            {
                "filename": filename,
                "model_type": type(model).__name__,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

    samples = test_x.reset_index(drop=True).copy()
    samples.insert(0, "sample_id", [f"wine-{index:03d}" for index in range(len(samples))])
    samples["target"] = test_y.reset_index(drop=True)
    samples.to_csv(destination / "samples.csv", index=False)

    class_names = {str(index): str(name) for index, name in enumerate(dataset.target_names)}
    manifest = {
        "task": "classification",
        "label_column": "target",
        "id_column": "sample_id",
        "image_column": None,
        "feature_columns": list(dataset.feature_names),
        "class_names": class_names,
        "dataset_note": (
            "Scikit-learn Wine three-class test split: 54 labelled rows and "
            "13 numeric features from a fixed stratified 70/30 split."
        ),
        "trusted_model_files": True,
        "explanation_method": "auto",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "dataset": "scikit-learn Wine",
        "total_samples": len(dataset.data),
        "training_samples": len(train_x),
        "test_samples": len(test_x),
        "feature_count": len(dataset.feature_names),
        "class_count": len(dataset.target_names),
        "class_names": class_names,
        "models": metrics,
        "upload": {
            "task": "Classification",
            "initial_models": [models[0][0], models[1][0]],
            "additional_models": [models[2][0], models[3][0]],
            "dataset": "samples.csv",
            "manifest": "manifest.json",
        },
    }
    summary_path = destination / "README.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": destination,
        "model_count": len(models),
        "training_samples": len(train_x),
        "sample_count": len(test_x),
        "feature_count": len(dataset.feature_names),
        "class_count": len(dataset.target_names),
        "best_accuracy": max(item["accuracy"] for item in metrics),
        "summary_path": summary_path,
    }
