"""Build a five-model Iris classification upload demo."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_iris_classification_demo(output_dir: str | Path) -> dict[str, object]:
    """Train five classifier families and write an upload-ready Iris bundle."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = load_iris(as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.30,
        stratify=dataset.target,
        random_state=42,
    )
    models = [
        (
            "model_1_logistic_regression.joblib",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        (
            "model_2_knn.joblib",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ),
        (
            "model_3_svm.joblib",
            make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=42), cv=3, ensemble=False
                ),
            ),
        ),
        ("model_4_gaussian_nb.joblib", GaussianNB()),
        (
            "model_5_extra_trees.joblib",
            ExtraTreesClassifier(n_estimators=60, random_state=42, n_jobs=1),
        ),
    ]
    train_values = train_x.to_numpy(dtype=float)
    test_values = test_x.to_numpy(dtype=float)
    target_values = train_y.to_numpy()
    test_targets = test_y.to_numpy()
    metrics = []
    for filename, model in models:
        model.fit(train_values, target_values)
        prediction = np.asarray(model.predict(test_values))
        joblib.dump(model, destination / filename)
        metrics.append(
            {
                "filename": filename,
                "model_type": type(model).__name__,
                "accuracy": float(np.mean(prediction == test_targets)),
                "macro_f1": float(f1_score(test_targets, prediction, average="macro")),
            }
        )

    samples = test_x.reset_index(drop=True).copy()
    samples.insert(0, "sample_id", [f"iris-{index:03d}" for index in range(len(samples))])
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
            "Scikit-learn Iris three-class test split: 45 labelled rows and "
            "4 numeric features from a fixed stratified 70/30 split."
        ),
        "trusted_model_files": True,
        "explanation_method": "auto",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": "scikit-learn Iris",
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
            "additional_models": [item[0] for item in models[2:]],
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
