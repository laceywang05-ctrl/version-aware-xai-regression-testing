"""Reproduce upload compatibility and exploratory LIME ranking checks.

Run with PYTHONPATH=src .venv/bin/python scripts/validate_multimodel_uploads.py.
Only local, trusted demo models are deserialized.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xai_regression.upload_comparison import (
    _tabular_lime_summary,
    compare_upload_payloads,
    summarize_upload_models,
    validate_upload_models,
)


ROOT = Path(__file__).resolve().parents[1]
BREAST = ROOT / "artifacts" / "breast_cancer_model_demo"
OUTPUT = ROOT / "artifacts" / "multimodel_upload_validation.json"


def top_five(result: dict) -> set[str]:
    if not result["available"]:
        raise RuntimeError(result["note"])
    return {item["feature"] for item in result["features"][:5]}


def stability_check() -> dict:
    frame = pd.read_csv(BREAST / "samples.csv")
    manifest = json.loads((BREAST / "manifest.json").read_text())
    features = frame.loc[:, manifest["feature_columns"]]
    filenames = [
        "model_2_tree.joblib",
        "model_3_random_forest.joblib",
        "model_4_logistic_regression.joblib",
    ]
    models = [(name, joblib.load(BREAST / name)) for name in filenames]
    runs = []
    for seed in (17, 42, 73):
        shuffled = features.sample(frac=1, random_state=seed).reset_index(drop=True)
        for row_count in (6, 20, 50):
            rankings = {
                name: top_five(_tabular_lime_summary(model, shuffled, max_rows=row_count))
                for name, model in models
            }
            votes = {feature: sum(feature in values for values in rankings.values()) for feature in features.columns}
            runs.append({
                "seed": seed,
                "rows": row_count,
                "model_top_five": {name: sorted(values) for name, values in rankings.items()},
                "shared_top_five": sorted(
                    votes, key=lambda feature: (-votes[feature], feature)
                )[:5],
            })
    reference = next(run for run in runs if run["seed"] == 17 and run["rows"] == 50)
    for run in runs:
        run["shared_top_five_overlap_with_reference"] = len(
            set(run["shared_top_five"]) & set(reference["shared_top_five"])
        )
        run["per_model_top_five_overlap_with_reference"] = {
            name: len(set(values) & set(reference["model_top_five"][name]))
            for name, values in run["model_top_five"].items()
        }
    return {"dataset_rows": len(features), "features": features.shape[1], "runs": runs}


def wine_check() -> dict:
    wine = load_wine(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, stratify=wine.target, random_state=42
    )
    trained = [
        ("dummy.joblib", DummyClassifier(strategy="most_frequent")),
        ("tree.joblib", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("forest.joblib", RandomForestClassifier(n_estimators=40, random_state=42)),
        ("logistic.joblib", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ]
    payloads = []
    for name, model in trained:
        model.fit(x_train, y_train)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        payloads.append((name, buffer.getvalue()))
    frame = x_test.reset_index(drop=True).assign(target=y_test.reset_index(drop=True))
    dataset = ("wine_test.csv", frame.to_csv(index=False).encode())
    manifest = json.dumps({
        "task": "classification", "label_column": "target",
        "feature_columns": list(wine.feature_names), "trusted_model_files": True,
    }).encode()
    validation = validate_upload_models(models=payloads, dataset=dataset, manifest=manifest)
    overview = summarize_upload_models(models=payloads, dataset=dataset, manifest=manifest)
    pair = compare_upload_payloads(
        model_v1=payloads[1], model_v2=payloads[2], dataset=dataset, manifest=manifest
    )
    return {
        "train_rows": len(x_train), "test_rows": len(x_test), "classes": 3,
        "features": x_test.shape[1], "preflight_ready": validation["ready"],
        "model_checks": validation["models"],
        "accuracy": {item["filename"]: item["accuracy"] for item in overview["models"]},
        "class_rows": len(overview["class_comparison"]),
        "pair_accuracy": [pair["summary"]["v1_accuracy"], pair["summary"]["v2_accuracy"]],
        "consensus_model_count": overview["explanation_consensus"]["model_count"],
    }


def main() -> None:
    result = {"wine": wine_check(), "lime_stability": stability_check()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "wine": {key: value for key, value in result["wine"].items() if key != "model_checks"},
        "stability_runs": len(result["lime_stability"]["runs"]),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
