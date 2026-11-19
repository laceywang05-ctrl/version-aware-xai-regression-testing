"""Build a precomputed, read-only dashboard demo from the local CIFAR-10 bundle."""

from pathlib import Path
import json

from xai_regression.upload_comparison import compare_upload_payloads, summarize_upload_models


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/cifar10_full_test_upload"
MODELS = ROOT / "artifacts/cifar10_cnn_full_2026-10-29/upload_bundle"
OUTPUT = ROOT / "dashboard/data/cifar10-10000-demo.json"


def file_pair(path):
    return path.name, path.read_bytes()


def main():
    models = [
        file_pair(MODELS / "cifar10_cnn_v1_epoch05.joblib"),
        file_pair(MODELS / "cifar10_cnn_v2_best.joblib"),
    ]
    dataset = file_pair(DATA / "samples.csv")
    manifest = (DATA / "manifest.json").read_bytes()
    overview = summarize_upload_models(models=models, dataset=dataset, manifest=manifest)
    comparison = compare_upload_payloads(
        model_v1=models[0], model_v2=models[1], dataset=dataset,
        manifest=manifest, images=file_pair(DATA / "images.zip"),
    )
    OUTPUT.write_text(json.dumps({"overview": overview, "comparison": comparison}), encoding="utf-8")
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1_000_000:.1f} MB)")
    print(comparison["summary"]["v1_accuracy"], comparison["summary"]["v2_accuracy"])


if __name__ == "__main__":
    main()
