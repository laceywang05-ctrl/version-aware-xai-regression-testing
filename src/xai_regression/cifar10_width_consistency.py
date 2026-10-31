"""Describe which official CIFAR-10 test images change across matched width runs."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def _predictions(run: Path) -> tuple[dict, list[int], list[int]]:
    summary = json.loads((run / "summary.json").read_text())
    with (run / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10000 or [int(row["official_test_index"]) for row in rows] != list(range(10000)):
        raise ValueError(f"Expected 10,000 ordered official-test rows in {run}.")
    return (
        summary["config"],
        [int(row["actual"]) for row in rows],
        [int(row["v2_prediction"]) for row in rows],
    )


def _overlap(first: set[int], second: set[int]) -> dict:
    union = first | second
    return {"intersection": len(first & second), "union": len(union),
            "jaccard": len(first & second) / len(union) if union else None}


def analyze(pairs: list[tuple[int, Path, Path]]) -> dict:
    if len(pairs) < 2 or len({seed for seed, _, _ in pairs}) != len(pairs):
        raise ValueError("Provide at least two matched pairs with distinct seeds.")
    results: dict[int, dict[str, set[int]]] = {}
    official_labels = None
    archive_hash = None
    for seed, baseline_run, candidate_run in sorted(pairs):
        base_config, base_labels, baseline = _predictions(baseline_run)
        cand_config, cand_labels, candidate = _predictions(candidate_run)
        if base_config["seed"] != seed or cand_config["seed"] != seed:
            raise ValueError(f"Seed mismatch for pair {seed}.")
        if base_config.get("widths", [32, 64, 128]) != [32, 64, 128] or cand_config["widths"] != [40, 80, 160]:
            raise ValueError(f"Unexpected CNN widths for pair {seed}.")
        if base_config["archive_sha256"] != cand_config["archive_sha256"]:
            raise ValueError(f"Different CIFAR-10 archives within pair {seed}.")
        allowed_changes = {"architecture", "widths", "device", "torch_version"}
        for key in (set(base_config) & set(cand_config)) - allowed_changes:
            if base_config[key] != cand_config[key]:
                raise ValueError(f"Training configuration differs at {key} for pair {seed}.")
        with np.load(baseline_run / "split_indices.npz") as first, np.load(candidate_run / "split_indices.npz") as second:
            for part in ("train", "validation"):
                if not np.array_equal(first[part], second[part]):
                    raise ValueError(f"Training split differs at {part} for pair {seed}.")
        if archive_hash is None:
            archive_hash = base_config["archive_sha256"]
        elif archive_hash != base_config["archive_sha256"]:
            raise ValueError("The pairs use different CIFAR-10 archives.")
        if base_labels != cand_labels or (official_labels is not None and base_labels != official_labels):
            raise ValueError("Official-test labels or row order differ between runs.")
        official_labels = base_labels
        results[seed] = {
            "changed": {i for i, (a, b) in enumerate(zip(baseline, candidate)) if a != b},
            "fixed": {i for i, (a, b, label) in enumerate(zip(baseline, candidate, base_labels))
                      if a != label and b == label},
            "regressed": {i for i, (a, b, label) in enumerate(zip(baseline, candidate, base_labels))
                          if a == label and b != label},
        }
    seeds = sorted(results)
    categories = ("changed", "fixed", "regressed")
    shared = {
        category: {
            "per_seed": {str(seed): len(results[seed][category]) for seed in seeds},
            "all_seeds": len(set.intersection(*(results[seed][category] for seed in seeds))),
            "at_least_two": sum(sum(i in results[seed][category] for seed in seeds) >= 2
                                for i in range(10000)),
            "any_seed": len(set.union(*(results[seed][category] for seed in seeds))),
            "pairwise": {f"{a}-{b}": _overlap(results[a][category], results[b][category])
                         for a, b in combinations(seeds, 2)},
        }
        for category in categories
    }
    return {
        "protocol": "Matched CIFAR-10 width changes on identical official-test rows; descriptive seed overlap",
        "seeds": seeds,
        "official_test_samples": 10000,
        "archive_sha256": archive_hash,
        "categories": shared,
        "both_fixed_and_regressed_across_seeds": len(
            set.union(*(results[seed]["fixed"] for seed in seeds)) &
            set.union(*(results[seed]["regressed"] for seed in seeds))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", nargs=3, metavar=("SEED", "V1_RUN", "V2_RUN"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze([(int(seed), Path(first), Path(second)) for seed, first, second in args.pair])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Seeds: {', '.join(map(str, report['seeds']))}")
    for category, values in report["categories"].items():
        print(f"{category}: all={values['all_seeds']}, at least two={values['at_least_two']}, any={values['any_seed']}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
