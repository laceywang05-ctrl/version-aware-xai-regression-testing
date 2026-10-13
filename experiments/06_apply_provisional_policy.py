"""Apply a two-level provisional explanation-regression policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASELINE_DIR = Path("artifacts/experiment_03_retraining_baseline/reports")
CANDIDATE_DIR = Path("artifacts/experiment_01/reports")
OUTPUT_DIR = Path("artifacts/experiment_04_policy/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

baseline_path = BASELINE_DIR / "per_sample_retraining_comparison.csv"
limits_path = BASELINE_DIR / "provisional_limits.json"
candidate_path = CANDIDATE_DIR / "per_sample_comparison.csv"

for required_path in (baseline_path, limits_path, candidate_path):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing {required_path}. Run experiments 03, 05, and then 06."
        )

baseline = pd.read_csv(baseline_path)
candidate = pd.read_csv(candidate_path)
limits = json.loads(limits_path.read_text(encoding="utf-8"))

metric_rules = [
    {
        "metric": "cosine_similarity",
        "direction": "lower",
        "sample_limit": limits["cosine_similarity_lower_5_percent"],
    },
    {
        "metric": "rank_correlation",
        "direction": "lower",
        "sample_limit": limits["rank_correlation_lower_5_percent"],
    },
    {
        "metric": "top_k_overlap",
        "direction": "lower",
        "sample_limit": limits["top_k_overlap_lower_5_percent"],
    },
    {
        "metric": "direction_flip_rate",
        "direction": "upper",
        "sample_limit": limits["direction_flip_rate_upper_95_percent"],
    },
]


def breach_mask(frame, rule):
    values = frame[rule["metric"]]
    if rule["direction"] == "lower":
        return values < rule["sample_limit"]
    return values > rule["sample_limit"]


policy_rows = []

for rule in metric_rules:
    baseline_with_breach = baseline.assign(breach=breach_mask(baseline, rule))
    benign_run_rates = baseline_with_breach.groupby("seed")["breach"].mean()
    run_level_limit = float(benign_run_rates.quantile(0.95))
    candidate_breach_rate = float(breach_mask(candidate, rule).mean())

    policy_rows.append(
        {
            "metric": rule["metric"],
            "sample_limit_direction": rule["direction"],
            "sample_limit": rule["sample_limit"],
            "benign_mean_breach_rate": float(benign_run_rates.mean()),
            "benign_max_breach_rate": float(benign_run_rates.max()),
            "run_level_limit_95_percent": run_level_limit,
            "candidate_breach_rate": candidate_breach_rate,
            "review_flag": bool(candidate_breach_rate > run_level_limit),
        }
    )

policy = pd.DataFrame(policy_rows)
overall_status = "REVIEW" if policy["review_flag"].any() else "WITHIN_BASELINE"

policy.to_csv(OUTPUT_DIR / "policy_evaluation.csv", index=False)

summary = {
    "status": overall_status,
    "meaning": (
        "At least one metric exceeds variation observed under benign retraining; "
        "investigate rather than automatically rejecting the model."
        if overall_status == "REVIEW"
        else "All monitored metrics are within the provisional benign baseline."
    ),
    "flagged_metrics": policy.loc[policy["review_flag"], "metric"].tolist(),
    "calibration": {
        "sample_level": "Pooled benign-retraining 5th/95th percentiles",
        "run_level": "95th percentile of breach rates across 10 retraining seeds",
    },
    "limitations": [
        "The policy is calibrated on one synthetic dataset and one model family.",
        "Ten retraining seeds provide only a provisional run-level baseline.",
        "A review flag is not proof of harmful regression or concept drift.",
    ],
}

(OUTPUT_DIR / "policy_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("Provisional explanation-regression policy")
print("-----------------------------------------")
print(policy.to_string(index=False))
print(f"\nOverall status: {overall_status}")
print(f"Flagged metrics: {', '.join(summary['flagged_metrics']) or 'none'}")
print(f"Reports saved to: {OUTPUT_DIR}")

