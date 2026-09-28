"""Compare two small attribution vectors."""

from pprint import pprint

from xai_regression.report import compare_attributions


baseline = [0.42, -0.31, 0.15, 0.08, -0.04]
candidate = [0.39, -0.18, 0.21, -0.02, -0.05]

report = compare_attributions(
    baseline,
    candidate,
    baseline_version="model-v1",
    candidate_version="model-v2",
    top_k=3,
)

pprint(report.as_dict())

