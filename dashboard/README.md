# Dashboard prototype

This local static dashboard presents the unified SHAP and LIME report without
uploading experiment data to an external service.

From the repository root, run:

```bash
python -m http.server 8000 --directory dashboard
```

Then open <http://localhost:8000>. The page initially uses the bundled example
report. Use **Load report JSON** to select
`artifacts/experiment_08_unified_report/reports/summary.json` after generating a
new report.

Opening `index.html` directly is not recommended because browsers may block the
example JSON request for local files.

To serve the latest generated unified report automatically, run:

```bash
PYTHONPATH=src python -m xai_regression serve
```

The local server exposes only the selected report at a fixed read-only route.
It binds to `127.0.0.1` by default and does not upload report data externally.

When the generated cross-dataset summary is available, the same command adds a
**Cross-dataset overview** containing the completed SHAP/LIME evidence matrix
for Breast Cancer Wisconsin and binary Wine.

When the final split-and-probe robustness summary is available, it also adds a
**Split robustness** view. This page presents each dataset/split result and the
overall `SPLIT_SENSITIVE` conclusion. The default report is
`artifacts/experiment_27_lime_split_probe_robustness/reports/summary.json`.
