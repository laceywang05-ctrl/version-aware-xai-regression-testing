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
