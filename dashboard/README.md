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

## Compare uploaded classification models

Open **Compare uploaded models** when the dashboard is running through
`xai_regression serve`. The first upload workflow accepts:

- a trusted baseline classifier named with the `.joblib` extension;
- a trusted candidate classifier in the same format;
- a labelled CSV with no more than 5,000 rows; and
- a JSON manifest describing the label, optional sample ID, ordered feature
  columns, optional class display names, and an optional image filename column;
- optionally, a ZIP containing PNG, JPEG, WebP or GIF sample images.

Download the example manifest from the page and edit it to match the CSV. Both
models must expose `predict()` and accept the same selected features. Optional
`predict_proba()` output is shown as confidence. Uploaded files are processed
in memory and are not persisted by the local server.

For image display, set `image_column` in the manifest and use that CSV column
to reference the exact relative filename inside the ZIP, for example
`digits/sample-1.png`. The comparison table shows a thumbnail for every
matching sample; selecting it opens the full image. Images are display evidence
only in this first iteration, while the model still receives the ordered
feature columns declared in the manifest.

Joblib files can execute Python while loading. Only use model files from a
trusted source. This stage intentionally supports classification only; the
same upload contract can later be extended with regression and explanation
adapters.

When the generated cross-dataset summary is available, the same command adds a
**Cross-dataset overview** containing the completed SHAP/LIME evidence matrix
for Breast Cancer Wisconsin and binary Wine.

When the final split-and-probe robustness summary is available, it also adds a
**Split robustness** view. This page presents each dataset/split result and the
overall `SPLIT_SENSITIVE` conclusion. The default report is
`artifacts/experiment_27_lime_split_probe_robustness/reports/summary.json`.
