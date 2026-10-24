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

### Ready-to-upload image demonstration

Create all five inputs without downloading an external dataset:

```bash
PYTHONPATH=src python -m xai_regression image-demo
```

Then choose the files under `artifacts/demo_image_upload/` as follows:

| Page field | Generated file |
| --- | --- |
| Model V1 | `model_v1.joblib` |
| Model V2 | `model_v2.joblib` |
| Labelled dataset | `samples.csv` |
| Manifest | `manifest.json` |
| Sample images | `images.zip` |

The bundle uses scikit-learn's offline 8×8 handwritten-digit dataset. Its
sample selection emphasizes both V2 improvements and the smaller number of V2
regressions, making the prediction-level page useful for a live demonstration.

The generated manifest includes `"image_shape": [8, 8]`. For compatible tree
models this enables a **Compare maps** action on every sample. The dialog shows
normalized Tree SHAP contributions toward each version's own predicted class:
red cells support that prediction, while blue cells oppose it. Comparing two
maps can suggest which pixel evidence changed between versions. Each map is
normalized independently, so color strength is not directly comparable. The
view does not establish that a highlighted pixel caused the prediction.

When the generated cross-dataset summary is available, the same command adds a
**Cross-dataset overview** containing the completed SHAP/LIME evidence matrix
for Breast Cancer Wisconsin and binary Wine.

When the final split-and-probe robustness summary is available, it also adds a
**Split robustness** view. This page presents each dataset/split result and the
overall `SPLIT_SENSITIVE` conclusion. The default report is
`artifacts/experiment_27_lime_split_probe_robustness/reports/summary.json`.
