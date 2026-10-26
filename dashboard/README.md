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

The server also exposes an initial `POST /api/regression/compare` endpoint for
two trusted `.joblib` regressors. It returns MAE, RMSE, R², paired improvement
and worsening counts, an exact paired sign-test result, and per-sample absolute
error changes. The upload page exposes this workflow through the **Prediction
task** selector. Regression mode keeps the two required model files, dataset
and manifest, while hiding classification-only images, additional model ranking
and pixel explanations. Its result panel reports aggregate metrics, paired
outcome counts and a filterable sample error table.

## Compare uploaded classification models

Open **Compare uploaded models** when the dashboard is running through
`xai_regression serve`. The upload workflow accepts:

- two required trusted classifiers plus up to three additional classifiers,
  each named with the `.joblib` extension;
- a labelled CSV with no more than 5,000 rows; and
- a JSON manifest describing the label, optional sample ID, ordered feature
  columns, optional class display names, and an optional image filename column;
- optionally, a ZIP containing PNG, JPEG, WebP or GIF sample images.

Download the example manifest from the page and edit it to match the CSV. Both
models must expose `predict()` and accept the same selected features. Optional
`predict_proba()` output is shown as confidence. Uploaded files are processed
in memory and are not persisted by the local server.

The sample table also provides confidence-risk triage. The browser defaults to
80% as the high-confidence boundary, and users can change it after a comparison
without rerunning either model. Users can filter candidate-model high-confidence
errors and the narrower set where the baseline was correct but the candidate
became confidently wrong. This descriptive threshold helps prioritize
inspection; it is not a calibrated statistical acceptance rule.

When both models produce explanation maps, the comparison also reports their
cosine similarity. The dashboard automatically uses the held-out benign-retraining
threshold for recognised MNIST (`0.486`) and CIFAR-10 RGB (`0.04665`) image
shapes. Other datasets retain an explicitly labelled exploratory default of
`0.80`. The selected boundary and its source are displayed beside the controls
and can be changed for sensitivity analysis. Values below it are flagged,
including a separate hidden-drift filter for samples whose predicted class stayed the same.
Hidden-drift samples also appear in a compact review gallery above the table.
Each card identifies the sample and similarity score and opens the original
image beside the two model-specific contribution heatmaps.
The maps must come from comparable feature layouts; this score describes model
behavior and does not establish a causal explanation.

After selecting two to five models, choose one as the baseline and another as
the candidate. Only that pair is sent to the local comparison endpoint. To
compare another pair, change the two selectors and run again; the browser keeps
the same dataset, manifest, and optional image ZIP selected.

Each run starts with a collection overview that ranks all selected models by
accuracy and lists pairwise prediction agreement. The remaining panels retain
the detailed baseline-versus-candidate comparison for the selected pair.
Every model card also shows the first 12 characters of its SHA-256 digest; the
full digest, dataset digest and manifest digest are retained in the downloaded
JSON audit artifact so comparisons remain traceable even when filenames are
reused.

That paired comparison includes a two-sided exact McNemar test over discordant
correctness outcomes: samples where V1 alone was correct versus samples where
V2 alone was correct. The page reports the unadjusted p-value and treats
`p < 0.05` as evidence of a paired difference. A larger p-value is labelled
exploratory; it does not prove the two versions are equivalent. The result only
describes the uploaded labelled sample and should be repeated on representative
held-out data.

For models exposing `predict_proba()`, each overview card also reports mean
top-class confidence, the signed confidence-minus-accuracy gap, and ten-bin
top-label expected calibration error (ECE). Lower ECE means the stated
confidence more closely matches observed correctness on the uploaded labelled
sample. These are descriptive diagnostics for that sample, not guarantees for
future data.

Each probabilistic model card includes a compact reliability chart using the
same ten equal-width confidence bins. The blue bar is mean stated confidence,
the green bar is observed accuracy, and the row tooltip gives the bin range and
sample count. Large separation between the two bars identifies locally
miscalibrated confidence ranges; sparse bins should be interpreted cautiously.

A preflight panel runs first and reports compatibility for every model. Formal
comparison proceeds only when all selected models load successfully, accept the
manifest-selected feature columns, and return one prediction per CSV row.

For the selected pair, the class-diagnostics table groups results by true label
and reports each model's accuracy plus the signed candidate-minus-baseline
change. Positive rows are improvements; negative rows are regressions.

The following misclassification-path table explains where those errors went,
for example `8 → 3`, and compares the number of occurrences in the baseline and
candidate. A negative error-count change indicates fewer candidate mistakes.

After a successful run, **Download JSON report** saves the compatibility,
multi-model overview, and detailed pair result together. **Download sample
CSV** saves the per-sample labels, predictions, confidence values, and
transition category. Both downloads are generated locally in the browser.

Uploading a JSON manifest is optional. The built-in manifest builder reads the
selected CSV header locally and provides selectors for the label, sample ID,
image filename, and image dimensions. It excludes those metadata columns and
uses the remaining columns as ordered model inputs. Users must still explicitly
confirm that joblib files come from a trusted source.

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

The generated manifest includes `"image_shape": [8, 8]`, enabling a **Compare
maps** action on every sample. Compatible tree models use Tree SHAP. Other
classifiers with `predict_proba()` and `classes_` use a pixel-occlusion fallback.
RGB manifests use `"image_shape": [height, width, 3]`; channel contributions
are combined into a two-dimensional spatial heatmap for comparison.
The dialog names the method used for each model and shows contributions toward
each version's own predicted class: red supports that prediction and blue
opposes it. Each map is normalized independently, so color strength is not
directly comparable. The view does not establish that a highlighted pixel
caused the prediction.

When the generated cross-dataset summary is available, the same command adds a
**Cross-dataset overview** containing the completed SHAP/LIME evidence matrix
for Breast Cancer Wisconsin and binary Wine.

When the final split-and-probe robustness summary is available, it also adds a
**Split robustness** view. This page presents each dataset/split result and the
overall `SPLIT_SENSITIVE` conclusion. The default report is
`artifacts/experiment_27_lime_split_probe_robustness/reports/summary.json`.
