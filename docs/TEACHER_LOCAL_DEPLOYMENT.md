# Run the interactive upload dashboard

This guide is for a supervisor who wants to run the **working upload website** on their own computer. The public GitHub Pages demo displays a precomputed report; it cannot run new model uploads because GitHub Pages does not run the Python comparison service.

## 1. Install and start

Requirements: Python 3.11 or 3.13, Git, and enough disk space to install the Python dependencies. Run these commands in a terminal:

```bash
git clone https://github.com/laceywang05-ctrl/version-aware-xai-regression-testing.git
cd version-aware-xai-regression-testing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
MPLCONFIGDIR=/tmp/xai-matplotlib-cache python -m xai_regression serve
```

On Windows, activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`. Then open **http://127.0.0.1:8000/** in a browser and choose **Compare uploaded models**. Leave the terminal running while using the site; press `Ctrl+C` to stop it. If port 8000 is occupied, add `--port 8001` to the final command and open `http://127.0.0.1:8001/`.

Do not open `dashboard/index.html` using a `file://` URL: the upload API requires the Python server. The service listens on the computer's loopback address by default, so this procedure makes it available to the person running it, not to the public internet.

## 2. Generate example uploads

The repository does not store trained `.joblib` examples because generated artifacts are ignored by Git. These two commands build small, ready-to-upload bundles locally without downloading an external dataset:

```bash
python -m xai_regression image-demo
python -m xai_regression regression-demo
```

### Example A: handwritten-digit classification

Select **Classification** in the upload form. Choose these files from `artifacts/demo_image_upload/`:

| Upload field | File |
| --- | --- |
| Model file 1 | `model_v1.joblib` |
| Model file 2 | `model_v2.joblib` |
| Labelled dataset | `samples.csv` |
| Manifest | `manifest.json` |
| Sample images ZIP | `images.zip` |

Confirm that the generated model files are trusted. Select `model_v1.joblib` as **Baseline (V1)** and `model_v2.joblib` as **Candidate (V2)**, then click **Run local comparison**. Open the result cards to inspect overall metrics, correctness transitions, class errors, explanation maps, and individual images.

### Example B: numeric regression

Select **Regression**. Choose these files from `artifacts/demo_regression_upload/`:

| Upload field | File |
| --- | --- |
| Model file 1 | `linear_regression.joblib` |
| Model file 2 | `gradient_boosting.joblib` |
| Labelled dataset | `samples.csv` |
| Manifest | `manifest.json` |

Select different baseline and candidate models if desired. The regression result shows MAE, RMSE, R² and sample-level absolute-error changes. There are no class transitions for a numeric target.

## 3. Use your own models

Upload two to five **already trained, trusted** `.joblib` models, one labelled CSV, and a manifest describing the target column and the exact feature-column order used during training. The dashboard can build a basic manifest from CSV column choices. Models must accept the same feature layout and expose `predict()`; classification confidence requires `predict_proba()`. Optional images require matching filenames and an image ZIP. Upload size and computation are bounded; see [dashboard/README.md](../dashboard/README.md) for current limits and supported explanation methods.

**Safety:** Loading `.joblib` can execute Python code. Do not upload files from unknown sources to this local prototype or expose this server directly as a public upload service. Public arbitrary-user uploads require a separately secured and isolated backend.

## 4. Troubleshooting

| Symptom | Action |
| --- | --- |
| `python: command not found` | Activate `.venv`, or use `.venv/bin/python` on macOS/Linux. |
| `Address already in use` | Reuse the existing server tab, stop the old server, or start with `--port 8001`. |
| `Failed to fetch` | Use the `http://127.0.0.1:8000/` address and make sure the terminal server is still running. |
| `No module named ...` | In the activated virtual environment, rerun `python -m pip install -e .`. |
| Model validation error | Check that both models use the same feature columns, preprocessing, and ordered manifest features as the CSV. |

The public read-only example is available at https://laceywang05-ctrl.github.io/version-aware-xai-regression-testing/demo/ .
