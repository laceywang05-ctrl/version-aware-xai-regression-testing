# Ready-to-upload model bundles

These are **small demonstration bundles tracked in Git**. They are available immediately after cloning the repository. Unlike the local `artifacts/` directory, no generation command or external dataset download is needed. The 10,000-image CIFAR-10 full-test bundle is intentionally excluded.

To use them, start the Python-backed dashboard as described in [the teacher deployment guide](../../docs/TEACHER_LOCAL_DEPLOYMENT.md), open `http://127.0.0.1:8000/`, and select **Compare uploaded models**. Every bundle contains a labelled `samples.csv`, a matching `manifest.json`, and already trained `.joblib` models. Upload the first two models in the table as **Model file 1** and **Model file 2**. For four-model bundles, select the other two together under **Additional models**. Select the baseline and candidate in the dropdowns before running the comparison. Upload `images.zip` where listed.

| Folder | Task / test rows | First two model files | Other files |
| --- | --- | --- | --- |
| `breast_cancer_model_demo/` | Breast Cancer tabular classification / 171 | `model_2_tree.joblib`, `model_3_random_forest.joblib` | `model_1_dummy.joblib`, `model_4_logistic_regression.joblib` |
| `demo_image_upload/` | Small 8×8 digit classification / 60 | `model_v1.joblib`, `model_v2.joblib` | `images.zip` |
| `digits_four_model_evaluation/` | Small 8×8 digit classification / 60 | `model_v1.joblib`, `model_v2.joblib` | `model_v3_extra_trees.joblib`, `model_v4_knn.joblib`, `images.zip` |
| `mnist_model_demo/` | MNIST digit classification / 1,000 | `model_2_tree.joblib`, `model_3_random_forest.joblib` | `model_1_dummy.joblib`, `model_4_extra_trees.joblib`, `images.zip` |
| `cifar10_model_demo/` | CIFAR-10 classification / 1,000 | `model_2_tree.joblib`, `model_3_random_forest.joblib` | `model_1_dummy.joblib`, `model_4_extra_trees.joblib`, `images.zip` |
| `mnist_neural_demo/` | MNIST neural classification / 500 | `model_1_mlp_small.joblib`, `model_2_mlp_large.joblib` | `model_3_cnn_shallow.joblib`, `model_4_cnn_deep.joblib`, `images.zip` |
| `cifar10_clear_quality_demo/` | CIFAR-10 neural classification / 1,000 | `baseline_seed42_best_epoch74.joblib`, `candidate_seed43_early_epoch05.joblib` | `images.zip` |
| `demo_regression_upload/` | Diabetes numeric regression / 133 | `linear_regression.joblib`, `gradient_boosting.joblib` | `decision_tree.joblib`, `random_forest.joblib` (other pair choices) |

Choose **Classification** for all folders except `demo_regression_upload/`, which requires **Regression**. Upload each folder's own `manifest.json`; it defines the target and the model feature order. Do not mix a model from one folder with another folder's CSV or manifest.

For neural-model bundles, also install the optional dependency with `python -m pip install -e '.[neural]'` in the active virtual environment. The included `.joblib` files are for this trusted project demonstration; loading arbitrary third-party joblib files can execute code. The public GitHub Pages demo is read-only, while these files are for the Python-backed upload website.
