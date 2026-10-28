# MNIST Neural Explanation Validation

## Purpose

This experiment verifies that the upload workflow can compare real neural
classifiers with both Integrated Gradients and Grad-CAM. It also checks that a
Tree SHAP calibration threshold is not reused for a different explanation
method.

## Experimental setup

- Dataset: official MNIST train/test arrays from `mnist.npz`
- Training sample: first 1,000 training images
- Evaluation sample: first 500 labelled test images
- Explanation sample: first 100 evaluation images
- Training: three epochs with Adam (`learning_rate=0.003`)
- Integrated Gradients baseline: zero image, 24 interpolation steps
- Grad-CAM layer: final convolutional layer

Four upload-ready models were evaluated:

| Model | Architecture | Parameters | Explanation | Accuracy |
| --- | --- | ---: | --- | ---: |
| `model_1_mlp_small.joblib` | 784–64–10 MLP | 50,890 | Integrated Gradients | 82.4% |
| `model_2_mlp_large.joblib` | 784–128–64–10 MLP | 109,386 | Integrated Gradients | 80.0% |
| `model_3_cnn_shallow.joblib` | 1×8 convolution + linear head | 15,770 | Grad-CAM | 83.4% |
| `model_4_cnn_deep.joblib` | 1×8×16 convolutions + pooled head | 3,818 | Grad-CAM | 68.6% |

## Paired comparison results

| Comparison | Accuracy V1 → V2 | Prediction agreement | Exact McNemar p | Explanation maps compared | Below exploratory similarity boundary | Same prediction, changed explanation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Integrated Gradients MLP pair | 82.4% → 80.0% | 84.4% | 0.1818 | 100 | 18 | 6 |
| Grad-CAM CNN pair | 83.4% → 68.6% | 73.8% | 1.17×10⁻¹² | 100 | 93 | 66 |

The MLP accuracy difference did not reach the conventional 0.05 significance
level. The CNN pair showed strong evidence of a prediction-performance change,
but it also deliberately compares architectures with different capacity; it is
therefore a system-validation case rather than a claim that deeper CNNs are
generally worse.

## Threshold interpretation

The current similarity boundary of 0.80 is explicitly **exploratory** for both
Integrated Gradients and Grad-CAM. The earlier MNIST threshold of 0.486 was
calibrated for Tree SHAP under benign retraining and is not method-transferable.
The software now checks the explanation method before applying a calibrated
threshold. A future experiment should estimate separate benign-retraining
distributions for Integrated Gradients and Grad-CAM before using their review
counts as confirmatory evidence.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression neural-image-demo \
  --input mnist.npz \
  --output-dir artifacts/mnist_neural_demo \
  --limit 1000 --test-limit 500 --epochs 3
```

The generated folder contains the four `.joblib` models, `samples.csv`,
`images.zip`, `manifest.json`, and architecture/accuracy metadata in
`README.json`.
