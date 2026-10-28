# MNIST Neural Explanation Calibration

## Design

Integrated Gradients and Grad-CAM were calibrated separately under benign
retraining. Each method used one fixed baseline model, three calibration seeds,
two held-out validation seeds, 1,000 MNIST training images, and 30 explanation
images per run. The proposed boundary is the fifth percentile of informative
calibration similarities.

## Results

| Method | Informative coverage | Fifth-percentile boundary | Held-out review rate | Decision |
| --- | ---: | ---: | ---: | --- |
| Integrated Gradients | 100.0% | 0.827263 | 10.0% | Calibrated |
| Grad-CAM | 86.7% | 0.000000 | 0.0% | Insufficient separation |

The Integrated Gradients boundary is now used automatically for MNIST when both
uploaded models use Integrated Gradients. It is not transferred to other
datasets or explanation methods.

The Grad-CAM experiment produced a fifth percentile of zero even after excluding
pairs where either heatmap contained no signal. This boundary cannot distinguish
ordinary retraining variation from severe drift, so the software deliberately
keeps Grad-CAM on an exploratory threshold. A larger calibration sample, longer
training, or a different target convolutional layer should be evaluated before
claiming a calibrated Grad-CAM policy.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache \
PYTHONPATH=src .venv/bin/python -m xai_regression neural-explanation-calibration \
  --input mnist.npz --train-limit 1000 --sample-limit 30 --epochs 2
```
