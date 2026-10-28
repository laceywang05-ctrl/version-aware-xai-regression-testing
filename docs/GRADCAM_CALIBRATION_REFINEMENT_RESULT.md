# Grad-CAM Calibration Refinement

## Motivation

The initial MNIST experiment used 1,000 training images and two epochs. Grad-CAM
had 86.7% informative-map coverage and a zero fifth-percentile similarity, so no
useful automatic threshold could be established. This refinement tests whether
the result was merely caused by under-training.

## Expanded design

- 3,000 MNIST training images
- 5 training epochs
- 50 explanation images per seed
- 3 benign-retraining calibration seeds
- 2 disjoint held-out seeds
- 150 calibration pairs and 100 held-out pairs per method

## Results

| Method | Informative coverage | Calibration median | Fifth percentile | Held-out review rate | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Integrated Gradients | 100.0% | 0.959297 | 0.896604 | 7.0% | Calibrated |
| Grad-CAM | 100.0% | 0.314381 | 0.009905 | 0.0% | Insufficient separation |

Longer training removed the Grad-CAM zero-map coverage problem, but benign
retraining still produced highly dissimilar spatial maps. A threshold near zero
would only flag almost completely disjoint maps and would not be useful as an
automatic regression rule. Grad-CAM therefore remains available for visual and
sample-level inspection but is not assigned a confirmatory drift threshold.

The larger experiment supersedes the initial Integrated Gradients estimate. The
MNIST upload workflow now uses `0.896604` for comparisons where both models use
Integrated Gradients.

## Interpretation

This is a meaningful negative result: prediction-level stability does not imply
that Grad-CAM maps are stable across independently trained versions. The system
reports the maps and similarity values, but it avoids presenting unstable
benign variation as evidence of a model regression.
