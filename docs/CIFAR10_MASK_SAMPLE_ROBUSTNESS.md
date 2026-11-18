# CIFAR-10 explanation audit: larger-sample masking check

## Question and design

The [masking-baseline sensitivity experiment](CIFAR10_MASK_BASELINE_SENSITIVITY.md)
used 50 jointly correct CIFAR-10 images per matched CNN width pair and three
random masks per image. Its mean-pixel and Gaussian-blur replacements gave
opposite directions. This follow-up asks whether that observed contrast
persists when the sample and random control are enlarged.

For each of the existing matched width pairs (seeds 42, 43, and 44), the audit
selects 20 images per actual class from the first 1,000 official test images
that *both* models classify correctly: 200 images per pair. Each top-ranked
10% or 20% pixel mask is compared with the average of **10** matched-size
random masks. The explained target remains the true class. Only positive
Integrated Gradients (IG) or Grad-CAM attribution entries are ranked. The
models, selected image indices, explanation maps, and random masks are fixed
between mean-pixel and blur runs for each pair; the replacement is the only
difference within that comparison. Model hashes and selected indices were
verified to match in the paired JSON reports.

The reported difference is `(original − top-masked true-class probability) −
(original − average random-masked true-class probability)`. Positive values
mean that the method's top-ranked mask reduced true-class probability more
than the random masks *under the stated replacement*. Values are probability
units, not accuracy percentage points.

## Results

Every condition had 200 informative images. Each row below is the average
top-minus-random difference across those 200 images.

| Seed | Model | Method | Mean 10% | Blur 10% | Mean 20% | Blur 20% |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 42 | V1 | IG | −0.0773 | +0.1063 | −0.0546 | +0.1006 |
| 42 | V1 | Grad-CAM | −0.1566 | +0.0558 | −0.1808 | +0.0753 |
| 42 | V2 | IG | −0.0528 | +0.0917 | −0.0639 | +0.0731 |
| 42 | V2 | Grad-CAM | −0.1384 | +0.0861 | −0.1558 | +0.1005 |
| 43 | V1 | IG | −0.1228 | +0.1223 | −0.0702 | +0.0511 |
| 43 | V1 | Grad-CAM | −0.1530 | +0.1142 | −0.1782 | +0.1079 |
| 43 | V2 | IG | −0.0677 | +0.1413 | −0.0814 | +0.0887 |
| 43 | V2 | Grad-CAM | −0.1064 | +0.1244 | −0.1383 | +0.1220 |
| 44 | V1 | IG | −0.0363 | +0.1094 | −0.0722 | +0.0592 |
| 44 | V1 | Grad-CAM | −0.0939 | +0.0947 | −0.0669 | +0.1327 |
| 44 | V2 | IG | −0.0611 | +0.1205 | −0.0586 | +0.0822 |
| 44 | V2 | Grad-CAM | −0.1336 | +0.0990 | −0.1460 | +0.0889 |

All **24/24** seed × model × method × mask-size average contrasts were negative
with mean-pixel replacement and positive with blur replacement. In individual
200-image conditions, **57–99/200** images had positive top-minus-random
differences under mean replacement, versus **115–141/200** under blur.

The larger sample and 10 random controls therefore preserve the earlier
*descriptive* direction flip. The contrast is not merely one unusual image or
one random mask. It remains a conditional result for these CNNs, these
jointly correct images, and these two replacements.

## Limits and decision

The images come from the same 1,000 official-test subset already explored in
the earlier audit. This is a sensitivity check, not an independent held-out
confirmation. The 24 conditions share images and related models; their signs
are descriptive and are not 24 independent statistical tests. Mean replacement
and blur both alter the input, and neither is established as the uniquely
correct intervention. Seed 42 retains the interrupted-training RNG caveat
documented in the earlier CIFAR-10 runs.

This result does not establish causal feature importance, rank IG against
Grad-CAM, or justify an automatic explanation-quality label. The platform
should keep perturbation diagnostics tied to their intervention and show the
underlying sample-level maps and predictions to the reviewer.

## Reproduce

The command below reproduces the seed-44 mean run. Change `--replacement` to
`blur` and choose a separate output path for the matched comparison. Seed-42
and seed-43 bundle and run paths are listed in the
[initial audit](CIFAR10_MAP_PERTURBATION_AUDIT.md).

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression.cifar10_map_perturbation \
  --bundle artifacts/cifar10_directional_version_comparison_seed44/upload_bundle \
  --baseline-run artifacts/cifar10_cnn_full_seed44 \
  --candidate-run artifacts/cifar10_cnn_wide_seed44 \
  --per-class 20 --random-repeats 10 --replacement mean \
  --output artifacts/cifar10_map_perturbation_large_mean_seed44.json
```

The six full JSON reports, including per-image values, selected indices, and
model SHA-256 hashes, remain in Git-ignored `artifacts/` as
`cifar10_map_perturbation_large_{mean,blur}_seed{42,43,44}.json`.
