# CIFAR-10 explanation audit: masking-baseline sensitivity

## Question and paired design

The [initial pixel-masking audit](CIFAR10_MAP_PERTURBATION_AUDIT.md) mostly
found that masking IG or Grad-CAM's highly ranked pixels reduced the target
class probability *less* than masking random pixels. This follow-up tests
whether that finding depends on what replaces the masked pixels.

The three existing matched CNN width pairs (seeds 42, 43 and 44), the 50
jointly correct images selected for each pair (five per class), the true-class
target, the IG/Grad-CAM maps, the 10% and 20% spatial mask sizes, and the three
random masks per image are identical to the first audit. The exported model
SHA-256 hashes and selected image-index lists were checked to match the
original reports. Only the replacement changed:

- **Mean:** replace each selected RGB pixel with the training-set mean RGB.
- **Blur:** replace each selected RGB pixel with its corresponding pixel from
  the *same image* after a Gaussian blur (`σ=2` pixels in height and width,
  reflection at the image boundary; RGB channels are not mixed).

For both methods, only positive attribution entries are ranked. The reported
contrast is `(original − top-masked target probability) − (original − mean
random-masked target probability)`. A positive number means the top-ranked
mask lowered the target probability more than three matched-size random masks
under that particular replacement. Values are probability units.

## Results

| Seed | Model | Method | Mean 10% | Blur 10% | Mean 20% | Blur 20% |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 42 | V1 | IG | −0.075 | +0.105 | −0.079 | +0.050 |
| 42 | V1 | Grad-CAM | −0.112 | +0.122 | −0.152 | +0.162 |
| 42 | V2 | IG | −0.115 | +0.094 | −0.149 | +0.045 |
| 42 | V2 | Grad-CAM | −0.084 | +0.137 | −0.110 | +0.175 |
| 43 | V1 | IG | −0.040 | +0.097 | −0.049 | +0.087 |
| 43 | V1 | Grad-CAM | −0.099 | +0.131 | −0.135 | +0.188 |
| 43 | V2 | IG | −0.004 | +0.100 | −0.036 | +0.068 |
| 43 | V2 | Grad-CAM | −0.072 | +0.139 | −0.061 | +0.229 |
| 44 | V1 | IG | −0.006 | +0.078 | −0.085 | +0.091 |
| 44 | V1 | Grad-CAM | −0.058 | +0.127 | −0.084 | +0.196 |
| 44 | V2 | IG | +0.022 | +0.102 | −0.031 | +0.113 |
| 44 | V2 | Grad-CAM | −0.083 | +0.124 | −0.125 | +0.143 |

With mean replacement, **1 of 24** seed × model × method × mask-size mean
contrasts was positive. With blur replacement, **24 of 24** were positive.
For the blur runs, 28–39 of 50 individual images in each condition had a
positive contrast. The complete per-image values, exact unrounded means,
selected indices, and model hashes are in the generated JSON reports.

## Interpretation and limits

This is direct evidence that this audit's *direction* depends on its masking
baseline. It does **not** prove that blur is the correct intervention, that
the maps are faithful, or that either IG or Grad-CAM is better. Even blurred
pixel replacement changes the input distribution, and the two methods encode
different attribution concepts. The selected images were conditional on both
models being correct; there are only 50 per pair and three random masks per
image. The 24 conditions were examined descriptively, without a
multiple-testing-adjusted confirmatory claim. Seed 42 retains its previously
documented interrupted-training RNG caveat.

For the platform, the conclusion is practical: do not turn a single
perturbation score into an automatic explanation-quality label or model-release
decision. Keep the maps and any perturbation diagnostic visibly tied to the
chosen replacement protocol and to the exact model pair.

## Reproduce

The command below runs the blur condition for seed 44. Repeat with the seed
42 and 43 input paths listed in the [initial audit](CIFAR10_MAP_PERTURBATION_AUDIT.md).
Run the same command with `--replacement mean` and a different output path
to reproduce the original condition.

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression.cifar10_map_perturbation \
  --bundle artifacts/cifar10_directional_version_comparison_seed44/upload_bundle \
  --baseline-run artifacts/cifar10_cnn_full_seed44 \
  --candidate-run artifacts/cifar10_cnn_wide_seed44 \
  --per-class 5 --random-repeats 3 --replacement blur \
  --output artifacts/cifar10_map_perturbation_blur_seed44.json
```

All output JSON files remain under Git-ignored `artifacts/`.
