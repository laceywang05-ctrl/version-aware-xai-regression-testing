# CIFAR-10 CNN explanation-map perturbation audit

## Question

For a trained model, does masking pixels that an explanation map marks as
positive evidence reduce the explained class's probability more than masking
the same number of randomly chosen pixels? This is an exploratory check of
the existing Integrated Gradients (IG) and Grad-CAM maps, motivated by the
need to assess maps beyond visual appearance. It is related to the deletion
diagnostics in [RISE (Petsiuk et al., BMVC 2018)](https://bmva-archive.org.uk/bmvc/2018/contents/papers/1064.pdf)
and the broader warning from [Sanity Checks for Saliency Maps (Adebayo et al., NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
This audit is **not** the full deletion/insertion benchmark in RISE.

## Fixed protocol

- Use the three existing matched CIFAR-10 CNN width pairs (seeds 42, 43, 44),
  without retraining. In each pair, V1 has 32/64/128 channels and V2 has
  40/80/160 channels.
- From each pair's first 1,000 official-test images, use a fixed random seed
  to select **five images per class that both models classified correctly**:
  50 images per pair. Different pairs can select different image indices.
- Explain the **true class**, which is also each selected model's predicted
  class. For both IG and Grad-CAM, rank the 32 × 32 spatial map's positive
  entries and mask its top 10% or 20% of pixel positions. An image enters a
  condition only if it has enough positive map positions for that mask size.
- Replace all three RGB channels at a selected pixel with the training-set
  mean RGB value. Compare the target-class probability before and after
  masking. For each image and mask size, compare the top-map mask with the
  average of **three matched-size random masks**.
- Report `top-minus-random = (original − top-masked probability) −
  (original − mean random-masked probability)`. Positive values mean the
  map-selected pixels caused a larger probability drop under this particular
  intervention. The random masks are reused across methods on the same image.

## Observed results

All 50 selected images had sufficiently many positive map positions in every
condition. The table shows mean top-minus-random probability change; values
are probability units, so `−0.075` corresponds to a 7.5-percentage-point
*smaller* drop than random masking.

| Seed | Model | IG, 10% | Grad-CAM, 10% | IG, 20% | Grad-CAM, 20% |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | V1 | −0.075 | −0.112 | −0.079 | −0.152 |
| 42 | V2 | −0.115 | −0.084 | −0.149 | −0.110 |
| 43 | V1 | −0.040 | −0.099 | −0.049 | −0.135 |
| 43 | V2 | −0.004 | −0.072 | −0.036 | −0.061 |
| 44 | V1 | −0.006 | −0.058 | −0.085 | −0.084 |
| 44 | V2 | +0.022 | −0.083 | −0.031 | −0.125 |

Only **one of the 24** seed × model × method × mask-size mean contrasts was
positive (seed 44 V2, IG at 10%). Even that condition was positive for only
24 of its 50 individual images. These measurements do **not** establish that
the maps are unfaithful, or that IG is superior to Grad-CAM. They show that
neither method consistently passed this chosen pixel-mean masking check.

## Boundaries and next interpretation

Replacing isolated pixels with a constant mean creates images outside the
natural CIFAR-10 distribution. The measured probability change may reflect
that intervention, spatial structure, model saturation, or class competition
as well as map quality. The 50 images per pair were selected conditional on
both models being correct and thus do not represent all test images. Three
random masks per image and three training seeds are limited replication. The
24 conditions were inspected descriptively, without a confirmatory multiple-
testing claim. IG uses a zero baseline; Grad-CAM and IG have different
semantics, so raw attribution values must not be compared directly.

The practical conclusion is to keep these maps as **visual, exploratory
evidence** in the platform. Their similarity values or this perturbation
score should not be converted into an automatic quality ranking or release
gate without a stronger, preregistered evaluation protocol.

## Reproduce

Run the following for each seed, substituting the paths shown below. The
generated JSON includes selected image indices, per-image contrasts, model
hashes and the complete protocol metadata; it stays under Git-ignored
`artifacts/`.

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression.cifar10_map_perturbation \
  --bundle artifacts/cifar10_directional_version_comparison_seed44/upload_bundle \
  --baseline-run artifacts/cifar10_cnn_full_seed44 \
  --candidate-run artifacts/cifar10_cnn_wide_seed44 \
  --per-class 5 --random-repeats 3 \
  --output artifacts/cifar10_map_perturbation_seed44.json
```

For seed 42, use `artifacts/cifar10_directional_version_comparison/upload_bundle`,
`artifacts/cifar10_cnn_full_2026-10-29` and
`artifacts/cifar10_cnn_wide_seed42`. For seed 43, replace `seed44` with
`seed43` in the three input paths and output path. The seed 42 V1 training
run has the previously documented resume/RNG caveat.
