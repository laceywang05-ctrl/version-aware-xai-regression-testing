# CIFAR-10 explanation masking on images unseen by the earlier audit

## Question and evaluation set

The [larger-sample audit](CIFAR10_MASK_SAMPLE_ROBUSTNESS.md) found opposite
masking-contrast directions for mean-pixel and Gaussian-blur replacement on
images drawn from official test indices 0–999. This follow-up checks whether
the pattern extends to **different images** from official test indices
1000–9999, using the same trained CNN pairs and the same audit protocol.

For each matched original-width V1 / wider-width V2 pair (training seeds 42,
43, 44), a fixed random seed selects 20 jointly correct images per CIFAR-10
class from indices 1000–9999, or 200 images per pair. The true-class
probability is measured after masking the top 10% or 20% of pixels ranked by
positive Integrated Gradients (IG) or Grad-CAM values, and compared with the
average of 10 equal-size random masks. The random masks, image indices and
model hashes match between mean-pixel and Gaussian-blur replacement within
each pair. Every condition had 200 informative images.

The score is `(original − top-masked probability) − (original − average
random-masked probability)`. A positive score favors top-ranked masking under
that replacement. It is a **probability-unit contrast**, not an accuracy
change. The official test labels and saved predictions were checked against
the uploaded models on the complete 1000–9999 window before sampling.

## Results

| Seed | Model | Method | Mean 10% | Blur 10% | Mean 20% | Blur 20% |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 42 | V1 | IG | −0.0369 | +0.1448 | −0.0580 | +0.1044 |
| 42 | V1 | Grad-CAM | −0.1480 | +0.1058 | −0.1936 | +0.1254 |
| 42 | V2 | IG | −0.0667 | +0.1308 | −0.0924 | +0.0886 |
| 42 | V2 | Grad-CAM | −0.1206 | +0.0920 | −0.1417 | +0.1378 |
| 43 | V1 | IG | −0.0925 | +0.1422 | −0.0946 | +0.0817 |
| 43 | V1 | Grad-CAM | −0.1413 | +0.0911 | −0.1762 | +0.1137 |
| 43 | V2 | IG | −0.0656 | +0.0899 | −0.0693 | +0.0659 |
| 43 | V2 | Grad-CAM | −0.1245 | +0.1095 | −0.1468 | +0.1331 |
| 44 | V1 | IG | −0.0361 | +0.1037 | −0.0813 | +0.0523 |
| 44 | V1 | Grad-CAM | −0.0857 | +0.0792 | −0.0999 | +0.0899 |
| 44 | V2 | IG | **+0.0044** | +0.1360 | −0.0540 | +0.0808 |
| 44 | V2 | Grad-CAM | −0.1224 | +0.0772 | −0.1699 | +0.0996 |

Under mean-pixel replacement, **23/24** condition means were negative; the
seed-44 V2 IG 10% condition was slightly positive (+0.0044). Under blur
replacement, **24/24** were positive. Within individual 200-image conditions,
52–97 images had a positive top-minus-random score under mean replacement and
106–140 under blur. The selected image sets do not overlap indices 0–999
used by the earlier masking audits; the seed pairs and model files are the
same.

## Interpretation and limits

The intervention-dependent direction persists on images not used by the
earlier masking analysis, with one near-zero mean-replacement exception.
This strengthens the *descriptive* sensitivity finding; it does not identify
one replacement as ground truth, establish causal importance, or rank IG
against Grad-CAM. The 24 conditions share images and related models and are
not independent hypothesis tests. The images were selected only when both
models were correct, so the result does not cover failure cases. The models'
full official-test predictions had been evaluated in prior development; this
is an unseen set **for the masking audit**, not a fresh test set for overall
model performance. Seed 42 retains the previously documented resumed-training
RNG caveat.

For the platform, retain the explanatory maps and underlying examples for
review. Any perturbation score must state its replacement, mask size and
selection rule, and should not become an automatic explanation-quality gate.

## Reproduce

For seed 44, run the following with `--replacement mean`, then repeat with
`--replacement blur` and a distinct output path. Substitute the seed-42 and
seed-43 paths from the [initial audit](CIFAR10_MAP_PERTURBATION_AUDIT.md) for
the other pairs.

```bash
MPLCONFIGDIR=/tmp/xai-matplotlib-cache PYTHONPATH=src \
.venv/bin/python -m xai_regression.cifar10_map_perturbation \
  --bundle artifacts/cifar10_directional_version_comparison_seed44/upload_bundle \
  --baseline-run artifacts/cifar10_cnn_full_seed44 \
  --candidate-run artifacts/cifar10_cnn_wide_seed44 \
  --archive cifar-10-python.tar.gz \
  --test-start 1000 --test-stop 10000 \
  --per-class 20 --random-repeats 10 --replacement mean \
  --output artifacts/cifar10_map_perturbation_unseen_mean_seed44.json
```

Six detailed JSON reports with selected official indices, per-image scores,
model SHA-256 hashes and protocol settings remain in Git-ignored `artifacts/`
as `cifar10_map_perturbation_unseen_{mean,blur}_seed{42,43,44}.json`.
