# Final evaluation summary

## 1. System objective

This project investigates whether model-version regression testing can extend
beyond aggregate predictive performance to include changes in local
explanations. The implemented local dashboard accepts two to five trusted model
artifacts, evaluates them on the same labelled sample, ranks the collection,
and provides a detailed baseline-versus-candidate comparison.

The final prototype supports:

- classification and regression tasks;
- tabular, grayscale-image and RGB-image inputs;
- two to five uploaded scikit-learn-compatible model artifacts;
- aggregate, class-level and sample-level predictive diagnostics;
- Tree SHAP, LIME, pixel occlusion, permutation importance, Integrated
  Gradients and Grad-CAM explanation evidence;
- calibrated dataset-specific explanation review thresholds for MNIST and
  CIFAR-10 RGB comparisons;
- a separately calibrated MNIST Integrated Gradients threshold, while Grad-CAM
  is deliberately retained as visual-only exploratory evidence;
- downloadable JSON, readable summaries and sample-level CSV files; and
- SHA-256 fingerprints for model, dataset and manifest traceability.

## 2. Research questions

### RQ1 — Can explanation regression testing distinguish controlled model changes from benign variation?

Repeated held-out Tree SHAP experiments compared seed-only retraining with
controlled tree-configuration changes. On Breast Cancer Wisconsin, the
combined-constraint scenario produced a 50.0% review rate versus 16.7% for the
control. Its paired comparison remained separated after Bonferroni adjustment
(`adjusted p = 0.039`). The other two candidate scenarios were descriptive but
did not remain below 0.05 after adjustment.

On the binary Wine task, the seed-only control review rate was 3.3%, whereas
the larger-leaf and combined-constraint scenarios each reached 76.7%. Both
paired comparisons remained separated after adjustment (`adjusted p < 0.001`).
The shallow-depth scenario did not separate from the control because the
reference forest was already shallow.

**Answer:** the method can separate some controlled changes from benign
retraining, but sensitivity depends on the dataset and the type of model
change. A review flag identifies evidence requiring inspection; it is not proof
that the candidate is harmful.

### RQ2 — Do the findings transfer across datasets and explanation methods?

The cross-dataset evidence matrix covers Breast Cancer Wisconsin and Wine with
both Tree SHAP and LIME. The strongest held-out SHAP candidate review rates were
50.0% and 76.7%, respectively. Independent-seed confirmatory LIME evaluation
retained separation under the multiplicity-aware policy:

| Dataset | Adjusted control review rate | Adjusted candidate review rate | Exact paired p |
| --- | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 6.7% | 53.3% | 0.000519 |
| Wine | 3.3% | 43.3% | 0.001831 |

The image experiments exposed an important non-transfer result. A similarity
threshold of 0.80 flagged 97.35% of benign MNIST comparisons and 100% of benign
CIFAR-10 RGB comparisons. Benign-retraining calibration produced provisional
fifth-percentile thresholds of 0.486 for MNIST and 0.047 for CIFAR-10. Their
held-out benign flag rates were 6.89% and 6.9%, close to the intended 5% review
workload.

The same limitation also appeared across neural explanation methods. An
expanded MNIST experiment used 3,000 training images, five epochs, 50
explanation images per seed, three calibration seeds and two disjoint held-out
seeds. Integrated Gradients achieved 100% informative coverage, a
fifth-percentile boundary of `0.896604`, and a 7.0% held-out review rate. This
threshold is active only when both compared MNIST models use Integrated
Gradients. Grad-CAM also reached 100% informative coverage, but its benign
similarities remained unstable: the fifth percentile was `0.009905` and the
median was `0.314381`. A threshold near zero would detect only almost-disjoint
maps, so Grad-CAM remains available for visual sample review without an
automatic confirmatory drift decision.

**Answer:** the overall workflow transfers, but numerical thresholds do not.
Thresholds must be calibrated for the dataset, representation, model family and
explanation method rather than treated as universal constants. Some methods may
fail to yield a useful automatic threshold even when their maps can still be
displayed and compared.

### RQ3 — Can the prototype make model-version differences concrete and auditable?

The image demonstrations ranked four uploaded models and exposed individual
samples that improved, regressed or retained their predictions. On MNIST,
Random Forest and Extra Trees achieved 84.7% and 84.8% accuracy with 89.3%
prediction agreement; the exact paired McNemar p-value was 1.0, so the
one-sample accuracy difference was inconclusive. The dashboard still revealed
the individual disagreements and explanation-map changes hidden by the nearly
identical aggregate scores.

On CIFAR-10, the initial 1,000-image training run reached 33.4% best accuracy.
A controlled scaling experiment increased the training sample to 5,000 and
raised the best accuracy to 37.7% and macro F1 to 36.9%. This improvement did
not remove the limitation of feeding flattened raw pixels to lightweight tree
models. The experiment validates the workflow, not competitive CIFAR-10 model
performance.

A later full-training CNN extension used all 50,000 official training images
for a 45,000/5,000 training/validation split and kept the 10,000-image
official test set separate. Across two random seeds, validation-selected CNN
checkpoints reached 88.54% and 88.83% test accuracy. Their respective
epoch-5 baselines reached 70.50% and 71.29%, yielding gains of 18.04 and
17.54 percentage points. The two final CNNs had 90.92% prediction agreement
on the same test images. See
[`CIFAR10_FULL_CNN_RESULT.md`](CIFAR10_FULL_CNN_RESULT.md) and
[`CIFAR10_CNN_SEED_REPEAT_RESULT.md`](CIFAR10_CNN_SEED_REPEAT_RESULT.md).

For every completed comparison, the dashboard can export matched predictions,
confidence, transitions, explanation similarity, compatibility results and
full SHA-256 artifact fingerprints. These records distinguish exact model
content even when filenames are reused.

**Answer:** the prototype provides a usable audit trail from collection-level
ranking to class-specific changes and individual labelled examples. Its output
supports human review rather than automatic deployment approval.

## 3. Consolidated findings

1. Predictive agreement and explanation stability measure different aspects of
   a version change. Candidate models can retain more than 99% prediction
   agreement while triggering explanation review.
2. Held-out controls are essential. Small calibration samples produced unstable
   control-warning rates that improved after calibration was expanded and
   separated from evaluation seeds.
3. Multiple-testing adjustment reduces false review selections. The
   Bonferroni-aware LIME policy preserved candidate-control separation on both
   tabular datasets while lowering control review rates.
4. Aggregate metrics are insufficient for release review. Class tables,
   confusion paths and sample-level transitions reveal regressions hidden by a
   small average improvement.
5. Explanation thresholds are context-specific. Transferring 0.80 from one
   setting to MNIST or CIFAR-10 created unusably large review queues.
6. Explanation availability is not equivalent to policy readiness. Integrated
   Gradients produced a usable MNIST benign-retraining boundary, whereas
   Grad-CAM remained too variable for confirmatory automatic review.
7. Reproducibility requires artifact identity as well as parameters. SHA-256
   fingerprints connect exported evidence to the exact uploaded bytes.

## 4. Evidence boundaries

The current evidence supports a research prototype, not a universal production
release gate. In particular:

- review flags indicate statistical or threshold-based separation, not model
  harm, unfairness, causality or deployment unsafety;
- Tree SHAP, LIME, pixel occlusion, permutation importance, Integrated
  Gradients and Grad-CAM have different semantics and their raw scores are not
  interchangeable;
- the main MNIST/CIFAR-10 ranking experiments use lightweight scikit-learn
  models, while the neural extension uses research MLP/CNN architectures
  rather than production-scale image networks; the full CIFAR-10 CNN result
  has only two training seeds;
- MNIST and CIFAR-10 thresholds are empirical workload controls estimated from
  small benign-retraining samples;
- user-uploaded `.joblib` models and CIFAR pickle archives must come from trusted
  sources because deserialization can execute Python code; and
- all reported statistical conclusions describe the supplied held-out samples
  and experimental model families.

## 5. Reproducibility status

- Automated test suite: **140 tests passed** after the full CIFAR-10 CNN extension.
- Data splits and random seeds are fixed and documented by each experiment.
- Calibration and evaluation seeds are disjoint in the confirmatory workflows.
- Machine-readable experiment summaries are generated under `artifacts/`.
- Human-readable result documents are stored under `docs/`.
- Uploaded comparison reports contain artifact fingerprints and can be exported
  locally without persisting the uploaded files on the server.

## 6. Overall conclusion

The project demonstrates an end-to-end, version-aware XAI regression-testing
workflow across classification and regression, tabular and image data, multiple
model versions and multiple explanation techniques. The strongest conclusion
is methodological: useful explanation monitoring requires matched held-out
evaluation, explicit benign controls, multiplicity-aware inference and
context-specific threshold calibration. The dashboard operationalizes these
requirements by connecting aggregate evidence to concrete samples and exact
artifact identities while keeping the final decision with the reviewer.

## 7. Supporting result documents

- [`HELDOUT_POLICY_VALIDATION_RESULT.md`](HELDOUT_POLICY_VALIDATION_RESULT.md)
- [`SECOND_DATASET_VALIDATION_RESULT.md`](SECOND_DATASET_VALIDATION_RESULT.md)
- [`LIME_CONFIRMATORY_POLICY_RESULT.md`](LIME_CONFIRMATORY_POLICY_RESULT.md)
- [`CROSS_DATASET_SUMMARY_RESULT.md`](CROSS_DATASET_SUMMARY_RESULT.md)
- [`DIGITS_FOUR_MODEL_RESULT.md`](DIGITS_FOUR_MODEL_RESULT.md)
- [`MNIST_MULTIMODEL_RESULT.md`](MNIST_MULTIMODEL_RESULT.md)
- [`MNIST_EXPLANATION_CALIBRATION_RESULT.md`](MNIST_EXPLANATION_CALIBRATION_RESULT.md)
- [`CIFAR10_MULTIMODEL_RESULT.md`](CIFAR10_MULTIMODEL_RESULT.md)
- [`CIFAR10_EXPLANATION_CALIBRATION_RESULT.md`](CIFAR10_EXPLANATION_CALIBRATION_RESULT.md)
- [`CIFAR10_TRAINING_SCALE_RESULT.md`](CIFAR10_TRAINING_SCALE_RESULT.md)
- [`MNIST_NEURAL_EXPLANATION_RESULT.md`](MNIST_NEURAL_EXPLANATION_RESULT.md)
- [`MNIST_NEURAL_CALIBRATION_RESULT.md`](MNIST_NEURAL_CALIBRATION_RESULT.md)
- [`GRADCAM_CALIBRATION_REFINEMENT_RESULT.md`](GRADCAM_CALIBRATION_REFINEMENT_RESULT.md)
