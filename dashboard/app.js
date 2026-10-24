const state = {
  report: null,
  selectedMethod: "SHAP",
  benchmarks: [],
  selectedBenchmark: 0,
  portfolio: null,
  robustness: null,
  uploadComparison: null,
  modelOverview: null,
  uploadValidation: null,
  regressionComparison: null,
};

const elements = {
  overallStatus: document.querySelector("#overall-status"),
  statusExplanation: document.querySelector("#status-explanation"),
  methodCount: document.querySelector("#method-count"),
  flagCount: document.querySelector("#flag-count"),
  agreementAverage: document.querySelector("#agreement-average"),
  methodCards: document.querySelector("#method-cards"),
  methodTabs: document.querySelector("#method-tabs"),
  metricGrid: document.querySelector("#metric-grid"),
  interpretation: document.querySelector("#interpretation-text"),
  upload: document.querySelector("#report-upload"),
  message: document.querySelector("#load-message"),
  viewTabs: document.querySelectorAll(".view-tab"),
  comparisonView: document.querySelector("#comparison-view"),
  benchmarkView: document.querySelector("#benchmark-view"),
  benchmarkSelector: document.querySelector("#benchmark-selector"),
  benchmarkContext: document.querySelector("#benchmark-context"),
  benchmarkSummary: document.querySelector("#benchmark-summary"),
  scenarioGrid: document.querySelector("#scenario-grid"),
  pairedResults: document.querySelector("#paired-results"),
  benchmarkInterpretation: document.querySelector("#benchmark-interpretation"),
  portfolioView: document.querySelector("#portfolio-view"),
  portfolioCoverage: document.querySelector("#portfolio-coverage"),
  portfolioGrid: document.querySelector("#portfolio-grid"),
  portfolioResults: document.querySelector("#portfolio-results"),
  portfolioInterpretation: document.querySelector("#portfolio-interpretation"),
  portfolioComparability: document.querySelector("#portfolio-comparability"),
  robustnessView: document.querySelector("#robustness-view"),
  robustnessStatus: document.querySelector("#robustness-status"),
  robustnessExplanation: document.querySelector("#robustness-explanation"),
  robustnessSummary: document.querySelector("#robustness-summary"),
  robustnessDatasetGrid: document.querySelector("#robustness-dataset-grid"),
  robustnessResults: document.querySelector("#robustness-results"),
  robustnessInterpretation: document.querySelector("#robustness-interpretation"),
  uploadView: document.querySelector("#upload-view"),
  uploadForm: document.querySelector("#classification-upload-form"),
  uploadTaskSelector: document.querySelector("#upload-task-selector"),
  modelV1Upload: document.querySelector("#model-v1-upload"),
  modelV2Upload: document.querySelector("#model-v2-upload"),
  additionalModelsUpload: document.querySelector("#additional-models-upload"),
  baselineModelSelector: document.querySelector("#baseline-model-selector"),
  candidateModelSelector: document.querySelector("#candidate-model-selector"),
  modelCollectionMessage: document.querySelector("#model-collection-message"),
  datasetUpload: document.querySelector("#dataset-upload"),
  manifestUpload: document.querySelector("#manifest-upload"),
  manifestBuilder: document.querySelector("#manifest-builder"),
  manifestBuilderMessage: document.querySelector("#manifest-builder-message"),
  manifestLabelColumn: document.querySelector("#manifest-label-column"),
  manifestIdColumn: document.querySelector("#manifest-id-column"),
  manifestImageColumn: document.querySelector("#manifest-image-column"),
  manifestImageHeight: document.querySelector("#manifest-image-height"),
  manifestImageWidth: document.querySelector("#manifest-image-width"),
  manifestTrustModels: document.querySelector("#manifest-trust-models"),
  imagesUpload: document.querySelector("#images-upload"),
  uploadSubmit: document.querySelector("#run-upload-comparison"),
  uploadMessage: document.querySelector("#upload-comparison-message"),
  uploadValidation: document.querySelector("#upload-validation"),
  uploadValidationSummary: document.querySelector("#upload-validation-summary"),
  uploadValidationGrid: document.querySelector("#upload-validation-grid"),
  uploadResults: document.querySelector("#upload-results"),
  regressionResults: document.querySelector("#regression-results"),
  regressionModelOverviewGrid: document.querySelector("#regression-model-overview-grid"),
  regressionScopeNote: document.querySelector("#regression-scope-note"),
  regressionV1Mae: document.querySelector("#regression-v1-mae"),
  regressionV2Mae: document.querySelector("#regression-v2-mae"),
  regressionV1Rmse: document.querySelector("#regression-v1-rmse"),
  regressionV2Rmse: document.querySelector("#regression-v2-rmse"),
  regressionV1R2: document.querySelector("#regression-v1-r2"),
  regressionV2R2: document.querySelector("#regression-v2-r2"),
  regressionEvidenceNote: document.querySelector("#regression-evidence-note"),
  regressionOutcomeGrid: document.querySelector("#regression-outcome-grid"),
  regressionParityChart: document.querySelector("#regression-parity-chart"),
  regressionErrorChart: document.querySelector("#regression-error-chart"),
  regressionImportanceResults: document.querySelector("#regression-importance-results"),
  regressionImportanceNote: document.querySelector("#regression-importance-note"),
  regressionOutcomeFilter: document.querySelector("#regression-outcome-filter"),
  regressionSampleResults: document.querySelector("#regression-sample-results"),
  regressionTableNote: document.querySelector("#regression-table-note"),
  modelOverviewGrid: document.querySelector("#model-overview-grid"),
  modelAgreementResults: document.querySelector("#model-agreement-results"),
  uploadTotal: document.querySelector("#upload-total"),
  uploadV1Accuracy: document.querySelector("#upload-v1-accuracy"),
  uploadV2Accuracy: document.querySelector("#upload-v2-accuracy"),
  uploadV1Label: document.querySelector("#upload-v1-label"),
  uploadV2Label: document.querySelector("#upload-v2-label"),
  uploadAgreement: document.querySelector("#upload-agreement"),
  pairedAccuracyChange: document.querySelector("#paired-accuracy-change"),
  pairedDiscordantCount: document.querySelector("#paired-discordant-count"),
  pairedPValue: document.querySelector("#paired-p-value"),
  pairedEvidenceStatus: document.querySelector("#paired-evidence-status"),
  pairedEvidenceNote: document.querySelector("#paired-evidence-note"),
  v1PredictionHeading: document.querySelector("#v1-prediction-heading"),
  v1ConfidenceHeading: document.querySelector("#v1-confidence-heading"),
  v2PredictionHeading: document.querySelector("#v2-prediction-heading"),
  v2ConfidenceHeading: document.querySelector("#v2-confidence-heading"),
  uploadScopeNote: document.querySelector("#upload-scope-note"),
  uploadTransitionGrid: document.querySelector("#upload-transition-grid"),
  classDiagnosticsResults: document.querySelector("#class-diagnostics-results"),
  classV1Heading: document.querySelector("#class-v1-heading"),
  classV2Heading: document.querySelector("#class-v2-heading"),
  confusionDiagnosticsResults: document.querySelector("#confusion-diagnostics-results"),
  confusionV1Heading: document.querySelector("#confusion-v1-heading"),
  confusionV2Heading: document.querySelector("#confusion-v2-heading"),
  confusionEmptyMessage: document.querySelector("#confusion-empty-message"),
  downloadComparisonJson: document.querySelector("#download-comparison-json"),
  downloadSamplesCsv: document.querySelector("#download-samples-csv"),
  uploadTransitionFilter: document.querySelector("#upload-transition-filter"),
  confidenceThresholdInput: document.querySelector("#confidence-threshold-input"),
  explanationThresholdInput: document.querySelector("#explanation-threshold-input"),
  confidenceRiskSummary: document.querySelector("#confidence-risk-summary"),
  uploadSampleResults: document.querySelector("#upload-sample-results"),
  uploadTableNote: document.querySelector("#upload-table-note"),
  uploadLimitations: document.querySelector("#upload-limitations"),
  explanationDialog: document.querySelector("#explanation-dialog"),
  explanationDialogTitle: document.querySelector("#explanation-dialog-title"),
  explanationMeta: document.querySelector("#explanation-meta"),
  explanationOriginal: document.querySelector("#explanation-original"),
  v1Heatmap: document.querySelector("#v1-heatmap"),
  v2Heatmap: document.querySelector("#v2-heatmap"),
  v1HeatmapCaption: document.querySelector("#v1-heatmap-caption"),
  v2HeatmapCaption: document.querySelector("#v2-heatmap-caption"),
  explanationBoundary: document.querySelector("#explanation-boundary"),
  closeExplanationDialog: document.querySelector("#close-explanation-dialog"),
};

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatScore(value) {
  return Number(value).toFixed(3);
}

function statusClass(status) {
  return status === "REVIEW" ? "review" : "within";
}

function validateReport(report) {
  if (!report || !Array.isArray(report.methods) || !Array.isArray(report.metrics)) {
    throw new Error("This file is not a unified XAI report.");
  }
  if (!report.methods.length || !report.metrics.length) {
    throw new Error("The report does not contain method or metric results.");
  }
  return report;
}

function validateBenchmark(report) {
  if (
    !report ||
    !Array.isArray(report.scenario_results) ||
    !Array.isArray(report.paired_control_comparisons)
  ) {
    throw new Error("This file is not a benchmark summary.");
  }
  if (!report.scenario_results.length) {
    throw new Error("The benchmark does not contain scenario results.");
  }
  return report;
}

function validatePortfolio(report) {
  if (
    !report ||
    !Array.isArray(report.entries) ||
    !report.coverage ||
    typeof report.coverage !== "object"
  ) {
    throw new Error("This file is not a cross-dataset summary.");
  }
  if (!report.entries.length) {
    throw new Error("The cross-dataset summary contains no evidence entries.");
  }
  return report;
}

function validateRobustness(report) {
  if (
    !report ||
    !Array.isArray(report.dataset_summaries) ||
    !Array.isArray(report.split_results) ||
    !Array.isArray(report.split_seeds)
  ) {
    throw new Error("This file is not a split-robustness summary.");
  }
  if (!report.dataset_summaries.length || !report.split_results.length) {
    throw new Error("The split-robustness summary contains no results.");
  }
  return report;
}

function datasetLabel(report) {
  if (report.dataset === "wine_binary") return "Wine (binary)";
  if (report.dataset === "breast_cancer" || !report.dataset) return "Breast cancer";
  return String(report.dataset).replaceAll("_", " ");
}

function scenarioLabel(id) {
  const labels = {
    seed_only_control: "Seed-only control",
    shallow_depth: "Shallow trees",
    larger_leaves: "Larger leaves",
    constrained_candidate: "Combined constraints",
  };
  return labels[id] || String(id).replaceAll("_", " ");
}

function switchView(view) {
  const benchmark = view === "benchmark";
  const portfolio = view === "portfolio";
  const robustness = view === "robustness";
  const upload = view === "upload";
  elements.comparisonView.hidden = benchmark || portfolio || robustness || upload;
  elements.benchmarkView.hidden = !benchmark;
  elements.portfolioView.hidden = !portfolio;
  elements.robustnessView.hidden = !robustness;
  elements.uploadView.hidden = !upload;
  for (const tab of elements.viewTabs) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
}

const transitionLabels = {
  both_correct: "Both correct",
  v1_correct_v2_wrong: "Correct → wrong",
  v1_wrong_v2_correct: "Wrong → correct",
  both_wrong: "Both wrong",
};

function formatConfidence(value) {
  return value === null || value === undefined ? "Not available" : formatPercent(value);
}

function formatSimilarity(value) {
  return value === null || value === undefined ? "Not available" : Number(value).toFixed(3);
}

function formatNumber(value) {
  return value === null || value === undefined ? "Not available" : Number(value).toFixed(4);
}

function triageThresholds() {
  const confidenceValue = Number(elements.confidenceThresholdInput.value);
  const explanationValue = Number(elements.explanationThresholdInput.value);
  const confidence = Math.min(100, Math.max(50, Number.isFinite(confidenceValue) ? confidenceValue : 80)) / 100;
  const explanation = Math.min(100, Math.max(0, Number.isFinite(explanationValue) ? explanationValue : 80)) / 100;
  return { confidence, explanation };
}

function sampleTriage(sample) {
  const thresholds = triageThresholds();
  const v1HighConfidenceWrong = sample.v1_correct === false
    && sample.v1_confidence !== null && sample.v1_confidence >= thresholds.confidence;
  const v2HighConfidenceWrong = sample.v2_correct === false
    && sample.v2_confidence !== null && sample.v2_confidence >= thresholds.confidence;
  const explanationDrift = sample.explanation_similarity !== null
    && sample.explanation_similarity !== undefined
    && sample.explanation_similarity < thresholds.explanation;
  return {
    v1_high_confidence_wrong: v1HighConfidenceWrong,
    v2_high_confidence_wrong: v2HighConfidenceWrong,
    high_confidence_regression: v2HighConfidenceWrong && sample.transition === "v1_correct_v2_wrong",
    explanation_drift: explanationDrift,
    hidden_explanation_drift: explanationDrift && sample.prediction_changed === false,
  };
}

function uploadedModels() {
  const files = [];
  if (elements.modelV1Upload.files[0]) files.push(elements.modelV1Upload.files[0]);
  if (elements.modelV2Upload.files[0]) files.push(elements.modelV2Upload.files[0]);
  files.push(...Array.from(elements.additionalModelsUpload.files).slice(0, 3));
  return files;
}

function parseCsvHeader(text) {
  const columns = [];
  let value = "";
  let quoted = false;
  let headerEnded = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      columns.push(value.trim());
      value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      columns.push(value.trim());
      headerEnded = true;
      break;
    } else {
      value += character;
    }
  }
  if (!headerEnded && value) columns.push(value.trim());
  return columns.filter(Boolean);
}

function fillColumnSelector(selector, columns, optional, preferredNames = []) {
  selector.replaceChildren();
  if (optional) {
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "None";
    selector.append(none);
  }
  for (const column of columns) {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    selector.append(option);
  }
  const preferred = columns.find((column) => preferredNames.includes(column.toLowerCase()));
  if (preferred) selector.value = preferred;
}

async function updateManifestBuilder() {
  const file = elements.datasetUpload.files[0];
  if (!file) return;
  const columns = parseCsvHeader(await file.slice(0, 65536).text());
  if (columns.length < 2) {
    elements.manifestBuilderMessage.textContent = "Could not detect at least two CSV columns.";
    return;
  }
  fillColumnSelector(elements.manifestLabelColumn, columns, false);
  elements.manifestLabelColumn.value = columns.find(
    (column) => ["target", "label", "class", "y"].includes(column.toLowerCase()),
  ) || columns[columns.length - 1];
  fillColumnSelector(elements.manifestIdColumn, columns, true, ["id", "sample_id"]);
  fillColumnSelector(elements.manifestImageColumn, columns, true, ["image", "image_path", "filename"]);
  elements.manifestBuilderMessage.textContent = `${columns.length} columns detected. Confirm the roles below or upload a JSON manifest.`;
  elements.manifestBuilder.open = true;
}

function selectedManifestFile() {
  if (elements.manifestUpload.files[0]) return elements.manifestUpload.files[0];
  const labelColumn = elements.manifestLabelColumn.value;
  if (!labelColumn) throw new Error("Choose the CSV label column or upload a manifest.");
  if (!elements.manifestTrustModels.checked) {
    throw new Error("Confirm that the uploaded joblib model files are trusted.");
  }
  const allColumns = Array.from(elements.manifestLabelColumn.options).map((option) => option.value);
  const idColumn = elements.manifestIdColumn.value || null;
  const isRegression = elements.uploadTaskSelector.value === "regression";
  const imageColumn = isRegression ? null : elements.manifestImageColumn.value || null;
  const excluded = new Set([labelColumn, idColumn, imageColumn].filter(Boolean));
  const manifest = {
    task: isRegression ? "regression" : "classification",
    label_column: labelColumn,
    id_column: idColumn,
    image_column: imageColumn,
    feature_columns: allColumns.filter((column) => !excluded.has(column)),
    trusted_model_files: true,
  };
  const height = isRegression ? 0 : Number(elements.manifestImageHeight.value);
  const width = isRegression ? 0 : Number(elements.manifestImageWidth.value);
  if (height || width) {
    if (!height || !width) throw new Error("Enter both image height and image width.");
    manifest.image_shape = [height, width];
  }
  if (imageColumn && !elements.imagesUpload.files[0]) {
    throw new Error("Upload the image ZIP or set the image filename column to None.");
  }
  return new File(
    [JSON.stringify(manifest, null, 2)],
    "generated-manifest.json",
    { type: "application/json" },
  );
}

function updateUploadTask() {
  const isRegression = elements.uploadTaskSelector.value === "regression";
  for (const element of document.querySelectorAll(".classification-only")) {
    element.hidden = isRegression;
  }
  elements.uploadResults.hidden = true;
  elements.regressionResults.hidden = true;
  elements.uploadValidation.hidden = true;
  elements.uploadMessage.textContent = isRegression
    ? "Upload two regressors and a numeric labelled CSV."
    : "Upload two to five classifiers and a labelled CSV.";
}

function updateModelSelectors() {
  const files = uploadedModels();
  const previousBaseline = elements.baselineModelSelector.value;
  const previousCandidate = elements.candidateModelSelector.value;
  for (const selector of [elements.baselineModelSelector, elements.candidateModelSelector]) {
    selector.replaceChildren();
    files.forEach((file, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = file.name;
      selector.append(option);
    });
  }
  if (previousBaseline !== "" && files[Number(previousBaseline)]) {
    elements.baselineModelSelector.value = previousBaseline;
  } else if (files.length) {
    elements.baselineModelSelector.value = "0";
  }
  if (previousCandidate !== "" && files[Number(previousCandidate)]) {
    elements.candidateModelSelector.value = previousCandidate;
  } else if (files.length > 1) {
    elements.candidateModelSelector.value = "1";
  }
  const extraCount = elements.additionalModelsUpload.files.length;
  elements.modelCollectionMessage.textContent = extraCount > 3
    ? "Only the first three additional models will be available (five total)."
    : `${files.length} model${files.length === 1 ? "" : "s"} ready; select two different models.`;
}

function reportModelName(key, fallback) {
  return state.uploadComparison?.models?.[key]?.filename || fallback;
}

function renderReliabilityChart(model) {
  const chart = document.createElement("div");
  chart.className = "reliability-chart";
  chart.setAttribute("aria-label", `Reliability chart for ${model.filename}`);
  if (!model.supports_confidence || !Array.isArray(model.calibration_bins) || !model.calibration_bins.length) {
    chart.textContent = "Probability calibration is unavailable.";
    chart.classList.add("unavailable");
    return chart;
  }
  const legend = document.createElement("div");
  legend.className = "reliability-legend";
  legend.textContent = "Confidence / observed accuracy";
  chart.append(legend);
  for (const bin of model.calibration_bins) {
    const row = document.createElement("div");
    row.className = "reliability-row";
    row.title = `${formatPercent(bin.lower)}–${formatPercent(bin.upper)} · ${bin.samples} sample${bin.samples === 1 ? "" : "s"}`;
    const label = document.createElement("span");
    label.textContent = `${Math.round(bin.upper * 100)}%`;
    const tracks = document.createElement("div");
    tracks.className = "reliability-tracks";
    const confidenceBar = document.createElement("span");
    confidenceBar.className = "confidence-bar";
    confidenceBar.style.width = formatPercent(bin.mean_confidence);
    const accuracyBar = document.createElement("span");
    accuracyBar.className = "accuracy-bar";
    accuracyBar.style.width = formatPercent(bin.accuracy);
    tracks.append(confidenceBar, accuracyBar);
    const count = document.createElement("span");
    count.textContent = `n=${bin.samples}`;
    row.append(label, tracks, count);
    chart.append(row);
  }
  return chart;
}

function renderModelOverview(report) {
  if (!report || !Array.isArray(report.models) || !Array.isArray(report.pairwise_agreements)) {
    throw new Error("The server returned an invalid multi-model overview.");
  }
  state.modelOverview = report;
  elements.modelOverviewGrid.replaceChildren();
  const models = [...report.models].sort((left, right) => left.rank - right.rank);
  for (const model of models) {
    const card = document.createElement("article");
    card.className = `model-overview-card${model.index === report.best_model_index ? " best" : ""}`;
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = model.index === report.best_model_index ? `Rank ${model.rank} · Best accuracy` : `Rank ${model.rank}`;
    const name = document.createElement("strong");
    name.textContent = model.filename;
    const accuracy = document.createElement("span");
    accuracy.className = "accuracy";
    accuracy.textContent = formatPercent(model.accuracy);
    const count = document.createElement("small");
    count.textContent = `${model.correct_count} of ${report.dataset.rows} samples correct`;
    const calibration = document.createElement("div");
    calibration.className = "calibration-summary";
    if (model.supports_confidence) {
      const gapDirection = model.confidence_gap > 0 ? "overconfident" : model.confidence_gap < 0 ? "underconfident" : "aligned";
      calibration.textContent = `Mean confidence ${formatPercent(model.mean_confidence)} · ECE ${formatPercent(model.expected_calibration_error)} · ${gapDirection} by ${formatPercent(Math.abs(model.confidence_gap))}`;
    } else {
      calibration.textContent = "Calibration unavailable: model has no probability output";
    }
    card.append(rank, name, accuracy, count, calibration, renderReliabilityChart(model));
    elements.modelOverviewGrid.append(card);
  }
  elements.modelAgreementResults.replaceChildren();
  for (const pair of report.pairwise_agreements) {
    const row = document.createElement("tr");
    const names = document.createElement("td");
    names.textContent = `${report.models[pair.left_index].filename} ↔ ${report.models[pair.right_index].filename}`;
    const agreement = document.createElement("td");
    agreement.textContent = formatPercent(pair.agreement);
    row.append(names, agreement);
    elements.modelAgreementResults.append(row);
  }
}

function renderRegressionModelOverview(report) {
  if (!report || report.task !== "regression" || !Array.isArray(report.models)) {
    throw new Error("The server returned an invalid regression model overview.");
  }
  elements.regressionModelOverviewGrid.replaceChildren();
  const models = [...report.models].sort((left, right) => left.rank - right.rank);
  for (const model of models) {
    const card = document.createElement("article");
    card.className = `model-overview-card${model.index === report.best_model_index ? " best" : ""}`;
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = model.index === report.best_model_index ? `Rank ${model.rank} · Lowest MAE` : `Rank ${model.rank}`;
    const name = document.createElement("strong");
    name.textContent = model.filename;
    const mae = document.createElement("span");
    mae.className = "accuracy";
    mae.textContent = `MAE ${formatNumber(model.mae)}`;
    const metrics = document.createElement("small");
    metrics.textContent = `RMSE ${formatNumber(model.rmse)} · R² ${formatNumber(model.r2)}`;
    card.append(rank, name, mae, metrics);
    elements.regressionModelOverviewGrid.append(card);
  }
}

function renderUploadValidation(report) {
  if (!report || !Array.isArray(report.models) || typeof report.ready !== "boolean") {
    throw new Error("The server returned an invalid compatibility report.");
  }
  state.uploadValidation = report;
  elements.uploadValidation.classList.toggle("ready", report.ready);
  elements.uploadValidation.classList.toggle("blocked", !report.ready);
  elements.uploadValidationSummary.textContent = report.ready
    ? `All ${report.models.length} models accept ${report.dataset.feature_count} features and are ready.`
    : "At least one model is incompatible. Resolve the highlighted item before comparing.";
  elements.uploadValidationGrid.replaceChildren();
  for (const model of report.models) {
    const card = document.createElement("article");
    card.className = `validation-card ${model.compatible ? "compatible" : "incompatible"}`;
    const name = document.createElement("strong");
    name.textContent = model.filename;
    const status = document.createElement("span");
    status.className = "validation-status";
    status.textContent = model.compatible ? "Compatible" : "Needs attention";
    const detail = document.createElement("span");
    detail.textContent = model.compatible
      ? `${model.model_type}${model.supports_confidence ? " · confidence available" : " · no probability confidence"}`
      : model.message;
    card.append(name, status, detail);
    elements.uploadValidationGrid.append(card);
  }
  elements.uploadValidation.hidden = false;
}

function downloadLocalFile(filename, content, mediaType) {
  const blob = new Blob([content], { type: mediaType });
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function comparisonForExport() {
  const thresholds = triageThresholds();
  const samples = state.uploadComparison.samples.map((sample) => ({
    ...sample,
    ...sampleTriage(sample),
  }));
  return {
    ...state.uploadComparison,
    summary: {
      ...state.uploadComparison.summary,
      high_confidence_threshold: thresholds.confidence,
      explanation_similarity_threshold: thresholds.explanation,
      v2_high_confidence_wrong_count: samples.filter((sample) => sample.v2_high_confidence_wrong).length,
      high_confidence_regression_count: samples.filter((sample) => sample.high_confidence_regression).length,
      explanation_drift_count: samples.filter((sample) => sample.explanation_drift).length,
      hidden_explanation_drift_count: samples.filter((sample) => sample.hidden_explanation_drift).length,
    },
    samples,
  };
}

function downloadComparisonJson() {
  if (!state.uploadComparison || !state.modelOverview) return;
  const artifact = {
    triage_settings: triageThresholds(),
    exported_at: new Date().toISOString(),
    validation: state.uploadValidation,
    overview: state.modelOverview,
    comparison: comparisonForExport(),
  };
  downloadLocalFile(
    `xai-comparison-${exportTimestamp()}.json`,
    JSON.stringify(artifact, null, 2),
    "application/json",
  );
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadSamplesCsv() {
  if (!state.uploadComparison) return;
  const columns = [
    "v1_model", "v2_model", "sample_id", "actual", "v1_prediction", "v1_confidence",
    "v2_prediction", "v2_confidence", "transition", "prediction_changed",
    "v1_high_confidence_wrong", "v2_high_confidence_wrong", "high_confidence_regression",
    "explanation_similarity", "explanation_drift", "hidden_explanation_drift",
  ];
  const rows = state.uploadComparison.samples.map((sample) => {
    const record = {
      ...sample,
      ...sampleTriage(sample),
      v1_model: reportModelName("v1", "Model V1"),
      v2_model: reportModelName("v2", "Model V2"),
    };
    return columns.map((column) => csvCell(record[column])).join(",");
  });
  downloadLocalFile(
    `xai-samples-${exportTimestamp()}.csv`,
    `${columns.join(",")}\n${rows.join("\n")}\n`,
    "text/csv;charset=utf-8",
  );
}

function renderPixelHeatmap(container, values) {
  container.replaceChildren();
  if (!Array.isArray(values) || !values.length || !Array.isArray(values[0])) {
    container.textContent = "Explanation unavailable";
    container.classList.add("unavailable");
    return;
  }
  container.classList.remove("unavailable");
  container.style.gridTemplateColumns = `repeat(${values[0].length}, 1fr)`;
  for (const row of values) {
    for (const rawValue of row) {
      const value = Number(rawValue);
      const cell = document.createElement("span");
      const intensity = Math.min(1, Math.abs(value));
      cell.style.background = value >= 0
        ? `rgba(239, 100, 54, ${0.08 + intensity * 0.92})`
        : `rgba(40, 125, 174, ${0.08 + intensity * 0.92})`;
      cell.title = `Normalized contribution: ${value.toFixed(3)}`;
      container.append(cell);
    }
  }
}

function openExplanation(sample) {
  elements.explanationDialogTitle.textContent = `Sample ${sample.sample_id}`;
  elements.explanationMeta.replaceChildren();
  const facts = [
    `Actual: ${sample.actual}`,
    `V1: ${sample.v1_prediction} (${formatConfidence(sample.v1_confidence)})`,
    `V2: ${sample.v2_prediction} (${formatConfidence(sample.v2_confidence)})`,
    transitionLabels[sample.transition] || sample.transition,
    `Explanation similarity: ${formatSimilarity(sample.explanation_similarity)}`,
  ];
  for (const fact of facts) {
    const item = document.createElement("span");
    item.textContent = fact;
    elements.explanationMeta.append(item);
  }
  elements.explanationOriginal.src = sample.image_data_url || "";
  elements.explanationOriginal.hidden = !sample.image_data_url;
  renderPixelHeatmap(elements.v1Heatmap, sample.v1_explanation);
  renderPixelHeatmap(elements.v2Heatmap, sample.v2_explanation);
  const v1Method = state.uploadComparison.explanations?.v1?.label || "Explanation";
  const v2Method = state.uploadComparison.explanations?.v2?.label || "Explanation";
  elements.v1HeatmapCaption.textContent = `${reportModelName("v1", "Model V1")} · ${v1Method} toward “${sample.v1_prediction}”`;
  elements.v2HeatmapCaption.textContent = `${reportModelName("v2", "Model V2")} · ${v2Method} toward “${sample.v2_prediction}”`;
  elements.v1Heatmap.title = state.uploadComparison.explanations?.v1?.note || "";
  elements.v2Heatmap.title = state.uploadComparison.explanations?.v2?.note || "";
  elements.explanationBoundary.textContent = state.uploadComparison.explanations?.interpretation
    || "The map describes feature contributions, not causal effects.";
  elements.explanationDialog.showModal();
}

function renderUploadSamples() {
  const report = state.uploadComparison;
  if (!report) return;
  const selected = elements.uploadTransitionFilter.value;
  const matching = report.samples.filter((sample) => {
    const triage = sampleTriage(sample);
    if (selected === "all") return true;
    if (selected === "v2_high_confidence_wrong") return triage.v2_high_confidence_wrong;
    if (selected === "high_confidence_regression") return triage.high_confidence_regression;
    if (selected === "explanation_drift") return triage.explanation_drift;
    if (selected === "hidden_explanation_drift") return triage.hidden_explanation_drift;
    return sample.transition === selected;
  });
  const displayed = matching.slice(0, 500);
  elements.uploadSampleResults.replaceChildren();
  for (const sample of displayed) {
    const triage = sampleTriage(sample);
    const row = document.createElement("tr");
    if (sample.transition === "v1_correct_v2_wrong") row.className = "regression-row";
    if (sample.transition === "v1_wrong_v2_correct") row.className = "improvement-row";
    if (triage.v2_high_confidence_wrong) row.classList.add("high-confidence-risk-row");
    if (triage.hidden_explanation_drift) row.classList.add("explanation-drift-row");
    const imageCell = document.createElement("td");
    if (sample.image_data_url) {
      const link = document.createElement("a");
      link.href = sample.image_data_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.title = `Open ${sample.image_name || "sample image"}`;
      const thumbnail = document.createElement("img");
      thumbnail.className = "sample-thumbnail";
      thumbnail.src = sample.image_data_url;
      thumbnail.alt = `Sample ${sample.sample_id}`;
      link.append(thumbnail);
      imageCell.append(link);
    } else {
      imageCell.textContent = "—";
    }
    row.append(imageCell);
    const values = [
      sample.sample_id,
      sample.actual,
      sample.v1_prediction,
      formatConfidence(sample.v1_confidence),
      sample.v2_prediction,
      formatConfidence(sample.v2_confidence),
      formatSimilarity(sample.explanation_similarity),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    const transitionCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `transition-pill ${sample.transition}`;
    badge.textContent = transitionLabels[sample.transition] || sample.transition;
    transitionCell.append(badge);
    row.append(transitionCell);
    const explanationCell = document.createElement("td");
    const explanationButton = document.createElement("button");
    explanationButton.className = "explanation-button";
    explanationButton.type = "button";
    const available = Array.isArray(sample.v1_explanation) || Array.isArray(sample.v2_explanation);
    explanationButton.textContent = available ? "Compare maps" : "Unavailable";
    explanationButton.disabled = !available;
    if (available) explanationButton.addEventListener("click", () => openExplanation(sample));
    explanationCell.append(explanationButton);
    row.append(explanationCell);
    elements.uploadSampleResults.append(row);
  }
  elements.uploadTableNote.textContent = matching.length > displayed.length
    ? `Showing the first ${displayed.length} of ${matching.length} matching samples.`
    : `Showing ${matching.length} matching sample${matching.length === 1 ? "" : "s"}.`;
}

function renderTriageSummary() {
  if (!state.uploadComparison) return;
  const report = state.uploadComparison;
  const thresholds = triageThresholds();
  const triage = report.samples.map(sampleTriage);
  const candidateRiskCount = triage.filter((item) => item.v2_high_confidence_wrong).length;
  const regressionRiskCount = triage.filter((item) => item.high_confidence_regression).length;
  const hiddenDriftCount = triage.filter((item) => item.hidden_explanation_drift).length;
  const explanationCount = report.samples.filter(
    (sample) => sample.explanation_similarity !== null && sample.explanation_similarity !== undefined,
  ).length;
  const v1Name = report.models?.v1?.filename || "Model V1";
  const v2Name = report.models?.v2?.filename || "Model V2";
  elements.confidenceRiskSummary.textContent =
    `${v2Name} made ${candidateRiskCount} wrong prediction${candidateRiskCount === 1 ? "" : "s"} `
    + `at or above ${formatPercent(thresholds.confidence)} confidence; ${regressionRiskCount} were regressions from a correct ${v1Name} result. `
    + `${explanationCount} samples had comparable explanation maps; ${hiddenDriftCount} kept the same prediction with similarity below ${formatPercent(thresholds.explanation)}.`;
}

function renderUploadComparison(report) {
  if (
    !report || report.task !== "classification" || !report.summary
    || !Array.isArray(report.samples) || !Array.isArray(report.class_diagnostics)
    || !Array.isArray(report.confusion_diagnostics)
  ) {
    throw new Error("The server returned an invalid classification comparison.");
  }
  state.uploadComparison = report;
  const summary = report.summary;
  const v1Name = report.models?.v1?.filename || "Model V1";
  const v2Name = report.models?.v2?.filename || "Model V2";
  elements.uploadTotal.textContent = summary.total_samples;
  elements.uploadV1Accuracy.textContent = formatPercent(summary.v1_accuracy);
  elements.uploadV2Accuracy.textContent = formatPercent(summary.v2_accuracy);
  elements.uploadAgreement.textContent = formatPercent(summary.prediction_agreement);
  elements.pairedAccuracyChange.textContent = `${summary.accuracy_change > 0 ? "+" : ""}${formatPercent(summary.accuracy_change)}`;
  elements.pairedDiscordantCount.textContent = summary.discordant_correctness_count;
  elements.pairedPValue.textContent = summary.mcnemar_exact_p < 0.001
    ? Number(summary.mcnemar_exact_p).toExponential(2)
    : Number(summary.mcnemar_exact_p).toFixed(3);
  const pairedDirection = summary.accuracy_change > 0 ? "improvement" : summary.accuracy_change < 0 ? "regression" : "difference";
  elements.pairedEvidenceStatus.textContent = summary.mcnemar_significant
    ? `Evidence of ${pairedDirection}`
    : "Exploratory";
  elements.pairedEvidenceStatus.className = summary.mcnemar_significant ? "evidence-strong" : "evidence-exploratory";
  elements.pairedEvidenceNote.textContent = summary.mcnemar_significant
    ? `The paired difference is significant at α=${summary.mcnemar_alpha}. This describes the uploaded labelled sample.`
    : `The paired difference did not reach α=${summary.mcnemar_alpha}. Treat the observed change as exploratory.`;
  elements.uploadV1Label.textContent = `${v1Name} accuracy`;
  elements.uploadV2Label.textContent = `${v2Name} accuracy`;
  elements.v1PredictionHeading.textContent = `${v1Name} prediction`;
  elements.v1ConfidenceHeading.textContent = `${v1Name} confidence`;
  elements.v2PredictionHeading.textContent = `${v2Name} prediction`;
  elements.v2ConfidenceHeading.textContent = `${v2Name} confidence`;
  elements.classV1Heading.textContent = `${v1Name} accuracy`;
  elements.classV2Heading.textContent = `${v2Name} accuracy`;
  elements.confusionV1Heading.textContent = `${v1Name} errors`;
  elements.confusionV2Heading.textContent = `${v2Name} errors`;
  elements.uploadScopeNote.textContent = report.dataset?.note
    || "Accuracy summarizes the supplied samples; the table shows the concrete changes behind it.";
  elements.confidenceThresholdInput.value = String((summary.high_confidence_threshold ?? 0.8) * 100);
  elements.explanationThresholdInput.value = String((summary.explanation_similarity_threshold ?? 0.8) * 100);
  renderTriageSummary();

  const transitions = [
    ["both_correct", "Both correct"],
    ["v1_wrong_v2_correct", "Wrong → correct"],
    ["v1_correct_v2_wrong", "Correct → wrong"],
    ["both_wrong", "Both wrong"],
  ];
  elements.uploadTransitionGrid.replaceChildren();
  for (const [key, label] of transitions) {
    const card = document.createElement("article");
    card.className = `transition-card ${key}`;
    const name = document.createElement("span");
    const value = document.createElement("strong");
    name.textContent = label;
    value.textContent = summary[key];
    card.append(name, value);
    elements.uploadTransitionGrid.append(card);
  }

  elements.classDiagnosticsResults.replaceChildren();
  const classDiagnostics = [...(report.class_diagnostics || [])].sort(
    (left, right) => right.accuracy_change - left.accuracy_change,
  );
  for (const item of classDiagnostics) {
    const row = document.createElement("tr");
    for (const value of [
      item.class_label,
      item.samples,
      formatPercent(item.v1_accuracy),
      formatPercent(item.v2_accuracy),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    const changeCell = document.createElement("td");
    const change = document.createElement("span");
    const direction = item.accuracy_change > 0
      ? "improved"
      : item.accuracy_change < 0 ? "regressed" : "unchanged";
    change.className = `class-change ${direction}`;
    change.textContent = `${item.accuracy_change > 0 ? "+" : ""}${formatPercent(item.accuracy_change)}`;
    changeCell.append(change);
    row.append(changeCell);
    elements.classDiagnosticsResults.append(row);
  }

  elements.confusionDiagnosticsResults.replaceChildren();
  elements.confusionEmptyMessage.hidden = report.confusion_diagnostics.length > 0;
  for (const item of report.confusion_diagnostics) {
    const row = document.createElement("tr");
    for (const value of [
      item.actual_class,
      item.predicted_class,
      item.v1_count,
      item.v2_count,
    ]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    const changeCell = document.createElement("td");
    const change = document.createElement("span");
    const direction = item.error_change < 0
      ? "reduced"
      : item.error_change > 0 ? "increased" : "unchanged";
    change.className = `error-change ${direction}`;
    change.textContent = `${item.error_change > 0 ? "+" : ""}${item.error_change}`;
    changeCell.append(change);
    row.append(changeCell);
    elements.confusionDiagnosticsResults.append(row);
  }

  elements.uploadLimitations.replaceChildren();
  for (const limitation of report.limitations || []) {
    const item = document.createElement("li");
    item.textContent = limitation;
    elements.uploadLimitations.append(item);
  }
  elements.uploadTransitionFilter.value = "all";
  renderUploadSamples();
  elements.uploadResults.hidden = false;
  elements.uploadResults.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderRegressionSamples() {
  if (!state.regressionComparison) return;
  const selected = elements.regressionOutcomeFilter.value;
  const matching = state.regressionComparison.samples.filter(
    (sample) => selected === "all" || sample.outcome === selected,
  );
  elements.regressionSampleResults.replaceChildren();
  for (const sample of matching.slice(0, 500)) {
    const row = document.createElement("tr");
    row.className = `${sample.outcome}-row`;
    const values = [
      sample.sample_id,
      formatNumber(sample.actual),
      formatNumber(sample.v1_prediction),
      formatNumber(sample.v1_absolute_error),
      formatNumber(sample.v2_prediction),
      formatNumber(sample.v2_absolute_error),
      `${sample.absolute_error_change > 0 ? "+" : ""}${formatNumber(sample.absolute_error_change)}`,
      sample.outcome,
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    elements.regressionSampleResults.append(row);
  }
  elements.regressionTableNote.textContent = matching.length > 500
    ? `Showing the first 500 of ${matching.length} matching samples.`
    : `Showing ${matching.length} matching sample${matching.length === 1 ? "" : "s"}.`;
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function regressionChartSamples(samples, limit = 180) {
  if (samples.length <= limit) return samples;
  const step = (samples.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, index) => samples[Math.round(index * step)]);
}

function renderRegressionCharts(samples) {
  const width = 560;
  const height = 300;
  const padding = 38;
  const plotted = regressionChartSamples(samples);
  const values = plotted.flatMap((sample) => [sample.actual, sample.v1_prediction, sample.v2_prediction]);
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const scaleX = (value) => padding + ((value - minimum) / (maximum - minimum)) * (width - padding * 2);
  const scaleY = (value) => height - padding - ((value - minimum) / (maximum - minimum)) * (height - padding * 2);

  elements.regressionParityChart.replaceChildren();
  elements.regressionParityChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.regressionParityChart.append(
    svgElement("line", { x1: padding, y1: height - padding, x2: width - padding, y2: padding, class: "parity-reference" }),
  );
  for (const sample of plotted) {
    elements.regressionParityChart.append(
      svgElement("circle", { cx: scaleX(sample.actual), cy: scaleY(sample.v1_prediction), r: 3.2, class: "parity-v1" }),
      svgElement("circle", { cx: scaleX(sample.actual), cy: scaleY(sample.v2_prediction), r: 3.2, class: "parity-v2" }),
    );
  }
  const parityLabels = [
    [padding, height - 12, `Actual ${formatNumber(minimum)}`],
    [width - padding, height - 12, `Actual ${formatNumber(maximum)}`],
    [8, padding + 4, `Pred. ${formatNumber(maximum)}`],
  ];
  for (const [x, y, label] of parityLabels) {
    const text = svgElement("text", { x, y, class: "chart-axis-label" });
    text.textContent = label;
    elements.regressionParityChart.append(text);
  }

  const ordered = [...samples].sort((a, b) => a.absolute_error_change - b.absolute_error_change);
  const errorSamples = regressionChartSamples(ordered);
  const largestChange = Math.max(1e-12, ...errorSamples.map((sample) => Math.abs(sample.absolute_error_change)));
  const zeroY = height / 2;
  const barWidth = (width - padding * 2) / Math.max(errorSamples.length, 1);
  elements.regressionErrorChart.replaceChildren();
  elements.regressionErrorChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.regressionErrorChart.append(svgElement("line", { x1: padding, y1: zeroY, x2: width - padding, y2: zeroY, class: "error-zero-line" }));
  errorSamples.forEach((sample, index) => {
    const magnitude = Math.abs(sample.absolute_error_change) / largestChange * (height / 2 - padding);
    const y = sample.absolute_error_change >= 0 ? zeroY - magnitude : zeroY;
    elements.regressionErrorChart.append(svgElement("rect", {
      x: padding + index * barWidth,
      y,
      width: Math.max(1, barWidth - 0.7),
      height: Math.max(1, magnitude),
      class: sample.outcome === "improved" ? "error-improved" : sample.outcome === "worsened" ? "error-worsened" : "error-unchanged",
    }));
  });
  for (const [y, label] of [[padding + 4, `+${formatNumber(largestChange)}`], [zeroY - 6, "0"], [height - padding, `−${formatNumber(largestChange)}`]]) {
    const text = svgElement("text", { x: 5, y, class: "chart-axis-label" });
    text.textContent = label;
    elements.regressionErrorChart.append(text);
  }
}

function renderRegressionImportance(report) {
  const importance = report.feature_importance;
  if (!importance || !Array.isArray(importance.features)) {
    elements.regressionImportanceResults.replaceChildren();
    elements.regressionImportanceNote.textContent = "Permutation importance was not returned for this comparison.";
    return;
  }
  const visible = importance.features.slice(0, 15);
  const maximum = Math.max(1e-12, ...visible.flatMap((item) => [Math.abs(item.v1_importance), Math.abs(item.v2_importance)]));
  elements.regressionImportanceResults.replaceChildren();
  for (const item of visible) {
    const row = document.createElement("article");
    row.className = "regression-importance-row";
    const feature = document.createElement("strong");
    feature.textContent = item.feature;
    const bars = document.createElement("div");
    bars.className = "importance-bars";
    for (const [version, value] of [["V1", item.v1_importance], ["V2", item.v2_importance]]) {
      const line = document.createElement("div");
      const label = document.createElement("span");
      label.textContent = version;
      const track = document.createElement("div");
      track.className = "importance-track";
      const fill = document.createElement("i");
      fill.className = version === "V1" ? "importance-v1" : "importance-v2";
      fill.style.width = `${Math.max(0, Math.abs(value) / maximum * 100)}%`;
      track.append(fill);
      const number = document.createElement("em");
      number.textContent = formatNumber(value);
      line.append(label, track, number);
      bars.append(line);
    }
    row.append(feature, bars);
    elements.regressionImportanceResults.append(row);
  }
  const scope = importance.truncated
    ? `The first ${importance.evaluated_feature_count} features were evaluated; showing the top ${visible.length}.`
    : `${importance.evaluated_feature_count} features were evaluated; showing the top ${visible.length}.`;
  elements.regressionImportanceNote.textContent = `${scope} Importance is the average MAE increase over ${importance.repeats} deterministic shuffles.`;
}

function renderRegressionComparison(report) {
  if (!report || report.task !== "regression" || !report.summary || !Array.isArray(report.samples)) {
    throw new Error("The server returned an invalid regression comparison.");
  }
  state.regressionComparison = report;
  const summary = report.summary;
  elements.regressionV1Mae.textContent = formatNumber(summary.v1.mae);
  elements.regressionV2Mae.textContent = formatNumber(summary.v2.mae);
  elements.regressionV1Rmse.textContent = formatNumber(summary.v1.rmse);
  elements.regressionV2Rmse.textContent = formatNumber(summary.v2.rmse);
  elements.regressionV1R2.textContent = formatNumber(summary.v1.r2);
  elements.regressionV2R2.textContent = formatNumber(summary.v2.r2);
  elements.regressionScopeNote.textContent = report.dataset.note
    || `${summary.total_samples} labelled samples; negative MAE change favors V2.`;
  const outcomes = [
    ["improved", "Improved", summary.improved_count],
    ["worsened", "Worsened", summary.worsened_count],
    ["unchanged", "Unchanged", summary.unchanged_count],
    ["mae", "MAE change", `${summary.mae_change > 0 ? "+" : ""}${formatNumber(summary.mae_change)}`],
    ["mae-ci", "MAE change 95% CI", `${formatNumber(summary.mae_change_ci_95.lower)} to ${formatNumber(summary.mae_change_ci_95.upper)}`],
  ];
  elements.regressionOutcomeGrid.replaceChildren();
  for (const [key, label, value] of outcomes) {
    const card = document.createElement("article");
    card.className = `transition-card ${key}`;
    const name = document.createElement("span");
    name.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    card.append(name, result);
    elements.regressionOutcomeGrid.append(card);
  }
  const intervalConclusion = summary.mae_change_ci_95.upper < 0
    ? "The paired MAE interval favors V2."
    : summary.mae_change_ci_95.lower > 0
      ? "The paired MAE interval favors V1."
      : "The paired MAE interval crosses zero.";
  const signConclusion = summary.paired_significant
    ? `Error directions differ at p=${formatNumber(summary.paired_sign_test_p)}.`
    : `Direction-only sign test p=${formatNumber(summary.paired_sign_test_p)}.`;
  elements.regressionEvidenceNote.textContent = `${intervalConclusion} ${signConclusion}`;
  elements.regressionOutcomeFilter.value = "all";
  elements.uploadResults.hidden = true;
  elements.regressionResults.hidden = false;
  renderRegressionCharts(report.samples);
  renderRegressionImportance(report);
  renderRegressionSamples();
  elements.regressionResults.scrollIntoView({ behavior: "smooth", block: "start" });
}

function datasetName(value) {
  return value === "wine_binary" ? "Wine (binary)" : "Breast Cancer Wisconsin";
}

function formatPValue(value) {
  const number = Number(value);
  return number < 0.001 ? number.toExponential(2) : number.toFixed(3);
}

function appendRobustnessSummary(label, value) {
  const item = document.createElement("article");
  const name = document.createElement("span");
  const result = document.createElement("strong");
  name.textContent = label;
  result.textContent = value;
  item.append(name, result);
  elements.robustnessSummary.append(item);
}

function renderRobustness(report, message) {
  state.robustness = validateRobustness(report);
  const separated = report.split_results.filter((row) => row.status === "SEPARATED").length;
  const total = report.split_results.length;
  const statusLabel = report.status === "ROBUST_ACROSS_SPLITS" ? "ROBUST" : "SPLIT SENSITIVE";

  elements.robustnessStatus.textContent = statusLabel;
  elements.robustnessStatus.className = `status-badge ${report.status === "ROBUST_ACROSS_SPLITS" ? "within" : "review"}`;
  elements.robustnessExplanation.textContent =
    report.status === "ROBUST_ACROSS_SPLITS"
      ? "Every tested dataset split separated the candidate from its held-out seed-only control."
      : `${separated} of ${total} dataset-split evaluations separated the candidate from control; at least one conclusion changed with the fitted model and probe population.`;

  elements.robustnessSummary.replaceChildren();
  appendRobustnessSummary("Data splits", report.split_seeds.length);
  appendRobustnessSummary("Separated results", `${separated} / ${total}`);
  appendRobustnessSummary("Family-wise alpha", Number(report.family_alpha).toFixed(2));

  elements.robustnessDatasetGrid.replaceChildren();
  for (const dataset of report.dataset_summaries) {
    const consistent = dataset.all_splits_separated && dataset.candidate_exceeds_control_on_all_splits;
    const card = document.createElement("article");
    card.className = `robustness-dataset-card ${consistent ? "consistent" : "sensitive"}`;
    card.innerHTML = `
      <div class="robustness-card-head">
        <div><p class="portfolio-method">LIME</p><h3></h3></div>
        <span class="portfolio-status"></span>
      </div>
      <p class="robustness-card-copy"></p>
      <div class="robustness-ranges">
        <div><span>Control review range</span><strong class="control-range"></strong></div>
        <div><span>Candidate review range</span><strong class="candidate-range"></strong></div>
      </div>`;
    card.querySelector("h3").textContent = datasetName(dataset.dataset);
    const badge = card.querySelector(".portfolio-status");
    badge.textContent = consistent ? "CONSISTENT" : "SPLIT SENSITIVE";
    badge.classList.add(consistent ? "evidence" : "review");
    card.querySelector(".robustness-card-copy").textContent = consistent
      ? `Candidate separation held on all ${dataset.split_count} tested splits.`
      : `Candidate separation did not hold on every tested split.`;
    card.querySelector(".control-range").textContent =
      `${formatPercent(dataset.control_review_rate_min)}–${formatPercent(dataset.control_review_rate_max)}`;
    card.querySelector(".candidate-range").textContent =
      `${formatPercent(dataset.candidate_review_rate_min)}–${formatPercent(dataset.candidate_review_rate_max)}`;
    elements.robustnessDatasetGrid.append(card);
  }

  elements.robustnessResults.replaceChildren();
  for (const result of report.split_results) {
    const row = document.createElement("tr");
    const difference = Number(result.candidate_review_rate) - Number(result.control_review_rate);
    const values = [
      datasetName(result.dataset),
      result.split_seed,
      `${formatPercent(result.control_review_rate)} (${result.control_review_count}/${result.run_count})`,
      `${formatPercent(result.candidate_review_rate)} (${result.candidate_review_count}/${result.run_count})`,
      `${difference >= 0 ? "+" : ""}${formatPercent(difference)}`,
      formatPValue(result.exact_mcnemar_p),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `evidence-pill ${result.status === "SEPARATED" ? "separated" : "exploratory"}`;
    status.textContent = result.status === "SEPARATED" ? "Separated" : "No separation";
    statusCell.append(status);
    row.append(statusCell);
    elements.robustnessResults.append(row);
  }

  elements.robustnessInterpretation.textContent = report.interpretation;
  if (message) elements.message.textContent = message;
}

function headlineLabel(entry) {
  const labels = {
    strongest_candidate_review_rate: "Strongest candidate review rate",
    any_metric_breach_rate: "Any-metric breach rate",
    candidate_review_rate: "Candidate review rate",
  };
  return labels[entry.headline_metric] || String(entry.headline_metric).replaceAll("_", " ");
}

function renderPortfolio(report, message) {
  state.portfolio = validatePortfolio(report);
  elements.portfolioCoverage.textContent =
    `${report.coverage.completed_cells} of ${report.coverage.expected_cells} dataset-method cells completed.`;
  elements.portfolioGrid.replaceChildren();
  elements.portfolioResults.replaceChildren();

  for (const entry of report.entries) {
    const card = document.createElement("article");
    card.className = "portfolio-card";
    card.innerHTML = `
      <div class="portfolio-card-head">
        <div><p class="portfolio-method"></p><h3 class="portfolio-dataset"></h3></div>
        <span class="portfolio-status"></span>
      </div>
      <p class="portfolio-evidence"></p>
      <div class="portfolio-stats">
        <div><span>Prediction agreement</span><strong class="portfolio-agreement"></strong></div>
        <div><span>Cosine similarity</span><strong class="portfolio-cosine"></strong></div>
        <div><span class="portfolio-headline-label"></span><strong class="portfolio-headline"></strong></div>
      </div>`;
    card.querySelector(".portfolio-method").textContent = entry.method;
    card.querySelector(".portfolio-dataset").textContent = entry.dataset;
    const status = card.querySelector(".portfolio-status");
    status.textContent = entry.status;
    const needsAttention = ["REVIEW", "SEPARATED"].includes(entry.status);
    status.className = `portfolio-status ${needsAttention ? "review" : "evidence"}`;
    card.querySelector(".portfolio-evidence").textContent = entry.evidence_type;
    card.querySelector(".portfolio-agreement").textContent = formatPercent(entry.prediction_agreement);
    card.querySelector(".portfolio-cosine").textContent = formatScore(entry.cosine_similarity);
    card.querySelector(".portfolio-headline-label").textContent = headlineLabel(entry);
    card.querySelector(".portfolio-headline").textContent = formatPercent(entry.headline_value);
    elements.portfolioGrid.append(card);

    const row = document.createElement("tr");
    const cells = [
      entry.dataset,
      entry.method,
      entry.evidence_type,
      formatPercent(entry.prediction_agreement),
      formatScore(entry.cosine_similarity),
      `${headlineLabel(entry)}: ${formatPercent(entry.headline_value)}`,
      entry.status,
    ];
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.portfolioResults.append(row);
  }
  elements.portfolioInterpretation.textContent = report.interpretation;
  elements.portfolioComparability.textContent = report.comparability_note;
  if (message) elements.message.textContent = message;
}

function renderSummary() {
  const { report } = state;
  const flagged = report.metrics.filter((metric) => metric.review_flag).length;
  const agreement = report.methods.reduce(
    (total, method) => total + Number(method.prediction_agreement),
    0,
  ) / report.methods.length;
  elements.overallStatus.textContent = report.overall_status;
  elements.overallStatus.className = `status-badge ${statusClass(report.overall_status)}`;
  elements.statusExplanation.textContent =
    flagged > 0
      ? `${flagged} method-level metric signal${flagged === 1 ? "" : "s"} exceeded its calibrated run limit and should be inspected.`
      : "No method-level metric exceeded its calibrated run limit.";
  elements.methodCount.textContent = report.methods.length;
  elements.flagCount.textContent = flagged;
  elements.agreementAverage.textContent = formatPercent(agreement);
  elements.interpretation.textContent = report.interpretation;
}

function renderMethods() {
  elements.methodCards.replaceChildren();
  for (const method of state.report.methods) {
    const card = document.createElement("article");
    card.className = "method-card";
    card.innerHTML = `
      <div class="method-card-header">
        <div>
          <h3 class="method-name"></h3>
          <div class="method-baseline"></div>
        </div>
        <span class="mini-status"></span>
      </div>
      <div class="method-stats">
        <div><span>Probe samples</span><strong class="probe"></strong></div>
        <div><span>Prediction agreement</span><strong class="agreement"></strong></div>
        <div><span>Candidate accuracy</span><strong class="accuracy"></strong></div>
      </div>`;
    card.querySelector(".method-name").textContent = method.method;
    card.querySelector(".method-baseline").textContent = `Baseline: ${method.baseline_type}`;
    const badge = card.querySelector(".mini-status");
    badge.textContent = method.status;
    badge.classList.add(statusClass(method.status));
    card.querySelector(".probe").textContent = method.probe_samples;
    card.querySelector(".agreement").textContent = formatPercent(method.prediction_agreement);
    card.querySelector(".accuracy").textContent = formatPercent(method.candidate_accuracy);
    elements.methodCards.append(card);
  }
}

function renderTabs() {
  elements.methodTabs.replaceChildren();
  for (const method of state.report.methods) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab";
    button.role = "tab";
    button.textContent = method.method;
    button.setAttribute("aria-selected", method.method === state.selectedMethod);
    button.addEventListener("click", () => {
      state.selectedMethod = method.method;
      renderTabs();
      renderMetrics();
    });
    elements.methodTabs.append(button);
  }
}

function renderMetrics() {
  const metrics = state.report.metrics.filter(
    (metric) => metric.method === state.selectedMethod,
  );
  elements.metricGrid.replaceChildren();
  for (const metric of metrics) {
    const maxScale = Math.max(
      Number(metric.candidate_breach_rate),
      Number(metric.run_level_limit),
      0.01,
    ) * 1.2;
    const fillWidth = Math.min(100, Number(metric.candidate_breach_rate) / maxScale * 100);
    const limitPosition = Math.min(100, Number(metric.run_level_limit) / maxScale * 100);
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-title-row">
        <h3 class="metric-title"></h3>
        <span class="flag-pill"></span>
      </div>
      <div class="metric-values">
        <div><span>Baseline mean</span><strong class="baseline"></strong></div>
        <div><span>Candidate mean</span><strong class="candidate"></strong></div>
      </div>
      <div class="meter-label"><span>Candidate breach rate</span><strong class="breach"></strong></div>
      <div class="meter-track" aria-label="Breach rate compared with run-level limit">
        <div class="meter-fill"></div><div class="meter-limit"></div>
      </div>
      <div class="meter-label"><span>0%</span><span class="limit-label"></span></div>`;
    card.querySelector(".metric-title").textContent = metric.metric.replaceAll("_", " ");
    const flag = card.querySelector(".flag-pill");
    flag.textContent = metric.review_flag ? "REVIEW" : "WITHIN LIMIT";
    flag.classList.add(metric.review_flag ? "flagged" : "clear");
    card.querySelector(".baseline").textContent = formatScore(metric.baseline_mean);
    card.querySelector(".candidate").textContent = formatScore(metric.candidate_mean);
    card.querySelector(".breach").textContent = formatPercent(metric.candidate_breach_rate);
    card.querySelector(".meter-fill").style.width = `${fillWidth}%`;
    card.querySelector(".meter-limit").style.left = `${limitPosition}%`;
    card.querySelector(".limit-label").textContent = `Limit ${formatPercent(metric.run_level_limit)}`;
    elements.metricGrid.append(card);
  }
}

function renderReport(report, message) {
  state.report = validateReport(report);
  if (!report.methods.some((method) => method.method === state.selectedMethod)) {
    state.selectedMethod = report.methods[0].method;
  }
  renderSummary();
  renderMethods();
  renderTabs();
  renderMetrics();
  elements.message.textContent = message;
}

function appendSummaryItem(label, value) {
  const item = document.createElement("article");
  const name = document.createElement("span");
  const result = document.createElement("strong");
  name.textContent = label;
  result.textContent = value;
  item.append(name, result);
  elements.benchmarkSummary.append(item);
}

function renderBenchmark() {
  const report = state.benchmarks[state.selectedBenchmark];
  if (!report) return;

  const control = report.scenario_results.find(
    (scenario) => scenario.change_type === "control",
  );
  const strongest = report.scenario_results
    .filter((scenario) => scenario.change_type !== "control")
    .reduce((best, scenario) => (
      !best || Number(scenario.review_rate) > Number(best.review_rate)
        ? scenario
        : best
    ), null);
  const separated = report.paired_control_comparisons.filter(
    (comparison) => Number(comparison.bonferroni_adjusted_p) < 0.05,
  ).length;

  const datasetBits = [datasetLabel(report)];
  if (report.samples) datasetBits.push(`${report.samples} samples`);
  if (report.features) datasetBits.push(`${report.features} features`);
  if (report.probe_samples) datasetBits.push(`${report.probe_samples} fixed probes`);
  elements.benchmarkContext.textContent =
    `${datasetBits.join(" · ")}. Calibration and evaluation use separate random seeds.`;

  elements.benchmarkSummary.replaceChildren();
  appendSummaryItem("Candidate runs", report.candidate_run_count);
  appendSummaryItem("Control review rate", formatPercent(control.review_rate));
  appendSummaryItem("Strongest signal", formatPercent(strongest.review_rate));
  appendSummaryItem("Separated scenarios", `${separated} / ${report.paired_control_comparisons.length}`);

  elements.scenarioGrid.replaceChildren();
  for (const scenario of report.scenario_results) {
    const card = document.createElement("article");
    card.className = `scenario-card ${scenario.change_type === "control" ? "control" : ""}`;
    card.innerHTML = `
      <div class="scenario-title-row">
        <div><h3></h3><p class="scenario-description"></p></div>
        <strong class="scenario-rate"></strong>
      </div>
      <div class="ci-track" aria-label="Review rate with Wilson 95% confidence interval">
        <div class="ci-range"></div><div class="ci-point"></div>
      </div>
      <div class="ci-labels"><span class="ci-low"></span><span class="ci-high"></span></div>
      <div class="scenario-foot"><span class="agreement"></span><span class="cosine"></span></div>`;
    card.querySelector("h3").textContent = scenarioLabel(scenario.scenario_id);
    card.querySelector(".scenario-description").textContent = scenario.description;
    card.querySelector(".scenario-rate").textContent = formatPercent(scenario.review_rate);
    card.querySelector(".ci-range").style.left = `${Number(scenario.review_rate_ci_lower) * 100}%`;
    card.querySelector(".ci-range").style.width = `${(Number(scenario.review_rate_ci_upper) - Number(scenario.review_rate_ci_lower)) * 100}%`;
    card.querySelector(".ci-point").style.left = `${Number(scenario.review_rate) * 100}%`;
    card.querySelector(".ci-low").textContent = `CI ${formatPercent(scenario.review_rate_ci_lower)}`;
    card.querySelector(".ci-high").textContent = formatPercent(scenario.review_rate_ci_upper);
    card.querySelector(".agreement").textContent = `Agreement ${formatPercent(scenario.mean_prediction_agreement)}`;
    card.querySelector(".cosine").textContent = `Cosine ${formatScore(scenario.mean_cosine_similarity)}`;
    elements.scenarioGrid.append(card);
  }

  elements.pairedResults.replaceChildren();
  for (const comparison of report.paired_control_comparisons) {
    const row = document.createElement("tr");
    const adjusted = Number(comparison.bonferroni_adjusted_p);
    const cells = [
      scenarioLabel(comparison.candidate_scenario_id),
      `${comparison.review_rate_difference >= 0 ? "+" : ""}${formatPercent(comparison.review_rate_difference)}`,
      adjusted < 0.001 ? adjusted.toExponential(2) : adjusted.toFixed(3),
    ];
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const evidenceCell = document.createElement("td");
    const evidence = document.createElement("span");
    const isSeparated = adjusted < 0.05;
    evidence.className = `evidence-pill ${isSeparated ? "separated" : "exploratory"}`;
    evidence.textContent = isSeparated ? "Separated from control" : "Exploratory";
    evidenceCell.append(evidence);
    row.append(evidenceCell);
    elements.pairedResults.append(row);
  }
  elements.benchmarkInterpretation.textContent = report.interpretation;
}

function setBenchmarks(reports, message) {
  state.benchmarks = reports.map(validateBenchmark);
  state.selectedBenchmark = 0;
  elements.benchmarkSelector.replaceChildren();
  state.benchmarks.forEach((report, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = datasetLabel(report);
    elements.benchmarkSelector.append(option);
  });
  renderBenchmark();
  switchView("benchmark");
  elements.message.textContent = message;
}

for (const tab of elements.viewTabs) {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
}

elements.benchmarkSelector.addEventListener("change", (event) => {
  state.selectedBenchmark = Number(event.target.value);
  renderBenchmark();
});

elements.uploadTransitionFilter.addEventListener("change", renderUploadSamples);
for (const input of [elements.confidenceThresholdInput, elements.explanationThresholdInput]) {
  input.addEventListener("change", () => {
    renderTriageSummary();
    renderUploadSamples();
  });
}
elements.downloadComparisonJson.addEventListener("click", downloadComparisonJson);
elements.downloadSamplesCsv.addEventListener("click", downloadSamplesCsv);
for (const input of [
  elements.modelV1Upload,
  elements.modelV2Upload,
  elements.additionalModelsUpload,
]) {
  input.addEventListener("change", updateModelSelectors);
}
elements.datasetUpload.addEventListener("change", updateManifestBuilder);
elements.uploadTaskSelector.addEventListener("change", updateUploadTask);
elements.regressionOutcomeFilter.addEventListener("change", renderRegressionSamples);
elements.closeExplanationDialog.addEventListener("click", () => {
  elements.explanationDialog.close();
});
elements.explanationDialog.addEventListener("click", (event) => {
  if (event.target === elements.explanationDialog) elements.explanationDialog.close();
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const isRegression = elements.uploadTaskSelector.value === "regression";
  const models = uploadedModels();
  const baselineIndex = Number(elements.baselineModelSelector.value);
  const candidateIndex = Number(elements.candidateModelSelector.value);
  if (models.length < 2 || baselineIndex === candidateIndex) {
    elements.uploadMessage.className = "upload-comparison-message error";
    elements.uploadMessage.textContent = "Select two different uploaded models.";
    return;
  }
  let manifestFile;
  try {
    manifestFile = selectedManifestFile();
  } catch (error) {
    elements.uploadMessage.className = "upload-comparison-message error";
    elements.uploadMessage.textContent = error.message;
    return;
  }
  const collectionForm = () => {
    const form = new FormData();
    models.forEach((model, index) => form.append(`model_${index}`, model));
    form.append("dataset", elements.datasetUpload.files[0]);
    form.append("manifest", manifestFile);
    return form;
  };
  const comparisonForm = new FormData();
  comparisonForm.append("model_v1", models[baselineIndex]);
  comparisonForm.append("model_v2", models[candidateIndex]);
  comparisonForm.append("dataset", elements.datasetUpload.files[0]);
  comparisonForm.append("manifest", manifestFile);
  if (elements.imagesUpload.files[0]) {
    comparisonForm.append("images", elements.imagesUpload.files[0]);
  }
  elements.uploadSubmit.disabled = true;
  elements.uploadSubmit.textContent = "Comparing…";
  elements.uploadMessage.className = "upload-comparison-message working";
  elements.uploadMessage.textContent = isRegression
    ? "Running the regression comparison locally."
    : "Checking model compatibility before running the comparison.";
  try {
    if (isRegression) {
      const overviewResponse = await fetch("/api/regression/overview", {
        method: "POST",
        body: collectionForm(),
      });
      const overview = await overviewResponse.json();
      if (!overviewResponse.ok) throw new Error(overview.error || `HTTP ${overviewResponse.status}`);
      renderRegressionModelOverview(overview);
      const response = await fetch("/api/regression/compare", {
        method: "POST",
        body: comparisonForm,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      renderRegressionComparison(payload);
      elements.uploadMessage.className = "upload-comparison-message success";
      elements.uploadMessage.textContent = "Regression comparison complete. No uploaded file was saved.";
      return;
    }
    const validationResponse = await fetch("/api/classification/validate", {
      method: "POST",
      body: collectionForm(),
    });
    const validation = await validationResponse.json();
    if (!validationResponse.ok) throw new Error(validation.error || `HTTP ${validationResponse.status}`);
    renderUploadValidation(validation);
    if (!validation.ready) throw new Error("One or more uploaded models are incompatible.");
    const overviewResponse = await fetch("/api/classification/overview", {
      method: "POST",
      body: collectionForm(),
    });
    const overview = await overviewResponse.json();
    if (!overviewResponse.ok) throw new Error(overview.error || `HTTP ${overviewResponse.status}`);
    renderModelOverview(overview);
    const response = await fetch("/api/classification/compare", {
      method: "POST",
      body: comparisonForm,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    renderUploadComparison(payload);
    elements.uploadMessage.className = "upload-comparison-message success";
    elements.uploadMessage.textContent = "Comparison complete. No uploaded file was saved.";
  } catch (error) {
    elements.uploadMessage.className = "upload-comparison-message error";
    elements.uploadMessage.textContent = `Could not run comparison: ${error.message}`;
  } finally {
    elements.uploadSubmit.disabled = false;
    elements.uploadSubmit.textContent = "Run local comparison";
  }
});

updateUploadTask();

elements.upload.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    if (Array.isArray(payload.split_results) && Array.isArray(payload.dataset_summaries)) {
      renderRobustness(payload, `Loaded split-robustness summary locally from ${file.name}.`);
      switchView("robustness");
    } else if (Array.isArray(payload.entries) && payload.coverage) {
      renderPortfolio(payload, `Loaded cross-dataset summary locally from ${file.name}.`);
      switchView("portfolio");
    } else if (Array.isArray(payload.benchmarks)) {
      setBenchmarks(payload.benchmarks, `Loaded benchmarks locally from ${file.name}.`);
    } else if (Array.isArray(payload.scenario_results)) {
      setBenchmarks([payload], `Loaded benchmark locally from ${file.name}.`);
    } else {
      renderReport(payload, `Loaded locally from ${file.name}.`);
      switchView("comparison");
    }
  } catch (error) {
    elements.message.textContent = `Could not load report: ${error.message}`;
  }
});

async function loadInitialReport() {
  let loadedComparison = false;
  try {
    const generated = await fetch("data/report.json", { cache: "no-store" });
    if (generated.ok) {
      renderReport(
        await generated.json(),
        "Loaded the latest generated workflow report.",
      );
      loadedComparison = true;
    } else {
      throw new Error(`HTTP ${generated.status}`);
    }
  } catch (_error) {
    // A normal static server does not provide the runtime report route.
  }

  if (!loadedComparison) {
    try {
      const example = await fetch("data/example-report.json");
      if (!example.ok) throw new Error(`HTTP ${example.status}`);
      renderReport(await example.json(), "Showing the bundled example report.");
    } catch (error) {
      elements.message.textContent = `Example report could not be loaded: ${error.message}`;
    }
  }

  try {
    const response = await fetch("data/benchmarks.json", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    if (Array.isArray(payload.benchmarks) && payload.benchmarks.length) {
      state.benchmarks = payload.benchmarks.map(validateBenchmark);
      elements.benchmarkSelector.replaceChildren();
      state.benchmarks.forEach((report, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = datasetLabel(report);
        elements.benchmarkSelector.append(option);
      });
      renderBenchmark();
    }
  } catch (_error) {
    // Benchmark data is optional when the dashboard is served statically.
  }

  try {
    const response = await fetch("data/portfolio.json", { cache: "no-store" });
    if (!response.ok) return;
    renderPortfolio(await response.json());
  } catch (_error) {
    // Cross-dataset evidence is optional when the dashboard is served statically.
  }

  try {
    const response = await fetch("data/robustness.json", { cache: "no-store" });
    if (!response.ok) return;
    renderRobustness(await response.json());
  } catch (_error) {
    // Split-robustness evidence is optional when served statically.
  }
}

loadInitialReport();
