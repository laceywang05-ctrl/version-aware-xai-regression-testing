const state = {
  report: null,
  selectedMethod: "SHAP",
  benchmarks: [],
  selectedBenchmark: 0,
  portfolio: null,
  robustness: null,
  uploadComparison: null,
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
  uploadSubmit: document.querySelector("#run-upload-comparison"),
  uploadMessage: document.querySelector("#upload-comparison-message"),
  uploadResults: document.querySelector("#upload-results"),
  uploadTotal: document.querySelector("#upload-total"),
  uploadV1Accuracy: document.querySelector("#upload-v1-accuracy"),
  uploadV2Accuracy: document.querySelector("#upload-v2-accuracy"),
  uploadAgreement: document.querySelector("#upload-agreement"),
  uploadScopeNote: document.querySelector("#upload-scope-note"),
  uploadTransitionGrid: document.querySelector("#upload-transition-grid"),
  uploadTransitionFilter: document.querySelector("#upload-transition-filter"),
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
  elements.v1HeatmapCaption.textContent = `Model V1 · ${v1Method} toward “${sample.v1_prediction}”`;
  elements.v2HeatmapCaption.textContent = `Model V2 · ${v2Method} toward “${sample.v2_prediction}”`;
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
  const matching = report.samples.filter(
    (sample) => selected === "all" || sample.transition === selected,
  );
  const displayed = matching.slice(0, 500);
  elements.uploadSampleResults.replaceChildren();
  for (const sample of displayed) {
    const row = document.createElement("tr");
    if (sample.transition === "v1_correct_v2_wrong") row.className = "regression-row";
    if (sample.transition === "v1_wrong_v2_correct") row.className = "improvement-row";
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

function renderUploadComparison(report) {
  if (!report || report.task !== "classification" || !report.summary || !Array.isArray(report.samples)) {
    throw new Error("The server returned an invalid classification comparison.");
  }
  state.uploadComparison = report;
  const summary = report.summary;
  elements.uploadTotal.textContent = summary.total_samples;
  elements.uploadV1Accuracy.textContent = formatPercent(summary.v1_accuracy);
  elements.uploadV2Accuracy.textContent = formatPercent(summary.v2_accuracy);
  elements.uploadAgreement.textContent = formatPercent(summary.prediction_agreement);
  elements.uploadScopeNote.textContent = report.dataset?.note
    || "Accuracy summarizes the supplied samples; the table shows the concrete changes behind it.";

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
elements.closeExplanationDialog.addEventListener("click", () => {
  elements.explanationDialog.close();
});
elements.explanationDialog.addEventListener("click", (event) => {
  if (event.target === elements.explanationDialog) elements.explanationDialog.close();
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(elements.uploadForm);
  const optionalImages = form.get("images");
  if (optionalImages instanceof File && optionalImages.size === 0) {
    form.delete("images");
  }
  elements.uploadSubmit.disabled = true;
  elements.uploadSubmit.textContent = "Comparing…";
  elements.uploadMessage.className = "upload-comparison-message working";
  elements.uploadMessage.textContent = "Running both models on the labelled samples locally.";
  try {
    const response = await fetch("/api/classification/compare", {
      method: "POST",
      body: form,
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
