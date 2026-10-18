const state = {
  report: null,
  selectedMethod: "SHAP",
  benchmarks: [],
  selectedBenchmark: 0,
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
  elements.comparisonView.hidden = benchmark;
  elements.benchmarkView.hidden = !benchmark;
  for (const tab of elements.viewTabs) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
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

elements.upload.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    if (Array.isArray(payload.benchmarks)) {
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
}

loadInitialReport();
