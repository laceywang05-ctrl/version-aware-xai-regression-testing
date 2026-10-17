const state = { report: null, selectedMethod: "SHAP" };

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

elements.upload.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    const report = JSON.parse(await file.text());
    renderReport(report, `Loaded locally from ${file.name}.`);
  } catch (error) {
    elements.message.textContent = `Could not load report: ${error.message}`;
  }
});

fetch("data/example-report.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((report) => renderReport(report, "Showing the bundled example report."))
  .catch((error) => {
    elements.message.textContent = `Example report could not be loaded: ${error.message}`;
  });
