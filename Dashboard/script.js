const API_BASE = "http://127.0.0.1:8000";
const SERIES_LEN = 30;

function initSeries() {
  return new Array(SERIES_LEN).fill(null);
}

const state = {
  systemMode: "normal",
  confidence: 0.0,
  activeThreats: {
    safe: true,
    spoofing: false,
    jamming: false,
  },
  firstDetected: null,
  healthStatus: "healthy",
  modelStatus: "active",
  dataStatus: "ingesting",
  rfSeries: {
    band1583: initSeries(),
    band1224: initSeries(),
    bandDelta: initSeries(),
  },
  satelliteSeries: {
    numSvUsed: initSeries(),
    meanCno: initSeries(),
    maxAbsPrRes: initSeries(),
    clockDrift: initSeries(),
    pdop: initSeries(),
  },
  anomalySeries: {
    clockDrift: initSeries(),
    pdop: initSeries(),
    cnoDrop: initSeries(),
    prRes: initSeries(),
  },
  attackSeries: {
    spoofing: initSeries(),
    jamming: initSeries(),
    clockAnomaly: initSeries(),
    satGeometry: initSeries(),
  },
  healthSnapshot: null,
  logFilter: "all",
};

const featureImportanceData = [
  { name: "prRes Outlier", value: 0.0, reason: "--" },
  { name: "Clock Drift", value: 0.0, reason: "--" },
  { name: "DOP Jump", value: 0.0, reason: "--" },
  { name: "ECEF Position Jump", value: 0.0, reason: "--" },
  { name: "C/N0 Drop", value: 0.0, reason: "--" },
  { name: "Band Power Rise", value: 0.0, reason: "--" },
];

const defaultHealthSnapshot = {
  updated_at: null,
  boot_time: null,
  uptime_seconds: null,
  system: { status: "unknown", metrics: [] },
  model: { status: "unknown", metrics: [] },
  data: { status: "unknown", metrics: [] },
  history: [],
};

function formatTime(date) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const hh = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const mm = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function formatSnapshotUptime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "--:--:--";
  }
  return formatDuration(seconds * 1000);
}

function setStatusPill(el, text, cls) {
  el.classList.remove("healthy", "degraded", "critical", "normal", "danger");
  el.classList.add(cls);
  el.textContent = text;
}

function applyLiveFeatureImportance(sample) {
  const features = sample?.prediction?.feature_importance?.top_features ?? [];

  if (!features.length) {
    return;
  }

  featureImportanceData.splice(0, featureImportanceData.length, ...features.slice(0, featureImportanceData.length).map((feature) => ({
    name: feature.name,
    value: feature.value,
    reason: feature.reason,
  })));
}

function createFeatureBars() {
  const featureBars = document.getElementById("featureBars");
  featureBars.innerHTML = "";
  featureImportanceData.forEach((feature) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    row.innerHTML = `
      <div class="feature-meta">
        <div class="feature-name">${feature.name}</div>
        <div class="feature-reason">${feature.reason}</div>
      </div>
      <div class="feature-track"><div class="feature-value" style="width:${Math.round(feature.value * 100)}%"></div></div>
      <div class="feature-score">${Math.round(feature.value * 100)}%</div>
    `;
    featureBars.appendChild(row);
  });
}

function renderMainStatus() {
  const statusPill = document.getElementById("systemStatus");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidenceScore = document.getElementById("confidenceScore");

  if (state.systemMode === "danger") {
    setStatusPill(statusPill, "DANGER", "danger");
  } else {
    setStatusPill(statusPill, "NORMAL", "normal");
  }

  confidenceBar.style.width = `${Math.round(state.confidence * 100)}%`;
  confidenceScore.textContent = `${Math.round(state.confidence * 100)}%`;

  [
    ["safe", "safeIndicator"],
    ["spoofing", "spoofingIndicator"],
    ["jamming", "jammingIndicator"],
  ].forEach(([key, indicatorId]) => {
    const el = document.getElementById(indicatorId);
    el.classList.toggle("active", state.activeThreats[key]);
  });

  const detectText = document.getElementById("firstDetected");
  if (!state.firstDetected) {
    detectText.textContent = "First detected: --:--:-- -- (-- ago)";
  } else {
    const elapsed = Date.now() - state.firstDetected.getTime();
    detectText.textContent = `First detected: ${formatTime(state.firstDetected)} (${formatDuration(elapsed)} ago)`;
  }
}

function getCanvasContext2D(canvasId) {
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = Math.max(1, Math.floor(canvas.clientWidth));
  const displayHeight = Math.max(1, Math.floor(canvas.clientHeight));
  const targetWidth = Math.floor(displayWidth * dpr);
  const targetHeight = Math.floor(displayHeight * dpr);

  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: displayWidth, h: displayHeight };
}

function drawMultiLineChart(canvasId, lines, options) {
  const { ctx, w, h } = getCanvasContext2D(canvasId);
  const margin = { top: 14, right: 8, bottom: 44, left: 36 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  if (plotW <= 10 || plotH <= 10) {
    return;
  }

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fcffff";
  ctx.fillRect(0, 0, w, h);

  const yTicks = 4;
  ctx.strokeStyle = "#deebee";
  ctx.lineWidth = 1;
  for (let i = 0; i <= yTicks; i += 1) {
    const y = margin.top + (plotH / yTicks) * i;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(w - margin.right, y);
    ctx.stroke();

    const value = options.yMax - ((options.yMax - options.yMin) * i) / yTicks;
    ctx.fillStyle = "#5a747d";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yLabel = options.yFormatter ? options.yFormatter(value) : value.toFixed(1);
    ctx.fillText(yLabel, margin.left - 4, y);
  }

  ctx.strokeStyle = "#92adb4";
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, h - margin.bottom);
  ctx.lineTo(w - margin.right, h - margin.bottom);
  ctx.stroke();

  const xTicks = 4;
  const totalSeconds = Math.round((lines[0].data.length - 1) * 2.2);
  for (let i = 0; i <= xTicks; i += 1) {
    const x = margin.left + (plotW / xTicks) * i;
    ctx.strokeStyle = "#deebee";
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, h - margin.bottom);
    ctx.stroke();

    ctx.fillStyle = "#5a747d";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const remaining = Math.round(totalSeconds * (1 - i / xTicks));
    ctx.fillText(remaining === 0 ? "now" : `-${remaining}s`, x, h - margin.bottom + 3);
  }

  lines.forEach((line) => {
    let started = false;
    ctx.beginPath();
    line.data.forEach((value, i) => {
      if (value === null || Number.isNaN(value)) {
        return;
      }
      const x = margin.left + (plotW / (line.data.length - 1)) * i;
      const ratio = (value - options.yMin) / (options.yMax - options.yMin || 1);
      const y = margin.top + (1 - ratio) * plotH;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });

    if (!started) {
      return;
    }

    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.stroke();

    const endValue = [...line.data].reverse().find((value) => value !== null && !Number.isNaN(value));
    if (endValue === undefined) {
      return;
    }
    const endIndex = line.data.lastIndexOf(endValue);
    const endX = w - margin.right;
    const endRatio = (endValue - options.yMin) / (options.yMax - options.yMin || 1);
    const endY = margin.top + (1 - endRatio) * plotH;
    const endXAligned = margin.left + (plotW / (line.data.length - 1)) * endIndex;
    ctx.beginPath();
    ctx.fillStyle = line.color;
    ctx.arc(endXAligned, endY, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });

  const legendOffset = options && typeof options.legendOffset === "number" ? options.legendOffset : 0;
  const legendStartX = margin.left + legendOffset;
  const legendEndX = w - margin.right;
  const legendSlotW = Math.max(1, (legendEndX - legendStartX) / Math.max(lines.length, 1));

  lines.forEach((line, index) => {
    const legendX = legendStartX + index * legendSlotW;
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(legendX, 8);
    ctx.lineTo(legendX + 11, 8);
    ctx.stroke();
    ctx.fillStyle = "#48656d";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(line.name, legendX + 14, 8);
  });

  ctx.save();
  ctx.translate(10, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5a747d";
  ctx.font = "10px IBM Plex Mono";
  ctx.textAlign = "center";
  ctx.fillText(options.yLabel, 0, 0);
  ctx.restore();
}

function drawLiveOverviewChart() {
  drawMultiLineChart(
    "liveOverviewChart",
    [
      { name: "Clock Drift Change", data: state.anomalySeries.clockDrift, color: "#c7363f" },
      { name: "pDOP Change", data: state.anomalySeries.pdop, color: "#8f4bc9" },
      { name: "C/N0 Drop", data: state.anomalySeries.cnoDrop, color: "#2b74c7" },
      { name: "prRes Change", data: state.anomalySeries.prRes, color: "#d39419" },
    ],
    {
      yMin: 0,
      yMax: 1,
      yLabel: "Anomaly Score",
      yFormatter: (v) => v.toFixed(1),
      legendOffset: 24,
    },
  );
}

function pushSeries(series, value) {
  series.push(value);
  series.shift();
}

function normalizedChange(current, previous, scale, invert = false) {
  if (current === null || previous === null || !Number.isFinite(current) || !Number.isFinite(previous)) {
    return null;
  }
  const delta = invert ? previous - current : current - previous;
  const magnitude = Math.abs(delta) / scale;
  return Math.max(0, Math.min(1, magnitude));
}

function renderLogs() {
  const logSearch = document.getElementById("logSearch").value.toLowerCase();
  const logEl = document.getElementById("eventLog");
  const history = state.healthSnapshot?.history ?? [];

  const visible = history.filter((log) => {
    const filterHit = state.logFilter === "all" || log.type === state.logFilter;
    const searchHit =
      `${log.message} ${log.stream} ${log.satellite} ${log.type} ${log.label} ${log.sample_index}`.toLowerCase().includes(logSearch);
    return filterHit && searchHit;
  });

  if (!visible.length) {
    logEl.innerHTML = '<div class="log-item">No real sample history is available yet.</div>';
    return;
  }

  logEl.innerHTML = visible
    .map(
      (log) => `
      <article class="log-item ${log.type}">
        <div class="log-head">
          <span>${formatTime(new Date(log.ts))}</span>
          <span>${log.type.toUpperCase()}</span>
        </div>
        <div class="log-text">${log.message}${log.stream ? ' | ' + log.stream : ''}</div>
      </article>
    `,
    )
    .join("");
}

function metricsTemplate(items) {
  return items
    .map(
      (item) => `
      <div class="metric">
        <div class="name">${item.name}</div>
        <div class="value">${item.value}</div>
      </div>
    `,
    )
    .join("");
}

function renderHealth() {
  const snapshot = state.healthSnapshot ?? defaultHealthSnapshot;
  const uptimeStr = formatSnapshotUptime(snapshot.uptime_seconds);
  const bootTime = snapshot.boot_time ? new Date(snapshot.boot_time) : null;
  const sinceStr = bootTime
    ? `${String(bootTime.getMonth() + 1).padStart(2, "0")}/${String(bootTime.getDate()).padStart(2, "0")}/${String(bootTime.getFullYear()).slice(-2)} ${bootTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
    : "--/--/-- --:--";
  const updatedAt = snapshot.updated_at ? new Date(snapshot.updated_at) : null;

  document.getElementById("systemUptime").textContent = `System uptime: ${uptimeStr} (since ${sinceStr})`;
  document.getElementById("lastUpdated").textContent = `Last updated: ${updatedAt ? formatTime(updatedAt) : "--:--:--"}`;

  const systemHealthEl = document.getElementById("systemHealthStatus");
  const modelStatusEl = document.getElementById("modelStatus");
  const dataStatusEl = document.getElementById("dataStatus");

  const systemMap = {
    healthy: ["HEALTHY", "healthy"],
    degraded: ["DEGRADED", "degraded"],
    critical: ["CRITICAL", "critical"],
    unknown: ["UNAVAILABLE", "degraded"],
  };

  const modelMap = {
    active: ["ACTIVE", "healthy"],
    delayed: ["DELAYED", "degraded"],
    failed: ["FAILED", "critical"],
    unknown: ["UNAVAILABLE", "degraded"],
  };

  const dataMap = {
    ingesting: ["INGESTING", "healthy"],
    lagging: ["LAGGING", "degraded"],
    dropped: ["DROPPED", "critical"],
    unknown: ["UNAVAILABLE", "degraded"],
  };

  const [systemText, systemClass] = systemMap[snapshot.system.status] ?? systemMap.unknown;
  const [modelText, modelClass] = modelMap[snapshot.model.status] ?? modelMap.unknown;
  const [dataText, dataClass] = dataMap[snapshot.data.status] ?? dataMap.unknown;

  setStatusPill(systemHealthEl, systemText, systemClass);
  setStatusPill(modelStatusEl, modelText, modelClass);
  setStatusPill(dataStatusEl, dataText, dataClass);

  document.getElementById("systemMetrics").innerHTML = metricsTemplate(snapshot.system.metrics ?? []);
  document.getElementById("modelMetrics").innerHTML = metricsTemplate(snapshot.model.metrics ?? []);
  document.getElementById("dataMetrics").innerHTML = metricsTemplate(snapshot.data.metrics ?? []);
}

function ingestLiveSample(sample) {
  const probabilities = sample.prediction.probabilities;
  const importance = sample.prediction.feature_importance?.top_features ?? [];
  const displayLabel = sample.prediction.display_label ?? sample.prediction.label;
  const sat = sample.satellite;
  const derived = sample.derived ?? {};

  state.confidence = sample.prediction.confidence;
  state.systemMode = sample.prediction.class_id === 0 ? "normal" : "danger";
  state.activeThreats.safe = sample.prediction.class_id === 0;
  state.activeThreats.spoofing = sample.prediction.class_id === 1 || probabilities.Spoofing > probabilities.Clean;
  state.activeThreats.jamming = sample.prediction.class_id === 2 || probabilities.Jamming > probabilities.Clean;

  state.attackSeries.spoofing.push(Math.max(0, Math.min(1, probabilities.Spoofing ?? 0)));
  state.attackSeries.spoofing.shift();
  state.attackSeries.jamming.push(Math.max(0, Math.min(1, probabilities.Jamming ?? 0)));
  state.attackSeries.jamming.shift();

  const band1583 = typeof derived.rf_band_1583 === "number" ? derived.rf_band_1583 : state.rfSeries.band1583[state.rfSeries.band1583.length - 1];
  const band1224 = typeof derived.rf_band_1224 === "number" ? derived.rf_band_1224 : state.rfSeries.band1224[state.rfSeries.band1224.length - 1];
  const bandDelta = typeof derived.rf_band_delta === "number" ? derived.rf_band_delta : band1583 - band1224;
  state.rfSeries.band1583.push(Math.max(0, Math.min(40, band1583)));
  state.rfSeries.band1583.shift();
  state.rfSeries.band1224.push(Math.max(0, Math.min(40, band1224)));
  state.rfSeries.band1224.shift();
  pushSeries(state.rfSeries.bandDelta, bandDelta);

  if (sat) {
    const nextNumSvUsed = Math.max(0, Math.min(70, sat.num_sv_used ?? 0));
    const nextMeanCno = Math.max(0, Math.min(70, sat.mean_cno ?? 0));
    const nextMaxAbsPrRes = Math.max(0, Math.min(70, sat.max_abs_pr_res ?? 0));
    const nextClockDrift = Math.max(0, Math.min(70, Math.abs(sat.clock_drift ?? 0)));
    const nextPdop = Math.max(0, Math.min(70, sat.pdop ?? 0));

    const prevClock = state.satelliteSeries.clockDrift[state.satelliteSeries.clockDrift.length - 1];
    const prevPdop = state.satelliteSeries.pdop[state.satelliteSeries.pdop.length - 1];
    const prevCno = state.satelliteSeries.meanCno[state.satelliteSeries.meanCno.length - 1];
    const prevPrRes = state.satelliteSeries.maxAbsPrRes[state.satelliteSeries.maxAbsPrRes.length - 1];

    pushSeries(state.satelliteSeries.numSvUsed, nextNumSvUsed);
    pushSeries(state.satelliteSeries.meanCno, nextMeanCno);
    pushSeries(state.satelliteSeries.maxAbsPrRes, nextMaxAbsPrRes);
    pushSeries(state.satelliteSeries.clockDrift, nextClockDrift);
    pushSeries(state.satelliteSeries.pdop, nextPdop);
    pushSeries(state.anomalySeries.clockDrift, normalizedChange(nextClockDrift, prevClock, 0.4));
    pushSeries(state.anomalySeries.pdop, normalizedChange(nextPdop, prevPdop, 0.8));
    pushSeries(state.anomalySeries.cnoDrop, normalizedChange(prevCno, nextMeanCno, 1.8, true));
    pushSeries(state.anomalySeries.prRes, normalizedChange(nextMaxAbsPrRes, prevPrRes, 2.5));
  }

  if (state.systemMode === "danger" && !state.firstDetected) {
    state.firstDetected = new Date();
  }
  if (state.systemMode === "normal") {
    state.firstDetected = null;
  }

  if (importance.length) {
    featureImportanceData.splice(0, featureImportanceData.length, ...importance.slice(0, featureImportanceData.length).map((feature) => ({
      name: feature.name,
      value: feature.value,
      reason: feature.reason,
    })));
  }

  const predictionEl = document.getElementById("currentPrediction");
  if (predictionEl) {
    predictionEl.textContent = `${displayLabel} (${Math.round(sample.prediction.confidence * 100)}%)`;
  }
}

function renderAll() {
  renderMainStatus();
  createFeatureBars();
  drawLiveOverviewChart();
  drawMultiLineChart(
    "rfChart",
    [
      { name: "RF Delta", data: state.rfSeries.bandDelta, color: "#0f766e" },
    ],
    {
      yMin: -25,
      yMax: 25,
      yLabel: "Delta (dB)",
      yFormatter: (v) => `${Math.round(v)}`,
    },
  );
  drawMultiLineChart(
    "attackScoreChart",
    [
      {
        name: "Safe",
        data: state.attackSeries.spoofing.map((_, i) => {
          const spoof = state.attackSeries.spoofing[i];
          const jam = state.attackSeries.jamming[i];
          if (spoof === null || jam === null) {
            return null;
          }
          return 1 - Math.max(spoof, jam);
        }),
        color: "#1e8f4d",
      },
      { name: "Spoof", data: state.attackSeries.spoofing, color: "#d39419" },
      { name: "Jam", data: state.attackSeries.jamming, color: "#8f4bc9" },
    ],
    {
      yMin: 0,
      yMax: 1,
      yLabel: "Risk Score",
      yFormatter: (v) => v.toFixed(1),
    },
  );
  renderLogs();
  renderHealth();
}

async function fetchNextSample() {
  const response = await fetch(`${API_BASE}/api/next`, {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API request failed with ${response.status}`);
  }

  return response.json();
}

async function fetchHealthSnapshot() {
  const response = await fetch(`${API_BASE}/api/health`, {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Health request failed with ${response.status}`);
  }

  return response.json();
}

async function resetStream(index) {
  const response = await fetch(`${API_BASE}/api/reset?index=${encodeURIComponent(index)}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Reset failed with ${response.status}`);
  }

  return response.json();
}

function setupTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const views = document.querySelectorAll(".tab-view");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      views.forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

function setupFilters() {
  const filterButtons = document.querySelectorAll(".filter-btn");
  filterButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.logFilter = btn.dataset.filter;
      renderLogs();
    });
  });

  document.getElementById("logSearch").addEventListener("input", renderLogs);
}

function boot() {
  setupTabs();
  setupFilters();

  const refreshHealth = async () => {
    try {
      state.healthSnapshot = await fetchHealthSnapshot();
    } catch (error) {
      console.error(error);
    }

    renderHealth();
  };

  const tick = async () => {
    try {
      const sample = await fetchNextSample();
      ingestLiveSample(sample);
    } catch (error) {
      console.error(error);
    }

    renderAll();
  };

  const jumpInput = document.getElementById("healthJumpInput");
  const jumpButton = document.getElementById("healthJumpButton");
  const applyJump = async () => {
    const parsed = Number.parseInt(jumpInput.value, 10);
    const safeIndex = Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
    jumpInput.value = String(safeIndex);

    try {
      await resetStream(safeIndex);
      await tick();
    } catch (error) {
      console.error(error);
    }
  };

  if (jumpButton && jumpInput) {
    jumpButton.addEventListener("click", applyJump);
    jumpInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        applyJump();
      }
    });
  }

  tick();
  refreshHealth();

  setInterval(tick, 2200);
  setInterval(refreshHealth, 1000);

  window.addEventListener("resize", () => {
    renderAll();
  });
}

boot();
