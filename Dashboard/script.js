const state = {
  systemMode: "normal",
  confidence: 0.0,
  activeThreats: {
    ddos: false,
    spoofing: false,
    jamming: false,
  },
  firstDetected: null,
  healthStatus: "healthy",
  modelStatus: "active",
  dataStatus: "ingesting",
  trafficSeries: new Array(30).fill(20),
  attackSeries: new Array(30).fill(2),
  logFilter: "all",
  logs: [],
};

const featureImportanceData = [
  { name: "Pkt Rate", value: 0.0 },
  { name: "Src Entropy", value: 0.0 },
  { name: "Signal SNR", value: 0.0 },
  { name: "Burst Count", value: 0.0 },
  { name: "RSSI Drift", value: 0.0 },
  { name: "Seq Gap", value: 0.0 },
];

const topIps = {
  source: ["--", "--", "--"],
  destination: ["--", "--", "--"],
  offenders: {
    spoofing: "--",
    jamming: "--",
    ddos: "--",
  },
};

const systemStartedAt = new Date();
systemStartedAt.setHours(systemStartedAt.getHours() - 3);

function randomIp() {
  const oct = () => Math.floor(Math.random() * 255);
  return `${oct()}.${oct()}.${oct()}.${oct()}`;
}

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

function setStatusPill(el, text, cls) {
  el.classList.remove("healthy", "degraded", "critical", "normal", "danger");
  el.classList.add(cls);
  el.textContent = text;
}

function createFeatureBars() {
  const featureBars = document.getElementById("featureBars");
  featureBars.innerHTML = "";
  featureImportanceData.forEach((feature) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    row.innerHTML = `
      <span>${feature.name}</span>
      <div class="feature-track"><div class="feature-value" style="width:${Math.round(feature.value * 100)}%"></div></div>
      <strong>${Math.round(feature.value * 100)}%</strong>
    `;
    featureBars.appendChild(row);
  });
}

function renderIpLists() {
  const source = document.getElementById("sourceIpList");
  const destination = document.getElementById("destinationIpList");
  source.innerHTML = topIps.source.map((ip) => `<li>${ip}</li>`).join("");
  destination.innerHTML = topIps.destination.map((ip) => `<li>${ip}</li>`).join("");

  document.getElementById("spoofingIp").textContent = topIps.offenders.spoofing;
  document.getElementById("jammingIp").textContent = topIps.offenders.jamming;
  document.getElementById("ddosIp").textContent = topIps.offenders.ddos;
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

  ["ddos", "spoofing", "jamming"].forEach((threat) => {
    const el = document.getElementById(`${threat}Indicator`);
    el.classList.toggle("active", state.activeThreats[threat]);
  });

  const detectText = document.getElementById("firstDetected");
  if (!state.firstDetected) {
    detectText.textContent = "First detected: --:--:-- -- (-- ago)";
  } else {
    const elapsed = Date.now() - state.firstDetected.getTime();
    detectText.textContent = `First detected: ${formatTime(state.firstDetected)} (${formatDuration(elapsed)} ago)`;
  }
}

function drawSimpleLineChart(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#d6e6e8";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = Math.max(1, max - min);

  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (w / (data.length - 1)) * i;
    const y = h - ((v - min) / span) * (h - 10) - 5;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function renderLogs() {
  const logSearch = document.getElementById("logSearch").value.toLowerCase();
  const logEl = document.getElementById("eventLog");

  const visible = state.logs.filter((log) => {
    const filterHit = state.logFilter === "all" || log.type === state.logFilter;
    const searchHit =
      `${log.message} ${log.src} ${log.dst} ${log.type}`.toLowerCase().includes(logSearch);
    return filterHit && searchHit;
  });

  if (!visible.length) {
    logEl.innerHTML = '<div class="log-item">No events match this filter.</div>';
    return;
  }

  logEl.innerHTML = visible
    .map(
      (log) => `
      <article class="log-item ${log.type}">
        <div class="log-head">
          <span>${log.ts}</span>
          <span>${log.type.toUpperCase()}</span>
        </div>
        <div class="log-text">${log.message} | src: ${log.src} -> dst: ${log.dst}</div>
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
  const now = new Date();
  const uptime = Date.now() - systemStartedAt.getTime();
  const uptimeStr = formatDuration(uptime);
  const sinceStr = `${String(systemStartedAt.getMonth() + 1).padStart(2, "0")}/${String(systemStartedAt.getDate()).padStart(2, "0")}/${String(systemStartedAt.getFullYear()).slice(-2)} ${systemStartedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  document.getElementById("systemUptime").textContent = `System uptime: ${uptimeStr} (since ${sinceStr})`;
  document.getElementById("lastUpdated").textContent = `Last updated: ${formatTime(now)}`;

  const systemHealthEl = document.getElementById("systemHealthStatus");
  const modelStatusEl = document.getElementById("modelStatus");
  const dataStatusEl = document.getElementById("dataStatus");

  const systemMap = {
    healthy: ["HEALTHY", "healthy"],
    degraded: ["DEGRADED", "degraded"],
    critical: ["CRITICAL", "critical"],
  };

  const modelMap = {
    active: ["ACTIVE", "healthy"],
    delayed: ["DELAYED", "degraded"],
    failed: ["FAILED", "critical"],
  };

  const dataMap = {
    ingesting: ["INGESTING", "healthy"],
    lagging: ["LAGGING", "degraded"],
    dropped: ["DROPPED", "critical"],
  };

  const [systemText, systemClass] = systemMap[state.healthStatus];
  const [modelText, modelClass] = modelMap[state.modelStatus];
  const [dataText, dataClass] = dataMap[state.dataStatus];

  setStatusPill(systemHealthEl, systemText, systemClass);
  setStatusPill(modelStatusEl, modelText, modelClass);
  setStatusPill(dataStatusEl, dataText, dataClass);

  const cpu = `${(22 + Math.random() * 38).toFixed(1)}%`;
  const gpu = `${(18 + Math.random() * 45).toFixed(1)}%`;
  const ram = `${(45 + Math.random() * 35).toFixed(1)}%`;
  const diskIo = `${(150 + Math.random() * 280).toFixed(0)} MB/s`;
  const netLatency = `${(3 + Math.random() * 12).toFixed(1)} ms`;

  const inferenceLatency = `${(8 + Math.random() * 24).toFixed(2)} ms`;
  const predSec = `${(430 + Math.random() * 180).toFixed(0)} /s`;
  const queueDepth = `${Math.floor(2 + Math.random() * 25)}`;
  const batchSize = `${Math.floor(16 + Math.random() * 48)}`;
  const modelVersion = "v0.0-placeholder";

  const packetsCap = `${(38000 + Math.random() * 6000).toFixed(0)} /s`;
  const packetsDrop = `${(8 + Math.random() * 40).toFixed(0)} /s`;
  const dropRate = `${(Math.random() * 0.6).toFixed(2)}%`;
  const lagSec = `${(Math.random() * 3.5).toFixed(2)} s`;
  const queueUtil = `${(24 + Math.random() * 34).toFixed(1)}%`;

  document.getElementById("systemMetrics").innerHTML = metricsTemplate([
    { name: "CPU Usage", value: cpu },
    { name: "GPU Usage", value: gpu },
    { name: "RAM Usage", value: ram },
    { name: "Disk IO", value: diskIo },
    { name: "Network Latency", value: netLatency },
    { name: "Thread Count", value: `${Math.floor(120 + Math.random() * 40)}` },
  ]);

  document.getElementById("modelMetrics").innerHTML = metricsTemplate([
    { name: "Inference Latency", value: inferenceLatency },
    { name: "Predictions/s", value: predSec },
    { name: "Queue Depth", value: queueDepth },
    { name: "Batch Size", value: batchSize },
    { name: "Model Version", value: modelVersion },
    { name: "Drift Score", value: `${(Math.random() * 0.2).toFixed(3)}` },
  ]);

  document.getElementById("dataMetrics").innerHTML = metricsTemplate([
    { name: "Packets Captured/s", value: packetsCap },
    { name: "Packets Dropped/s", value: packetsDrop },
    { name: "Drop Rate", value: dropRate },
    { name: "Ingestion Lag", value: lagSec },
    { name: "Data Queue Util", value: queueUtil },
    { name: "Parser Errors", value: `${Math.floor(Math.random() * 2)}` },
  ]);
}

function pushRandomLog() {
  const types = ["ddos", "spoofing", "jamming"];
  const type = types[Math.floor(Math.random() * types.length)];
  const now = new Date();
  const messages = {
    ddos: "Abnormal packet burst detected",
    spoofing: "Spoof signature matched source identity anomaly",
    jamming: "Signal interference pattern exceeded baseline",
  };

  state.logs.unshift({
    ts: formatTime(now),
    type,
    src: randomIp(),
    dst: randomIp(),
    message: messages[type],
  });

  if (state.logs.length > 80) {
    state.logs.pop();
  }
}

function simulateMainState() {
  const dangerMode = Math.random() > 0.7;
  state.systemMode = dangerMode ? "danger" : "normal";

  state.activeThreats.ddos = dangerMode && Math.random() > 0.45;
  state.activeThreats.spoofing = dangerMode && Math.random() > 0.5;
  state.activeThreats.jamming = dangerMode && Math.random() > 0.55;

  const anyThreat =
    state.activeThreats.ddos || state.activeThreats.spoofing || state.activeThreats.jamming;

  if (anyThreat && !state.firstDetected) {
    state.firstDetected = new Date();
  }
  if (!anyThreat) {
    state.firstDetected = null;
  }

  state.confidence = dangerMode ? 0.76 + Math.random() * 0.2 : 0.35 + Math.random() * 0.22;

  featureImportanceData.forEach((feature) => {
    feature.value = Math.random() * 0.95;
  });

  topIps.source = [randomIp(), randomIp(), randomIp()];
  topIps.destination = [randomIp(), randomIp(), randomIp()];

  topIps.offenders.spoofing = state.activeThreats.spoofing ? randomIp() : "--";
  topIps.offenders.jamming = state.activeThreats.jamming ? randomIp() : "--";
  topIps.offenders.ddos = state.activeThreats.ddos ? randomIp() : "--";

  const addPoint = (series, delta, min, max) => {
    const next = Math.max(min, Math.min(max, series[series.length - 1] + (Math.random() * delta * 2 - delta)));
    series.push(next);
    series.shift();
  };

  addPoint(state.trafficSeries, 9, 8, 95);
  addPoint(state.attackSeries, 2.5, 0, 28);

  if (dangerMode || Math.random() > 0.6) {
    pushRandomLog();
  }

  const healthStates = ["healthy", "degraded", "critical"];
  const modelStates = ["active", "delayed", "failed"];
  const dataStates = ["ingesting", "lagging", "dropped"];
  state.healthStatus = healthStates[Math.floor(Math.random() * healthStates.length)];
  state.modelStatus = modelStates[Math.floor(Math.random() * modelStates.length)];
  state.dataStatus = dataStates[Math.floor(Math.random() * dataStates.length)];
}

function renderAll() {
  renderMainStatus();
  createFeatureBars();
  renderIpLists();
  drawSimpleLineChart("trafficChart", state.trafficSeries, "#0f9ea8");
  drawSimpleLineChart("attackChart", state.attackSeries, "#c7363f");
  renderLogs();
  renderHealth();
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

  for (let i = 0; i < 12; i += 1) {
    pushRandomLog();
  }

  simulateMainState();
  renderAll();

  setInterval(() => {
    simulateMainState();
    renderAll();
  }, 2200);

  setInterval(() => {
    renderMainStatus();
    renderHealth();
  }, 1000);
}

boot();
