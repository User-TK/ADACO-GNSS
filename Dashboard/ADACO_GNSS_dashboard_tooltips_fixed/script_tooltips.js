const API_BASE = "http://127.0.0.1:8000";
const SERIES_LEN = 30;

const TOOLTIP_TEXT = {
  featureName: "Name of the signal or navigation feature that contributed to the current model prediction.",
  featureReason: "Plain-language reason this feature may support a spoofing, jamming, or clean classification.",
  featureTrack: "Relative contribution strength for this feature in the current prediction, from 0% to 100%.",
  featureScore: "Numeric version of the feature contribution bar.",
  logTime: "Time when this event was added to the dashboard history.",
  logType: "Event category used by the filter buttons.",
  logText: "Short explanation of the anomaly, GNSS stream, and satellite associated with the event.",
};

function escapeHTML(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function tooltipHTML(text, tooltip, className = "") {
  const classes = className ? ` ${className}` : "";
  return `<span class="tip${classes}" tabindex="0" data-tooltip="${escapeHTML(tooltip)}">${escapeHTML(text)}</span>`;
}

const chartTooltipRegions = new Map();
let activePinnedTooltipTarget = null;
let floatingTooltipEl = null;

const CHART_TOOLTIP_TEXT = {
  "1583MHz": "Recent relative RF power around the monitored upper GNSS-related band. A sudden rise can support a jamming interpretation.",
  "1224MHz": "Recent relative RF power around the monitored lower GNSS-related band. Compare it with 1583 MHz for band-specific interference changes.",
  "Safe": "Model score for the clean, non-attack class. This is computed as the inverse of the strongest attack score in the live display.",
  "Spoof": "Model score for spoofing risk. Higher values mean the current signal pattern more strongly resembles misleading GNSS behavior.",
  "Jam": "Model score for jamming risk. Higher values mean the current RF pattern more strongly resembles interference or noise injection.",
  "Clock Drift Change": "Normalized change in receiver clock drift. Large changes can accompany spoofing or unstable receiver timing.",
  "pDOP Change": "Normalized change in position dilution of precision, a satellite geometry quality measure.",
  "C/N0 Drop": "Normalized drop in carrier-to-noise density. Broad C/N0 drops can indicate signal degradation or jamming.",
  "prRes Change": "Normalized change in pseudorange residuals. Large residual shifts can indicate spoofing or inconsistent satellite ranging.",
  "Power (dB)": "Vertical axis for RF band power. Higher values indicate stronger measured signal or interference power.",
  "Risk Score": "Vertical axis for normalized model risk scores. 0 means weak support and 1 means strong support.",
  "Anomaly Score": "Vertical axis for normalized GNSS feature-change scores. Higher values indicate stronger recent anomalies.",
  "Time Window": "Horizontal axis showing the recent rolling window, with older samples on the left and the newest sample at 'now'.",
};

function ensureFloatingTooltip() {
  if (floatingTooltipEl) {
    return floatingTooltipEl;
  }

  floatingTooltipEl = document.createElement("div");
  floatingTooltipEl.id = "floatingTooltip";
  floatingTooltipEl.setAttribute("role", "tooltip");
  document.body.appendChild(floatingTooltipEl);
  return floatingTooltipEl;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function positionFloatingTooltip(anchor, preferBelow = false) {
  const tooltip = ensureFloatingTooltip();
  const pad = 8;
  const tipRect = tooltip.getBoundingClientRect();
  const anchorRect = anchor instanceof DOMRect ? anchor : null;
  const anchorX = anchorRect ? anchorRect.left + anchorRect.width / 2 : anchor.x;
  const anchorTop = anchorRect ? anchorRect.top : anchor.y;
  const anchorBottom = anchorRect ? anchorRect.bottom : anchor.y;

  let left = anchorX - tipRect.width / 2;
  let top = preferBelow ? anchorBottom + 10 : anchorTop - tipRect.height - 10;

  left = clamp(left, pad, window.innerWidth - tipRect.width - pad);

  if (!preferBelow && top < pad) {
    top = anchorBottom + 10;
  }
  if (preferBelow && top + tipRect.height > window.innerHeight - pad) {
    top = anchorTop - tipRect.height - 10;
  }
  top = clamp(top, pad, window.innerHeight - tipRect.height - pad);

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function showFloatingTooltip(text, anchor, options = {}) {
  if (!text) {
    return;
  }
  const tooltip = ensureFloatingTooltip();
  tooltip.textContent = text;
  tooltip.classList.add("visible");
  positionFloatingTooltip(anchor, options.preferBelow ?? false);
}

function hideFloatingTooltip({ force = false } = {}) {
  if (!floatingTooltipEl) {
    return;
  }
  if (activePinnedTooltipTarget && !force) {
    return;
  }
  floatingTooltipEl.classList.remove("visible");
}

function bindTooltipDismissal() {
  ensureFloatingTooltip();

  document.addEventListener("pointerover", (event) => {
    const target = event.target.closest("[data-tooltip]");
    if (!target || activePinnedTooltipTarget) {
      return;
    }
    showFloatingTooltip(target.dataset.tooltip, target.getBoundingClientRect(), {
      preferBelow: target.closest(".metric, .log-item, .feature-row") !== null,
    });
  });

  document.addEventListener("focusin", (event) => {
    const target = event.target.closest("[data-tooltip]");
    if (!target || activePinnedTooltipTarget) {
      return;
    }
    showFloatingTooltip(target.dataset.tooltip, target.getBoundingClientRect(), {
      preferBelow: target.closest(".metric, .log-item, .feature-row") !== null,
    });
  });

  document.addEventListener("pointerout", (event) => {
    const target = event.target.closest("[data-tooltip]");
    if (!target || activePinnedTooltipTarget) {
      return;
    }
    if (event.relatedTarget && target.contains(event.relatedTarget)) {
      return;
    }
    hideFloatingTooltip();
  });

  document.addEventListener("focusout", () => {
    if (!activePinnedTooltipTarget) {
      hideFloatingTooltip();
    }
  });

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-tooltip]");
    if (!target) {
      activePinnedTooltipTarget = null;
      hideFloatingTooltip({ force: true });
      return;
    }

    if (activePinnedTooltipTarget === target) {
      activePinnedTooltipTarget = null;
      hideFloatingTooltip({ force: true });
      return;
    }

    activePinnedTooltipTarget = target;
    showFloatingTooltip(target.dataset.tooltip, target.getBoundingClientRect(), {
      preferBelow: target.closest(".metric, .log-item, .feature-row") !== null,
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    activePinnedTooltipTarget = null;
    hideFloatingTooltip({ force: true });
  });
}

function addChartTooltipRegion(canvasId, region) {
  if (!chartTooltipRegions.has(canvasId)) {
    chartTooltipRegions.set(canvasId, []);
  }
  chartTooltipRegions.get(canvasId).push(region);
}

function chartTooltipForLabel(label, fallback) {
  return CHART_TOOLTIP_TEXT[label] || fallback || "Chart label. Hover or press near this label to see what the displayed value represents.";
}

function setupCanvasTooltips() {
  document.querySelectorAll("canvas").forEach((canvas) => {
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", "Interactive chart with tooltip explanations for legend and axis labels");

    const findRegion = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const regions = chartTooltipRegions.get(canvas.id) || [];
      return {
        rect,
        x,
        y,
        region: regions.find((r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h),
      };
    };

    canvas.addEventListener("mousemove", (event) => {
      if (activePinnedTooltipTarget) {
        return;
      }
      const hit = findRegion(event);
      if (!hit.region) {
        canvas.style.cursor = "default";
        hideFloatingTooltip();
        return;
      }
      canvas.style.cursor = "help";
      showFloatingTooltip(hit.region.tooltip, {
        x: hit.rect.left + hit.x,
        y: hit.rect.top + hit.y,
      }, { preferBelow: hit.y < 28 });
    });

    canvas.addEventListener("mouseleave", () => {
      if (!activePinnedTooltipTarget) {
        canvas.style.cursor = "default";
        hideFloatingTooltip();
      }
    });

    canvas.addEventListener("click", (event) => {
      const hit = findRegion(event);
      if (!hit.region) {
        activePinnedTooltipTarget = null;
        hideFloatingTooltip({ force: true });
        return;
      }
      activePinnedTooltipTarget = canvas;
      showFloatingTooltip(hit.region.tooltip, {
        x: hit.rect.left + hit.x,
        y: hit.rect.top + hit.y,
      }, { preferBelow: hit.y < 28 });
    });

    canvas.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const regions = chartTooltipRegions.get(canvas.id) || [];
      const region = regions[0];
      if (region) {
        activePinnedTooltipTarget = canvas;
        showFloatingTooltip(region.tooltip, {
          x: rect.left + region.x + region.w / 2,
          y: rect.top + region.y + region.h / 2,
        }, { preferBelow: true });
      }
    });
  });
}


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
  logFilter: "all",
  logs: [],
};

const featureImportanceData = [
  { name: "prRes Outlier", value: 0.0, reason: "--" },
  { name: "Clock Drift", value: 0.0, reason: "--" },
  { name: "DOP Jump", value: 0.0, reason: "--" },
  { name: "ECEF Position Jump", value: 0.0, reason: "--" },
  { name: "C/N0 Drop", value: 0.0, reason: "--" },
  { name: "Band Power Rise", value: 0.0, reason: "--" },
];

const systemStartedAt = new Date();
systemStartedAt.setHours(systemStartedAt.getHours() - 3);

function randomSatellite() {
  const gnss = ["GPS", "GLO", "GAL", "BDS"];
  const svId = Math.floor(1 + Math.random() * 32);
  return `${gnss[Math.floor(Math.random() * gnss.length)]}-${String(svId).padStart(2, "0")}`;
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
    const percentage = `${Math.round(feature.value * 100)}%`;
    row.innerHTML = `
      <div class="feature-meta">
        <div class="feature-name">${tooltipHTML(feature.name, `${TOOLTIP_TEXT.featureName} Current value: ${feature.name}.`)}</div>
        <div class="feature-reason">${tooltipHTML(feature.reason, TOOLTIP_TEXT.featureReason)}</div>
      </div>
      <div class="feature-track tip" tabindex="0" data-tooltip="${escapeHTML(`${TOOLTIP_TEXT.featureTrack} Current strength: ${percentage}.`)}"><div class="feature-value" style="width:${percentage}"></div></div>
      <div class="feature-score">${tooltipHTML(percentage, TOOLTIP_TEXT.featureScore)}</div>
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
  chartTooltipRegions.set(canvasId, []);

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
    addChartTooltipRegion(canvasId, {
      x: 0,
      y: y - 9,
      w: margin.left - 2,
      h: 18,
      tooltip: `${yLabel} on the ${options.yLabel} axis. ${chartTooltipForLabel(options.yLabel)}`
    });
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
    const xLabel = remaining === 0 ? "now" : `-${remaining}s`;
    ctx.fillText(xLabel, x, h - margin.bottom + 3);
    addChartTooltipRegion(canvasId, {
      x: x - 22,
      y: h - margin.bottom,
      w: 44,
      h: 18,
      tooltip: `${xLabel} in the recent rolling time window. ${chartTooltipForLabel("Time Window")}`
    });
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

  let legendX = margin.left;
  lines.forEach((line) => {
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
    const legendW = Math.min(88, line.name.length * 7 + 24);
    addChartTooltipRegion(canvasId, {
      x: legendX - 4,
      y: 0,
      w: legendW,
      h: 18,
      tooltip: `${line.name}: ${chartTooltipForLabel(line.name)}`
    });
    legendX += legendW;
  });

  ctx.save();
  ctx.translate(10, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5a747d";
  ctx.font = "10px IBM Plex Mono";
  ctx.textAlign = "center";
  ctx.fillText(options.yLabel, 0, 0);
  ctx.restore();

  addChartTooltipRegion(canvasId, {
    x: 0,
    y: margin.top,
    w: 22,
    h: plotH,
    tooltip: `${options.yLabel}: ${chartTooltipForLabel(options.yLabel)}`
  });
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

  const visible = state.logs.filter((log) => {
    const filterHit = state.logFilter === "all" || log.type === state.logFilter;
    const searchHit =
      `${log.message} ${log.stream} ${log.satellite} ${log.type}`.toLowerCase().includes(logSearch);
    return filterHit && searchHit;
  });

  if (!visible.length) {
    logEl.innerHTML = '<div class="log-item tip" tabindex="0" data-tooltip="The current filter and search text do not match any stored event log entries.">No events match this filter.</div>';
    return;
  }

  logEl.innerHTML = visible
    .map(
      (log) => `
      <article class="log-item ${escapeHTML(log.type)}">
        <div class="log-head">
          ${tooltipHTML(log.ts, TOOLTIP_TEXT.logTime)}
          ${tooltipHTML(log.type.toUpperCase(), TOOLTIP_TEXT.logType)}
        </div>
        <div class="log-text">${tooltipHTML(`${log.message} | ${log.stream} | sat: ${log.satellite}`, TOOLTIP_TEXT.logText)}</div>
      </article>
    `,
    )
    .join("");
}


function metricTooltip(name) {
  const tooltips = {
    "CPU Usage": "Approximate percentage of processor capacity being used by the dashboard environment.",
    "GPU Usage": "Approximate GPU utilization. Higher values may occur during model inference or visualization workloads.",
    "RAM Usage": "Approximate memory usage for the current runtime.",
    "Disk IO": "Approximate disk read/write throughput, useful for detecting slow dataset or log access.",
    "Network Latency": "Estimated communication delay between client and backend services.",
    "Thread Count": "Approximate number of active runtime threads.",
    "Inference Latency": "Estimated time required for the model to produce one prediction batch or sample output.",
    "Predictions/s": "Approximate number of predictions the model pipeline can produce per second.",
    "Queue Depth": "Number of pending samples waiting for processing.",
    "Batch Size": "Number of samples grouped together for a model inference or processing step.",
    "Model Version": "Identifier for the currently loaded model checkpoint or placeholder version.",
    "Drift Score": "Small indicator of possible model or data-distribution drift relative to expected behavior.",
    "Packets Captured/s": "Approximate number of incoming data packets captured per second.",
    "Packets Dropped/s": "Approximate number of packets missed or discarded per second.",
    "Drop Rate": "Percentage of captured input that is being dropped.",
    "Ingestion Lag": "Delay between data arrival and dashboard processing.",
    "NAV-PVT Rate": "Rate of navigation position/velocity/time updates.",
    "RAWX Rate": "Rate of raw GNSS measurement updates.",
  };
  return tooltips[name] ?? "Runtime or model metric shown for system monitoring.";
}

function metricsTemplate(items) {
  return items
    .map(
      (item) => `
      <div class="metric">
        <div class="name">${tooltipHTML(item.name, metricTooltip(item.name))}</div>
        <div class="value">${tooltipHTML(item.value, `Current displayed value for ${item.name}.`)}</div>
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
  const pvtRate = `${(8 + Math.random() * 4).toFixed(1)} Hz`;
  const satRate = `${(4 + Math.random() * 2).toFixed(1)} Hz`;
  const rawxRate = `${(8 + Math.random() * 5).toFixed(1)} Hz`;

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
    { name: "NAV-PVT Rate", value: pvtRate },
    { name: "RAWX Rate", value: rawxRate },
  ]);
}

function pushRandomLog() {
  const types = ["spoofing", "jamming", "clock-anomaly", "sat-geometry"];
  const type = types[Math.floor(Math.random() * types.length)];
  const now = new Date();
  const streams = ["MON-SPAN", "NAV-CLOCK", "NAV-DOP", "NAV-SAT", "NAV-PVT", "RXM-RAWX"];
  const messages = {
    spoofing: "Spoof signature matched pseudorange residual inconsistency",
    jamming: "RF power abnormal rise detected in protected GNSS band",
    "clock-anomaly": "Receiver clock bias/drift exceeded expected baseline",
    "sat-geometry": "DOP jump suggests geometry inconsistency",
  };

  state.logs.unshift({
    ts: formatTime(now),
    type,
    stream: streams[Math.floor(Math.random() * streams.length)],
    satellite: randomSatellite(),
    message: messages[type],
  });

  if (state.logs.length > 80) {
    state.logs.pop();
  }
}

function simulateMainState() {
  const dangerMode = Math.random() > 0.68;

  const addPoint = (series, delta, min, max) => {
    const next = Math.max(min, Math.min(max, series[series.length - 1] + (Math.random() * delta * 2 - delta)));
    series.push(next);
    series.shift();
  };

  addPoint(state.rfSeries.band1583, dangerMode ? 2.9 : 1.4, 6, 38);
  addPoint(state.rfSeries.band1224, dangerMode ? 2.4 : 1.2, 5, 34);

  const spoofScore = dangerMode ? 0.55 + Math.random() * 0.4 : Math.random() * 0.45;
  const jamScore = dangerMode ? 0.52 + Math.random() * 0.42 : Math.random() * 0.4;
  const clockScore = dangerMode ? 0.48 + Math.random() * 0.44 : Math.random() * 0.35;
  const dopScore = dangerMode ? 0.46 + Math.random() * 0.42 : Math.random() * 0.35;

  state.attackSeries.spoofing.push(spoofScore);
  state.attackSeries.spoofing.shift();
  state.attackSeries.jamming.push(jamScore);
  state.attackSeries.jamming.shift();
  state.attackSeries.clockAnomaly.push(clockScore);
  state.attackSeries.clockAnomaly.shift();
  state.attackSeries.satGeometry.push(dopScore);
  state.attackSeries.satGeometry.shift();

  const numSvUsed = 10 + Math.random() * 22;
  const meanCno = 22 + Math.random() * 26;
  const maxAbsPrRes = 4 + Math.random() * 24;
  const clockDrift = Math.abs(0.1 + Math.random() * 2.3);
  const pdop = 0.5 + Math.random() * 6;
  const prevClock = state.satelliteSeries.clockDrift[state.satelliteSeries.clockDrift.length - 1];
  const prevPdop = state.satelliteSeries.pdop[state.satelliteSeries.pdop.length - 1];
  const prevCno = state.satelliteSeries.meanCno[state.satelliteSeries.meanCno.length - 1];
  const prevPrRes = state.satelliteSeries.maxAbsPrRes[state.satelliteSeries.maxAbsPrRes.length - 1];
  pushSeries(state.satelliteSeries.numSvUsed, numSvUsed);
  pushSeries(state.satelliteSeries.meanCno, meanCno);
  pushSeries(state.satelliteSeries.maxAbsPrRes, maxAbsPrRes);
  pushSeries(state.satelliteSeries.clockDrift, clockDrift);
  pushSeries(state.satelliteSeries.pdop, pdop);
  pushSeries(state.anomalySeries.clockDrift, normalizedChange(clockDrift, prevClock, 0.4));
  pushSeries(state.anomalySeries.pdop, normalizedChange(pdop, prevPdop, 0.8));
  pushSeries(state.anomalySeries.cnoDrop, normalizedChange(prevCno, meanCno, 1.8, true));
  pushSeries(state.anomalySeries.prRes, normalizedChange(maxAbsPrRes, prevPrRes, 2.5));

  state.activeThreats.spoofing = spoofScore > 0.66;
  state.activeThreats.jamming = jamScore > 0.66;
  state.activeThreats.safe = !(state.activeThreats.spoofing || state.activeThreats.jamming);

  const anyThreat =
    state.activeThreats.spoofing
    || state.activeThreats.jamming
    || !state.activeThreats.safe;

  state.systemMode = anyThreat ? "danger" : "normal";

  if (anyThreat && !state.firstDetected) {
    state.firstDetected = new Date();
  }
  if (!anyThreat) {
    state.firstDetected = null;
  }

  state.confidence = Math.max(spoofScore, jamScore, clockScore, dopScore);

  const prRes = 28 + Math.random() * 86;
  const clkDrift = 0.15 + Math.random() * 1.95;
  const dopJump = 0.2 + Math.random() * 2.7;
  const ecefJump = 0.8 + Math.random() * 28;
  const cnoDrop = 1.5 + Math.random() * 16;
  const bandRise = 2 + Math.random() * 19;

  featureImportanceData[0].value = Math.min(1, prRes / 95);
  featureImportanceData[0].reason = `NAV-SAT prRes is satellite range error (${prRes.toFixed(1)} m). Large errors can indicate spoofing.`;
  featureImportanceData[1].value = Math.min(1, clkDrift / 2);
  featureImportanceData[1].reason = `NAV-CLOCK clkD is receiver clock drift (${clkDrift.toFixed(2)} ns/s). Unstable drift can indicate spoofing.`;
  featureImportanceData[2].value = Math.min(1, dopJump / 2.8);
  featureImportanceData[2].reason = `NAV-DOP pDOP measures satellite geometry quality. Sudden change (+${dopJump.toFixed(2)}) can indicate abnormal geometry.`;
  featureImportanceData[3].value = Math.min(1, ecefJump / 30);
  featureImportanceData[3].reason = `NAV-POSECEF shows receiver position in ECEF coordinates. Sudden jump (${ecefJump.toFixed(1)} m) can indicate spoofing.`;
  featureImportanceData[4].value = Math.min(1, cnoDrop / 17);
  featureImportanceData[4].reason = `NAV-SAT C/N0 is signal strength. A wide drop (${cnoDrop.toFixed(1)} dB-Hz) can indicate jamming.`;
  featureImportanceData[5].value = Math.min(1, bandRise / 20);
  featureImportanceData[5].reason = `MON-SPAN is RF spectrum power. Broad rise (+${bandRise.toFixed(1)} dB) can indicate jamming.`;

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
  state.rfSeries.band1583.push(Math.max(0, Math.min(40, band1583)));
  state.rfSeries.band1583.shift();
  state.rfSeries.band1224.push(Math.max(0, Math.min(40, band1224)));
  state.rfSeries.band1224.shift();

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
      { name: "1583MHz", data: state.rfSeries.band1583, color: "#0f9ea8" },
      { name: "1224MHz", data: state.rfSeries.band1224, color: "#2b74c7" },
    ],
    {
      yMin: 0,
      yMax: 40,
      yLabel: "Power (dB)",
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
  bindTooltipDismissal();
  setupCanvasTooltips();
  setupTabs();
  setupFilters();

  for (let i = 0; i < 12; i += 1) {
    pushRandomLog();
  }

  const tick = async () => {
    try {
      const sample = await fetchNextSample();
      ingestLiveSample(sample);
    } catch (error) {
      simulateMainState();
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

  setInterval(tick, 2200);

  setInterval(() => {
    renderMainStatus();
    renderHealth();
  }, 1000);

  window.addEventListener("resize", () => {
    renderAll();
  });
}

boot();
