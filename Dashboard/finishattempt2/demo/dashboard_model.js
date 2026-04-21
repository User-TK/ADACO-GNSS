const DEMO_API_URL = window.DEMO_API_URL || "/demo-api/random-prediction";
const POLL_INTERVAL_MS = 2200;
const HISTORY_LENGTH = 30;

const LABELS = {
  0: "Clean",
  1: "Spoofing",
  2: "Jamming",
};

const state = {
  systemMode: "normal",
  label: 0,
  labelName: "Clean",
  confidence: 0,
  sourceFile: "--",
  firstDetected: null,
  lastError: "",
  attackSeries: {
    clean: new Array(HISTORY_LENGTH).fill(0),
    spoofing: new Array(HISTORY_LENGTH).fill(0),
    jamming: new Array(HISTORY_LENGTH).fill(0),
  },
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

function setStatusPill(el, text, cls) {
  el.classList.remove("normal", "danger");
  el.classList.add(cls);
  el.textContent = text;
}

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function updateSeries(series, nextValue) {
  series.push(clamp(nextValue, 0, 1));
  while (series.length > HISTORY_LENGTH) {
    series.shift();
  }
}

function setActiveClass(id, isActive, kind) {
  const el = document.getElementById(id);
  el.classList.toggle("active", isActive);
  el.classList.toggle("clean", isActive && kind === "clean");
  el.classList.toggle("threat", isActive && kind === "threat");
}

function renderStatus() {
  const statusPill = document.getElementById("systemStatus");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidenceScore = document.getElementById("confidenceScore");
  const predictedLabel = document.getElementById("predictedLabel");
  const sourceFile = document.getElementById("sourceFile");
  const errorRow = document.getElementById("errorRow");
  const errorText = document.getElementById("errorText");

  if (state.systemMode === "danger") {
    setStatusPill(statusPill, "DANGER", "danger");
  } else {
    setStatusPill(statusPill, "NORMAL", "normal");
  }

  confidenceBar.style.width = `${Math.round(state.confidence * 100)}%`;
  confidenceScore.textContent = `${Math.round(state.confidence * 100)}%`;
  predictedLabel.textContent = `${state.labelName} (${state.confidence.toFixed(4)})`;
  sourceFile.textContent = state.sourceFile;

  setActiveClass("cleanIndicator", state.label === 0, "clean");
  setActiveClass("spoofingIndicator", state.label === 1, "threat");
  setActiveClass("jammingIndicator", state.label === 2, "threat");

  const detectText = document.getElementById("firstDetected");
  if (!state.firstDetected) {
    detectText.textContent = "--:--:-- -- (-- ago)";
  } else {
    const elapsed = Date.now() - state.firstDetected.getTime();
    detectText.textContent = `${formatTime(state.firstDetected)} (${formatDuration(elapsed)} ago)`;
  }

  if (state.lastError) {
    errorText.textContent = state.lastError;
    errorRow.hidden = false;
  } else {
    errorText.textContent = "";
    errorRow.hidden = true;
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

function drawAttackChart() {
  const lines = [
    { name: "Clean", data: state.attackSeries.clean, color: "#1e8f4d" },
    { name: "Spoofing", data: state.attackSeries.spoofing, color: "#d39419" },
    { name: "Jamming", data: state.attackSeries.jamming, color: "#8f4bc9" },
  ];

  const { ctx, w, h } = getCanvasContext2D("attackScoreChart");
  const margin = { top: 20, right: 20, bottom: 48, left: 46 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  if (plotW <= 10 || plotH <= 10) return;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fcffff";
  ctx.fillRect(0, 0, w, h);

  const yTicks = 5;
  ctx.strokeStyle = "#deebee";
  ctx.lineWidth = 1;
  for (let i = 0; i <= yTicks; i += 1) {
    const y = margin.top + (plotH / yTicks) * i;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(w - margin.right, y);
    ctx.stroke();

    const value = 1 - i / yTicks;
    ctx.fillStyle = "#5a747d";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(1), margin.left - 6, y);
  }

  ctx.strokeStyle = "#92adb4";
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, h - margin.bottom);
  ctx.lineTo(w - margin.right, h - margin.bottom);
  ctx.stroke();

  const xTicks = 4;
  const totalSeconds = Math.round((lines[0].data.length - 1) * (POLL_INTERVAL_MS / 1000));
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
    ctx.fillText(remaining === 0 ? "now" : `-${remaining}s`, x, h - margin.bottom + 6);
  }

  lines.forEach((line) => {
    ctx.beginPath();
    line.data.forEach((value, i) => {
      const x = margin.left + (plotW / (line.data.length - 1)) * i;
      const y = margin.top + (1 - clamp(value, 0, 1)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.4;
    ctx.stroke();

    const endValue = clamp(line.data[line.data.length - 1], 0, 1);
    const endY = margin.top + (1 - endValue) * plotH;
    const endX = w - margin.right;
    ctx.beginPath();
    ctx.fillStyle = line.color;
    ctx.arc(endX, endY, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  let legendX = margin.left;
  lines.forEach((line) => {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(legendX, 10);
    ctx.lineTo(legendX + 14, 10);
    ctx.stroke();
    ctx.fillStyle = "#48656d";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(line.name, legendX + 18, 10);
    legendX += 104;
  });

  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5a747d";
  ctx.font = "10px IBM Plex Mono";
  ctx.textAlign = "center";
  ctx.fillText("Probability", 0, 0);
  ctx.restore();
}

function applyPrediction(payload) {
  const prediction = payload.prediction || payload;
  const probabilities = prediction.probabilities || [];

  const clean = clamp(probabilities[0], 0, 1);
  const spoofing = clamp(probabilities[1], 0, 1);
  const jamming = clamp(probabilities[2], 0, 1);

  updateSeries(state.attackSeries.clean, clean);
  updateSeries(state.attackSeries.spoofing, spoofing);
  updateSeries(state.attackSeries.jamming, jamming);

  state.label = Number.isInteger(prediction.label) ? prediction.label : probabilities.indexOf(Math.max(clean, spoofing, jamming));
  state.labelName = prediction.label_name || LABELS[state.label] || "Unknown";
  state.confidence = clamp(prediction.confidence ?? probabilities[state.label] ?? 0, 0, 1);
  state.sourceFile = payload.source_file || "--";

  const threatDetected = state.label === 1 || state.label === 2;
  state.systemMode = threatDetected ? "danger" : "normal";

  if (threatDetected && !state.firstDetected) {
    state.firstDetected = new Date();
  }
  if (!threatDetected) {
    state.firstDetected = null;
  }

  state.lastError = "";
}

async function fetchPrediction() {
  try {
    const response = await fetch(DEMO_API_URL, {
      method: "GET",
      headers: { "Accept": "application/json" },
      cache: "no-store",
    });

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const errorBody = await response.json();
        detail = errorBody.detail || detail;
      } catch {
        // Keep status text if response is not JSON.
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    applyPrediction(payload);
  } catch (error) {
    state.lastError = error.message || String(error);
  }

  renderAll();
}

function renderAll() {
  renderStatus();
  drawAttackChart();
}

function boot() {
  renderAll();
  fetchPrediction();

  setInterval(fetchPrediction, POLL_INTERVAL_MS);
  setInterval(renderStatus, 1000);
  window.addEventListener("resize", renderAll);
}

boot();
