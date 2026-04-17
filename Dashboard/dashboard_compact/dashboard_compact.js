const state = {
  systemMode: "normal",
  confidence: 0.22,
  activeThreats: {
    spoofing: false,
    jamming: false,
    neither: false,
    both: false,
  },
  firstDetected: null,
  attackSeries: {
    spoofing: new Array(30).fill(0.18),
    jamming: new Array(30).fill(0.14),
    neither: new Array(30).fill(0.1),
    both: new Array(30).fill(0.1),
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

function renderStatus() {
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
    ["spoofing", "spoofingIndicator"],
    ["jamming", "jammingIndicator"],
    ["neither", "neitherIndicator"],
    ["both", "bothIndicator"],
  ].forEach(([key, id]) => {
    document.getElementById(id).classList.toggle("active", state.activeThreats[key]);
  });

  const detectText = document.getElementById("firstDetected");
  if (!state.firstDetected) {
    detectText.textContent = "--:--:-- -- (-- ago)";
  } else {
    const elapsed = Date.now() - state.firstDetected.getTime();
    detectText.textContent = `${formatTime(state.firstDetected)} (${formatDuration(elapsed)} ago)`;
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
    { name: "Spoofing", data: state.attackSeries.spoofing, color: "#d39419" },
    { name: "Jamming", data: state.attackSeries.jamming, color: "#8f4bc9" },
    { name: "Neither", data: state.attackSeries.neither, color: "#2b74c7" },
    { name: "Both", data: state.attackSeries.both, color: "#2b74c7" },
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
    ctx.fillText(remaining === 0 ? "now" : `-${remaining}s`, x, h - margin.bottom + 6);
  }

  lines.forEach((line) => {
    ctx.beginPath();
    line.data.forEach((value, i) => {
      const x = margin.left + (plotW / (line.data.length - 1)) * i;
      const y = margin.top + (1 - value) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.4;
    ctx.stroke();

    const endValue = line.data[line.data.length - 1];
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
    legendX += 94;
  });

  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5a747d";
  ctx.font = "10px IBM Plex Mono";
  ctx.textAlign = "center";
  ctx.fillText("Risk Score", 0, 0);
  ctx.restore();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateSeries(series, nextValue) {
  series.push(clamp(nextValue, 0, 1));
  series.shift();
}

function simulateState() {
  const elevated = Math.random() > 0.66;

  const spoofScore = elevated ? 0.5 + Math.random() * 0.42 : 0.1 + Math.random() * 0.34;
  const jamScore = elevated ? 0.44 + Math.random() * 0.46 : 0.08 + Math.random() * 0.3;
  const neitherScore = elevated ? 0.4 + Math.random() * 0.45 : 0.06 + Math.random() * 0.26;
  const bothScore = elevated ? 0.4 + Math.random() * 0.45 : 0.06 + Math.random() * 0.26;

  updateSeries(state.attackSeries.spoofing, spoofScore);
  updateSeries(state.attackSeries.jamming, jamScore);
  updateSeries(state.attackSeries.neither, neitherScore);
  updateSeries(state.attackSeries.both, bothScore);

  state.activeThreats.spoofing = spoofScore > 0.66;
  state.activeThreats.jamming = jamScore > 0.66;
  state.activeThreats.neither = neitherScore > 0.66;
  state.activeThreats.both = bothScore > 0.66;

  const anyThreat = Object.values(state.activeThreats).some(Boolean);
  state.systemMode = anyThreat ? "danger" : "normal";
  state.confidence = Math.max(spoofScore, jamScore, bothScore);

  if (anyThreat && !state.firstDetected) {
    state.firstDetected = new Date();
  }
  if (!anyThreat) {
    state.firstDetected = null;
  }
}

function renderAll() {
  renderStatus();
  drawAttackChart();
}

function boot() {
  simulateState();
  renderAll();

  setInterval(() => {
    simulateState();
    renderAll();
  }, 2200);

  setInterval(() => {
    renderStatus();
  }, 1000);

  window.addEventListener("resize", renderAll);
}

boot();
