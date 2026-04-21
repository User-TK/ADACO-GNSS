#!/usr/bin/env python3
"""
Standalone ADACO-GNSS local demo.

This replaces the FastAPI/Uvicorn demo server with one directly runnable Python file.

Expected folder layout:

  standalone_demo.py
  cnn_best_gpu_run1.pt
  spectrum_cnn.py          # required if cnn_best_gpu_run1.pt stores only model_state
  demodata/
    0.json
    1.json
    ...

Run:

  python standalone_demo.py

Optional:

  python standalone_demo.py --model models/cnn_best_gpu_run1.pt --data demodata --host 127.0.0.1 --port 8000

Then open:

  http://127.0.0.1:8000/

Important:
- The web server uses only Python stdlib.
- Model inference still requires torch + numpy.
- If your checkpoint stores a sklearn StandardScaler, torch.load may also require scikit-learn
  to be installed because the scaler object must be unpickled.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


LABEL_NAMES = {0: "Clean", 1: "Spoofing", 2: "Jamming"}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ADACO-GNSS Standalone Model Demo</title>
  <style>
    :root {
      --bg-0: #f4f9fb;
      --bg-1: #e5f2f4;
      --panel: rgba(255, 255, 255, 0.82);
      --line: #d4e3e5;
      --text: #102127;
      --muted: #567079;
      --accent: #087f8c;
      --danger: #c7363f;
      --success: #1e8f4d;
      --radius-lg: 20px;
      --radius-md: 14px;
      --shadow: 0 18px 42px rgba(19, 62, 70, 0.12);
    }

    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      margin: 0;
    }

    body {
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 8%, #ffffff 0%, rgba(255, 255, 255, 0) 45%),
        radial-gradient(circle at 88% 92%, #dbf1f7 0%, rgba(219, 241, 247, 0) 48%),
        linear-gradient(125deg, var(--bg-0), var(--bg-1));
    }

    h2, p { margin: 0; }

    .dashboard-shell {
      display: grid;
      grid-template-columns: minmax(300px, 28%) minmax(0, 72%);
      gap: 16px;
      height: 100dvh;
      padding: 16px;
    }

    .summary-column {
      min-height: 0;
      display: grid;
      grid-template-rows: 1.15fr 0.85fr;
      gap: 16px;
    }

    .card {
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }

    .status-card,
    .overview-card {
      display: grid;
      gap: 16px;
      padding: 18px;
      align-content: start;
    }

    .chart-card {
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      padding: 18px;
    }

    .card-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }

    .card-header.stacked { align-items: center; }
    .chart-header { margin-bottom: 10px; }

    .status-meta {
      display: grid;
      gap: 14px;
    }

    .meta-row {
      display: grid;
      gap: 8px;
    }

    .meta-label {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }

    .meta-value {
      color: var(--text);
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.9rem;
      overflow-wrap: anywhere;
    }

    .error-text { color: var(--danger); }

    .status-pill,
    .threat-pill {
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      border-radius: 999px;
      padding: 6px 12px;
      border: 1px solid transparent;
      white-space: nowrap;
    }

    .status-pill.normal {
      color: var(--success);
      background: #e6f7ee;
      border-color: #bce6ce;
    }

    .status-pill.danger {
      color: var(--danger);
      background: #fdeced;
      border-color: #f5b9be;
    }

    .threat-indicators {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .threat-pill {
      color: #6b8087;
      background: #edf2f4;
      border-color: #d9e3e6;
    }

    .threat-pill.active.clean {
      color: #ffffff;
      background: var(--success);
      border-color: #157a3e;
    }

    .threat-pill.active.threat {
      color: #ffffff;
      background: var(--danger);
      border-color: #b51e27;
    }

    .overview-copy {
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.96rem;
    }

    .confidence-value {
      font-family: Consolas, "Courier New", monospace;
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--accent);
    }

    .progress-wrap {
      height: 18px;
      border-radius: 999px;
      background: #e6eff1;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #2fb6ba);
      transition: width 0.45s ease;
    }

    .chart-caption {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.86rem;
    }

    .chart-area {
      min-height: 0;
      height: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(255, 255, 255, 0.9);
      overflow: hidden;
    }

    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    @media (max-width: 980px) {
      .dashboard-shell {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(420px, 1fr);
      }

      .summary-column {
        grid-template-rows: auto auto;
      }
    }
  </style>
</head>
<body>
  <main class="dashboard-shell">
    <section class="summary-column">
      <article class="card status-card">
        <div class="card-header">
          <h2>System Status</h2>
          <span class="status-pill normal" id="systemStatus">NORMAL</span>
        </div>

        <div class="status-meta">
          <div class="meta-row">
            <span class="meta-label">Predicted class</span>
            <span class="meta-value" id="predictedLabel">Waiting for model...</span>
          </div>

          <div class="meta-row">
            <span class="meta-label">Input sample</span>
            <span class="meta-value" id="sourceFile">--</span>
          </div>

          <div class="meta-row">
            <span class="meta-label">Class flags</span>
            <div class="threat-indicators">
              <span class="threat-pill" id="cleanIndicator">Clean</span>
              <span class="threat-pill" id="spoofingIndicator">Spoofing</span>
              <span class="threat-pill" id="jammingIndicator">Jamming</span>
            </div>
          </div>

          <div class="meta-row">
            <span class="meta-label">First threat detected</span>
            <span class="meta-value" id="firstDetected">--:--:-- -- (-- ago)</span>
          </div>

          <div class="meta-row" id="errorRow" hidden>
            <span class="meta-label">Demo status</span>
            <span class="meta-value error-text" id="errorText"></span>
          </div>
        </div>
      </article>

      <article class="card overview-card">
        <div class="card-header stacked">
          <h2>Model Overview</h2>
          <span class="confidence-value" id="confidenceScore">0%</span>
        </div>

        <div class="overview-copy">
          Confidence of the model-selected class from the current randomly sampled JSON input.
        </div>

        <div class="progress-wrap" aria-label="Confidence score bar">
          <div class="progress-fill" id="confidenceBar"></div>
        </div>
      </article>
    </section>

    <section class="card chart-card">
      <div class="card-header chart-header">
        <div>
          <h2>Attack Scores by Type</h2>
          <p class="chart-caption">Recent window • model probability from 0 to 1</p>
        </div>
      </div>
      <div class="chart-area">
        <canvas id="attackScoreChart"></canvas>
      </div>
    </section>
  </main>

<script>
const DEMO_API_URL = "/api/random-prediction";
const POLL_INTERVAL_MS = 2200;
const HISTORY_LENGTH = 30;

const LABELS = { 0: "Clean", 1: "Spoofing", 2: "Jamming" };

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
  while (series.length > HISTORY_LENGTH) series.shift();
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

  setStatusPill(
    statusPill,
    state.systemMode === "danger" ? "DANGER" : "NORMAL",
    state.systemMode === "danger" ? "danger" : "normal"
  );

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
    ctx.font = "10px Consolas";
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
    ctx.font = "10px Consolas";
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
    ctx.font = "10px Consolas";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(line.name, legendX + 18, 10);
    legendX += 104;
  });

  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5a747d";
  ctx.font = "10px Consolas";
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

  state.label = Number.isInteger(prediction.label)
    ? prediction.label
    : probabilities.indexOf(Math.max(clean, spoofing, jamming));

  state.labelName = prediction.label_name || LABELS[state.label] || "Unknown";
  state.confidence = clamp(prediction.confidence ?? probabilities[state.label] ?? 0, 0, 1);
  state.sourceFile = payload.source_file || "--";

  const threatDetected = state.label === 1 || state.label === 2;
  state.systemMode = threatDetected ? "danger" : "normal";

  if (threatDetected && !state.firstDetected) state.firstDetected = new Date();
  if (!threatDetected) state.firstDetected = null;

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
        const body = await response.json();
        detail = body.error || body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    applyPrediction(await response.json());
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
</script>
</body>
</html>
"""


def load_checkpoint(path: Path, device: torch.device) -> Any:
    """Load checkpoint while supporting older PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalise_spectrum(spec: np.ndarray) -> np.ndarray:
    """Z-score per RF block, matching the training dataset preprocessing."""
    mean = spec.mean(axis=-1, keepdims=True)
    std = spec.std(axis=-1, keepdims=True) + 1e-6
    return (spec - mean) / std


def require_list(payload: Dict[str, Any], key: str, expected_len: int) -> List[float]:
    if key not in payload:
        raise ValueError(f"Missing key '{key}'")

    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be a list, got {type(value).__name__}")

    if len(value) != expected_len:
        raise ValueError(f"'{key}' must contain {expected_len} values, got {len(value)}")

    try:
        return [float(x) for x in value]
    except Exception as exc:
        raise ValueError(f"'{key}' must contain only numeric values") from exc


def validate_gnss_input(payload: Dict[str, Any]) -> Dict[str, List[float]]:
    return {
        "nav_pvt": require_list(payload, "nav_pvt", 15),
        "nav_clock": require_list(payload, "nav_clock", 4),
        "nav_dop": require_list(payload, "nav_dop", 7),
        "nav_posecef": require_list(payload, "nav_posecef", 4),
        "spectrum_01": require_list(payload, "spectrum_01", 256),
        "spectrum_02": require_list(payload, "spectrum_02", 256),
    }


class GNSSPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path.resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"[demo] Loading model from {self.model_path}")
        print(f"[demo] Device: {self.device}")

        ckpt = load_checkpoint(self.model_path, self.device)

        self.scaler = None

        if isinstance(ckpt, dict):
            self.scaler = ckpt.get("scaler")

            if "model" in ckpt:
                self.model = ckpt["model"].to(self.device)
            elif "model_state" in ckpt:
                # Your current predictor.py constructs HybridGNSSCNN and then loads ckpt["model_state"].
                # Therefore this standalone file needs spectrum_cnn.py next to it or on PYTHONPATH.
                try:
                    from spectrum_cnn import HybridGNSSCNN
                except Exception as exc:
                    raise ImportError(
                        "cnn_best_gpu_run1.pt appears to store only model_state, so this demo must import "
                        "HybridGNSSCNN from spectrum_cnn.py. Put spectrum_cnn.py in the same folder "
                        "as standalone_demo.py, or paste the HybridGNSSCNN class into this file."
                    ) from exc

                self.model = HybridGNSSCNN(in_channels=2, scalar_dim=30).to(self.device)
                self.model.load_state_dict(ckpt["model_state"])
            else:
                # Sometimes people save the bare state_dict directly.
                # This branch tries to treat the checkpoint itself as a state_dict.
                try:
                    from spectrum_cnn import HybridGNSSCNN
                except Exception as exc:
                    raise ImportError(
                        "Could not find a full model object in the .pt file, and spectrum_cnn.py "
                        "is required to rebuild the model architecture."
                    ) from exc

                self.model = HybridGNSSCNN(in_channels=2, scalar_dim=30).to(self.device)
                self.model.load_state_dict(ckpt)
        else:
            # Full model object was saved directly with torch.save(model, path).
            self.model = ckpt.to(self.device)

        if self.scaler is None:
            raise ValueError(
                "No 'scaler' found in checkpoint. Your training code likely saved a StandardScaler "
                "inside cnn_best_gpu_run1.pt. This demo expects ckpt['scaler'] so scalar features are normalized "
                "the same way they were during training."
            )

        self.model.eval()
        print("[demo] Model loaded and ready")

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = validate_gnss_input(payload)

        scalar = np.array(
            data["nav_pvt"] + data["nav_clock"] + data["nav_dop"] + data["nav_posecef"],
            dtype=np.float32,
        ).reshape(1, -1)

        scalar = self.scaler.transform(scalar)

        s1 = np.array(data["spectrum_01"], dtype=np.float32)
        s2 = np.array(data["spectrum_02"], dtype=np.float32)
        spec = np.stack([s1, s2], axis=0)[np.newaxis]  # (1, 2, 256)
        spec = normalise_spectrum(spec)

        with torch.no_grad():
            spec_t = torch.from_numpy(spec).to(self.device)
            scalar_t = torch.from_numpy(scalar).to(self.device)
            logits = self.model(spec_t, scalar_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            label = int(probs.argmax())

        return {
            "label": label,
            "label_name": LABEL_NAMES.get(label, f"Class {label}"),
            "confidence": float(probs[label]),
            "probabilities": [float(x) for x in probs.tolist()],
        }


class DemoHandler(BaseHTTPRequestHandler):
    predictor: GNSSPredictor | None = None
    data_dir: Path | None = None

    server_version = "ADACOStandaloneDemo/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[http] " + fmt % args + "\n")

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, status: int, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.send_text(200, HTML)
            return

        if path == "/api/health":
            self.send_json(200, {
                "status": "ok",
                "model_loaded": self.predictor is not None,
                "data_dir": str(self.data_dir),
            })
            return

        if path == "/api/random-prediction":
            self.handle_random_prediction(parsed.query)
            return

        self.send_json(404, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_json(404, {"error": f"Unknown route: {parsed.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            prediction = self.predictor.predict(payload)  # type: ignore[union-attr]
            self.send_json(200, prediction)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_random_prediction(self, query: str) -> None:
        try:
            if self.predictor is None:
                self.send_json(503, {"error": "Model not loaded"})
                return

            if self.data_dir is None:
                self.send_json(500, {"error": "No demodata directory configured"})
                return

            files = sorted(self.data_dir.glob("*.json"))
            if not files:
                self.send_json(404, {"error": f"No .json files found in {self.data_dir}"})
                return

            sample_path = random.choice(files)
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            prediction = self.predictor.predict(payload)

            params = parse_qs(query)
            include_input = params.get("include_input", ["false"])[0].lower() in {"1", "true", "yes"}

            response = {
                "source_file": sample_path.name,
                "prediction": prediction,
            }
            if include_input:
                response["input"] = payload

            self.send_json(200, response)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ADACO-GNSS model demo")
    parser.add_argument(
        "--model",
        default="cnn_best_gpu_run1.pt",
        help="Path to cnn_best_gpu_run1.pt. Default: ./cnn_best_gpu_run1.pt",
    )
    parser.add_argument(
        "--data",
        default="demodata",
        help="Path to demodata folder containing .json samples. Default: ./demodata",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port. Default: 8000",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)

    if not data_dir.exists():
        raise FileNotFoundError(f"Demodata directory not found: {data_dir.resolve()}")

    DemoHandler.predictor = GNSSPredictor(model_path)
    DemoHandler.data_dir = data_dir.resolve()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}/"

    print(f"[demo] Serving dashboard at {url}")
    print("[demo] Press Ctrl+C to stop")

    if not args.no_browser:
        threading.Timer(0.75, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] Stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
