from __future__ import annotations

import os
import importlib
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, jsonify, redirect, request, send_from_directory

try:
    psutil = importlib.import_module("psutil")
except ImportError:  # pragma: no cover - optional dependency fallback
    psutil = None


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATASET_PATH = ROOT / "datasets" / "gnss_dataset.h5"
DASHBOARD_DIR = ROOT / "Dashboard"
FULL_DASHBOARD_DIR = ROOT / "Dashboard"
CHECKPOINT_PATH = MODEL_DIR / "cnn_best_gpu_run1.pt"

sys.path.insert(0, str(MODEL_DIR))

from inference import GNSSDetector  # pyright: ignore[reportMissingImports]


class LiveGNSSStream:
    def __init__(self, h5_path: Path, detector: GNSSDetector):
        self._h5 = h5py.File(h5_path, "r")
        self._detector = detector
        self._lock = threading.Lock()
        self._cursor = 0
        self._last_sample: dict | None = None
        self._last_sample_at: datetime | None = None
        self._history: deque[dict] = deque(maxlen=80)
        self.total_samples = int(self._h5["day"].shape[0])
        self._process = psutil.Process() if psutil is not None else None

    def close(self) -> None:
        self._h5.close()

    @staticmethod
    def _decode_day(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _build_scalar(self, index: int) -> np.ndarray:
        return np.concatenate(
            [
                self._h5["nav_pvt"][index],
                self._h5["nav_clock"][index],
                self._h5["nav_dop"][index],
                self._h5["nav_posecef"][index],
            ]
        ).astype(np.float32, copy=False)

    def next_sample(self) -> dict:
        started_at = time.perf_counter()
        with self._lock:
            index = self._cursor
            self._cursor = (self._cursor + 1) % self.total_samples

        spectrum = np.stack(
            [
                self._h5["spectrum_01"][index],
                self._h5["spectrum_02"][index],
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        scalar = self._build_scalar(index)
        prediction = self._detector.predict_single_with_importance(spectrum, scalar)
        inference_ms = (time.perf_counter() - started_at) * 1000.0
        processed_at = datetime.now(timezone.utc)

        sat = self._h5["nav_sat"][index]
        sv_id = sat[:, 1]
        cno = sat[:, 2]
        pr_res = sat[:, 5]
        valid_sat = sv_id != 255
        valid_count = int(np.count_nonzero(valid_sat))
        mean_cno = float(cno[valid_sat].mean()) if valid_count else 0.0
        max_abs_pr_res = float(np.abs(pr_res[valid_sat]).max()) if valid_count else 0.0
        clock_drift = float(self._h5["nav_clock"][index][1])
        pdop = float(self._h5["nav_dop"][index][1])

        day = self._decode_day(self._h5["day"][index])
        hour = int(self._h5["hour"][index])
        ground_truth = int(self._h5["label"][index])

        probabilities = prediction["probabilities"]
        combined_attack = float(
            min(1.0, probabilities["Spoofing"] + probabilities["Jamming"])
        )
        rf_band_1583 = float(np.mean(spectrum[0]))
        rf_band_1224 = float(np.mean(spectrum[1]))
        rf_band_delta = rf_band_1583 - rf_band_1224
        display_label = "Safe" if prediction["class_id"] == 0 else prediction["label"]

        prediction["display_label"] = display_label

        sample = {
            "index": index,
            "total": self.total_samples,
            "day": day,
            "hour": hour,
            "ground_truth": ground_truth,
            "processed_at": processed_at.isoformat(),
            "timing": {
                "inference_ms": inference_ms,
            },
            "prediction": prediction,
            "satellite": {
                "num_sv_used": int(self._h5["nav_pvt"][index][2]),
                "num_visible": valid_count,
                "mean_cno": mean_cno,
                "max_abs_pr_res": max_abs_pr_res,
                "clock_drift": clock_drift,
                "pdop": pdop,
            },
            "derived": {
                "safe": float(probabilities["Clean"]),
                "clean": float(probabilities["Clean"]),
                "spoofing": float(probabilities["Spoofing"]),
                "jamming": float(probabilities["Jamming"]),
                "combined_attack": combined_attack,
                "rf_band_1583": rf_band_1583,
                "rf_band_1224": rf_band_1224,
                "rf_band_delta": rf_band_delta,
            },
        }

        with self._lock:
            self._last_sample = sample
            self._last_sample_at = processed_at
            self._append_history_entry(sample)

        return sample

    def set_cursor(self, index: int) -> int:
        with self._lock:
            self._cursor = int(index) % self.total_samples
            return self._cursor

    @staticmethod
    def _feature_to_history_type(feature_name: str, prediction_class_id: int) -> str:
        feature = feature_name.lower()
        # Explicitly mark safe predictions
        if prediction_class_id == 0:
            return "safe"
        if "clock" in feature:
            return "clock-anomaly"
        if "dop" in feature or "geometry" in feature:
            return "sat-geometry"
        if "c/n0" in feature or "band" in feature:
            return "jamming"
        if prediction_class_id == 1:
            return "spoofing"
        if prediction_class_id == 2:
            return "jamming"
        return "spoofing"

    def _append_history_entry(self, sample: dict) -> None:
        top_features = sample["prediction"].get("feature_importance", {}).get("top_features", [])
        top_feature = top_features[0] if top_features else {}
        class_id = int(sample["prediction"].get("class_id", 0))
        entry_type = self._feature_to_history_type(
            str(top_feature.get("name", "spoofing")),
            class_id,
        )

        if entry_type == "clock-anomaly":
            stream_name = "NAV-CLOCK"
        elif entry_type == "sat-geometry":
            stream_name = "NAV-DOP"
        elif entry_type == "jamming":
            stream_name = "MON-SPAN"
        elif entry_type == "safe":
            stream_name = "NAV-SAT"
        else:
            stream_name = "NAV-SAT"

        satellite = sample["satellite"]
        confidence = float(sample["prediction"].get("confidence", 0.0))
        label = sample["prediction"].get("display_label") or sample["prediction"].get("label", "Unknown")
        reason = str(top_feature.get("reason", "Processed sample"))

        inference_ms = None
        try:
            inference_ms = float(sample.get("timing", {}).get("inference_ms", 0.0))
        except Exception:
            inference_ms = None

        # Build a clear, human-readable message (no leading sample index)
        message_parts = []
        message_parts.append(f"Model: {label} ({confidence * 100:.1f}%)")
        if reason:
            message_parts.append(f"Reason: {reason}")
        # Expand SVs to readable labels
        message_parts.append(f"Visible Satellites: {satellite.get('num_visible', '?')}")
        message_parts.append(f"Satellites Used: {satellite.get('num_sv_used', '?')}")
        message_parts.append(f"Mean C/N0: {satellite.get('mean_cno', 0.0):.1f} dB-Hz")
        message_parts.append(f"Max |prRes|: {satellite.get('max_abs_pr_res', 0.0):.2f} m")
        if inference_ms is not None:
            message_parts.append(f"Inference: {inference_ms:.1f} ms")

        message = " | ".join(message_parts)

        # Use empty stream name for generic entries to avoid confusing labels like 'NAV-SAT'
        self._history.appendleft(
            {
                "ts": sample.get("processed_at"),
                "type": entry_type,
                "stream": "",
                "label": label,
                "confidence": confidence,
                "satellite": f"Visible Satellites: {satellite['num_visible']} / Used: {satellite['num_sv_used']}",
                "message": message,
            }
        )

    def health_snapshot(self) -> dict:
        now = datetime.now(timezone.utc)

        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(ROOT))
            boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
            load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
            process = self._process or psutil.Process()
            process_rss_mb = process.memory_info().rss / (1024 * 1024)
            thread_count = process.num_threads()
        else:  # pragma: no cover - exercised only when psutil is unavailable
            cpu_percent = 0.0
            memory = type("Memory", (), {"percent": 0.0})()
            disk = type("Disk", (), {"percent": 0.0})()
            boot_time = now
            load_average = None
            process_rss_mb = 0.0
            thread_count = threading.active_count()

        with self._lock:
            sample = self._last_sample
            sample_at = self._last_sample_at
            history = list(self._history)

        inference_ms = float(sample["timing"]["inference_ms"]) if sample else None
        confidence = float(sample["prediction"]["confidence"]) if sample else None
        sample_age_seconds = (
            max(0.0, (now - sample_at).total_seconds()) if sample_at is not None else None
        )

        if cpu_percent >= 92.0 or memory.percent >= 92.0 or disk.percent >= 95.0:
            system_status = "critical"
        elif cpu_percent >= 75.0 or memory.percent >= 80.0 or disk.percent >= 90.0:
            system_status = "degraded"
        else:
            system_status = "healthy"

        if sample is None:
            model_status = "delayed"
            data_status = "lagging"
        else:
            model_status = (
                "active"
                if inference_ms is not None and inference_ms < 150.0 and confidence is not None and confidence >= 0.5
                else "delayed"
            )
            if sample_age_seconds is None or sample_age_seconds < 3.5:
                data_status = "ingesting"
            elif sample_age_seconds < 10.0:
                data_status = "lagging"
            else:
                data_status = "dropped"

        return {
            "updated_at": now.isoformat(),
            "uptime_seconds": max(0.0, (now - boot_time).total_seconds()),
            "boot_time": boot_time.isoformat(),
            "system": {
                "status": system_status,
                "metrics": [
                    {"name": "CPU Usage", "value": f"{cpu_percent:.1f}%"},
                    {"name": "Memory Usage", "value": f"{memory.percent:.1f}%"},
                    {"name": "Disk Usage", "value": f"{disk.percent:.1f}%"},
                    {"name": "Load Average", "value": f"{load_average:.2f}" if load_average is not None else "n/a"},
                    {"name": "Process RSS", "value": f"{process_rss_mb:.1f} MB"},
                    {"name": "Thread Count", "value": str(thread_count)},
                ],
            },
            "model": {
                "status": model_status,
                "metrics": [
                    {"name": "Model Version", "value": CHECKPOINT_PATH.name},
                    {"name": "Device", "value": str(self._detector.device)},
                    {"name": "Inference Latency", "value": f"{inference_ms:.2f} ms" if inference_ms is not None else "n/a"},
                    {"name": "Predictions/s", "value": f"{(1000.0 / inference_ms):.1f} /s" if inference_ms else "n/a"},
                    {"name": "Batch Size", "value": "1"},
                    {"name": "Confidence", "value": f"{confidence * 100:.1f}%" if confidence is not None else "n/a"},
                ],
            },
            "data": {
                "status": data_status,
                "metrics": [
                    {"name": "Sample Progress", "value": f"{sample['index'] + 1}/{sample['total']}" if sample else f"0/{self.total_samples}"},
                    {"name": "Day", "value": str(sample["day"]) if sample else "n/a"},
                    {"name": "Hour", "value": f"{int(sample['hour']):02d}:00" if sample else "n/a"},
                    {"name": "Visible SVs", "value": str(sample["satellite"]["num_visible"]) if sample else "n/a"},
                    {"name": "Mean C/N0", "value": f"{sample['satellite']['mean_cno']:.2f} dB-Hz" if sample else "n/a"},
                    {"name": "Max |prRes|", "value": f"{sample['satellite']['max_abs_pr_res']:.2f} m" if sample else "n/a"},
                ],
            },
            "history": history,
        }


app = Flask(__name__)
detector = GNSSDetector(str(CHECKPOINT_PATH))
stream = LiveGNSSStream(DATASET_PATH, detector)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def root():
    return redirect("/dashboard/index.html")


@app.get("/dashboard/")
@app.get("/dashboard/index.html")
def dashboard_full():
    return send_from_directory(FULL_DASHBOARD_DIR, "index.html")


@app.get("/dashboard/<path:filename>")
def dashboard_full_assets(filename: str):
    return send_from_directory(FULL_DASHBOARD_DIR, filename)


@app.get("/dashboard_compact/")
@app.get("/dashboard_compact/dashboard_compact.html")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, "dashboard_compact.html")


@app.get("/dashboard_compact/<path:filename>")
def dashboard_assets(filename: str):
    return send_from_directory(DASHBOARD_DIR, filename)


@app.get("/api/meta")
def api_meta():
    return jsonify(
        {
            "model": "cnn_best_gpu_run1.pt",
            "dataset": "gnss_dataset.h5",
            "total_samples": stream.total_samples,
            "device": str(detector.device),
            "classes": ["Safe", "Spoofing", "Jamming"],
        }
    )


@app.get("/api/health")
def api_health():
    return jsonify(stream.health_snapshot())


@app.get("/api/next")
def api_next():
    raw_start = request.args.get("start")
    if raw_start is not None:
        try:
            stream.set_cursor(int(raw_start))
        except ValueError:
            stream.set_cursor(0)
    return jsonify(stream.next_sample())


@app.post("/api/reset")
def api_reset():
    from flask import request

    raw_index = request.args.get("index", default="0")
    try:
        index = int(raw_index)
    except ValueError:
        index = 0

    cursor = stream.set_cursor(index)
    return jsonify({"ok": True, "cursor": cursor, "total": stream.total_samples})


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
    finally:
        stream.close()