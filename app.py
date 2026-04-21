from __future__ import annotations

import sys
import threading
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, jsonify, redirect, request, send_from_directory


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATASET_PATH = ROOT / "datasets" / "gnss_dataset.h5"
DASHBOARD_DIR = ROOT / "Dashboard" / "dashboard_compact"
CHECKPOINT_PATH = MODEL_DIR / "cnn_best_gpu_run1.pt"

sys.path.insert(0, str(MODEL_DIR))

from inference import GNSSDetector


class LiveGNSSStream:
    def __init__(self, h5_path: Path, detector: GNSSDetector):
        self._h5 = h5py.File(h5_path, "r")
        self._detector = detector
        self._lock = threading.Lock()
        self._cursor = 0
        self.total_samples = int(self._h5["day"].shape[0])

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
        prediction = self._detector.predict_single(spectrum, scalar)

        day = self._decode_day(self._h5["day"][index])
        hour = int(self._h5["hour"][index])
        ground_truth = int(self._h5["label"][index])

        probabilities = prediction["probabilities"]
        combined_attack = float(
            min(1.0, probabilities["Spoofing"] + probabilities["Jamming"])
        )

        return {
            "index": index,
            "total": self.total_samples,
            "day": day,
            "hour": hour,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "derived": {
                "clean": float(probabilities["Clean"]),
                "spoofing": float(probabilities["Spoofing"]),
                "jamming": float(probabilities["Jamming"]),
                "combined_attack": combined_attack,
            },
        }

    def set_cursor(self, index: int) -> int:
        with self._lock:
            self._cursor = int(index) % self.total_samples
            return self._cursor


app = Flask(__name__)
detector = GNSSDetector(str(CHECKPOINT_PATH))
stream = LiveGNSSStream(DATASET_PATH, detector)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def root():
    return redirect("/dashboard_compact/dashboard_compact.html")


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
            "classes": ["Clean", "Spoofing", "Jamming"],
        }
    )


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