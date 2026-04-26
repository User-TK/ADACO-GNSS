"""
inference.py
this is a inference wrapper for our HybridGNSSCNN

Usage
-----
from inference import GNSSDetector

detector = GNSSDetector("cnn_best.pt")
result   = detector.predict_single(spectrum_array, scalar_array)
print(result)   # {"label": "Spoofing", "confidence": 0.97, "probabilities": {...}}
"""

import numpy as np
import torch
from pathlib import Path
from spectrum_cnn import HybridGNSSCNN
from train_cnn import GNSSDataset

CLASS_NAMES = {0: "Clean", 1: "Spoofing", 2: "Jamming"}
FEATURE_NAMES = (
    ["pvt_" + f for f in ["fixType", "gnssFixOk", "numSV", "lat", "lon", "height",
                           "hMSL", "hAcc", "vAcc", "gSpeed", "pDOP", "carrSoln",
                           "difSoln", "invalidLlh", "tAcc"]] +
    ["clock_" + f for f in ["clkB", "clkD", "tAcc", "fAcc"]] +
    ["dop_" + f for f in ["gDOP", "pDOP", "tDOP", "vDOP", "hDOP", "nDOP", "eDOP"]] +
    ["pos_" + f for f in ["ecefX", "ecefY", "ecefZ", "pAcc"]]
)

FEATURE_LABELS = {
    "pvt_fixType": "Fix Type (NAV-PVT)",
    "pvt_gnssFixOk": "Fix Valid Flag (NAV-PVT)",
    "pvt_numSV": "Satellites Used (NAV-PVT)",
    "pvt_lat": "Latitude (NAV-PVT)",
    "pvt_lon": "Longitude (NAV-PVT)",
    "pvt_height": "Ellipsoid Height (NAV-PVT)",
    "pvt_hMSL": "Height Above Mean Sea Level (NAV-PVT)",
    "pvt_hAcc": "Horizontal Accuracy (NAV-PVT)",
    "pvt_vAcc": "Vertical Accuracy (NAV-PVT)",
    "pvt_gSpeed": "Ground Speed (NAV-PVT)",
    "pvt_pDOP": "Position DOP (NAV-PVT)",
    "pvt_carrSoln": "Carrier Solution Quality (NAV-PVT)",
    "pvt_difSoln": "Differential Solution Flag (NAV-PVT)",
    "pvt_invalidLlh": "Invalid Position Flag (NAV-PVT)",
    "pvt_tAcc": "Time Accuracy (NAV-PVT)",
    "clock_clkB": "Clock Bias (NAV-CLOCK)",
    "clock_clkD": "Clock Drift (NAV-CLOCK)",
    "clock_tAcc": "Clock Time Accuracy (NAV-CLOCK)",
    "clock_fAcc": "Clock Frequency Accuracy (NAV-CLOCK)",
    "dop_gDOP": "Geometric DOP (NAV-DOP)",
    "dop_pDOP": "Position DOP (NAV-DOP)",
    "dop_tDOP": "Time DOP (NAV-DOP)",
    "dop_vDOP": "Vertical DOP (NAV-DOP)",
    "dop_hDOP": "Horizontal DOP (NAV-DOP)",
    "dop_nDOP": "Northing DOP (NAV-DOP)",
    "dop_eDOP": "Easting DOP (NAV-DOP)",
    "pos_ecefX": "ECEF X Position (NAV-POSECEF)",
    "pos_ecefY": "ECEF Y Position (NAV-POSECEF)",
    "pos_ecefZ": "ECEF Z Position (NAV-POSECEF)",
    "pos_pAcc": "Position Accuracy (NAV-POSECEF)",
    "Spectrum RF-1": "RF Power Band 1 (MON-SPAN)",
    "Spectrum RF-2": "RF Power Band 2 (MON-SPAN)",
}

FEATURE_OPERATOR_REASON = {
    "pvt_fixType": "Navigation fix mode. Sudden downgrades can indicate signal problems.",
    "pvt_gnssFixOk": "Whether the receiver says the current fix is valid.",
    "pvt_numSV": "How many satellites are used. Sudden drops can indicate interference.",
    "pvt_lat": "Estimated latitude from the navigation solution.",
    "pvt_lon": "Estimated longitude from the navigation solution.",
    "pvt_height": "Estimated receiver altitude.",
    "pvt_hMSL": "Height above mean sea level from NAV-PVT.",
    "pvt_hAcc": "Estimated horizontal uncertainty. Larger values mean lower trust.",
    "pvt_vAcc": "Estimated vertical uncertainty. Larger values mean lower trust.",
    "pvt_gSpeed": "Estimated ground speed. Unexpected spikes can be suspicious.",
    "pvt_pDOP": "Position geometry quality from NAV-PVT. Lower is better.",
    "pvt_carrSoln": "Carrier-phase solution quality indicator from NAV-PVT.",
    "pvt_difSoln": "Whether differential corrections are being used.",
    "pvt_invalidLlh": "Flag indicating invalid latitude/longitude/height output.",
    "pvt_tAcc": "Estimated timing uncertainty from NAV-PVT.",
    "clock_clkB": "Receiver clock time offset. Sudden jumps can indicate spoofing.",
    "clock_clkD": "How fast clock error changes. Instability can indicate spoofing.",
    "clock_tAcc": "Estimated uncertainty of clock time offset.",
    "clock_fAcc": "Estimated uncertainty of clock frequency offset.",
    "dop_gDOP": "Overall satellite geometry quality. Lower is better.",
    "dop_pDOP": "Position geometry quality. Sharp changes may indicate abnormal geometry.",
    "dop_tDOP": "Time-solution geometry quality from NAV-DOP.",
    "dop_vDOP": "Vertical geometry quality from NAV-DOP.",
    "dop_hDOP": "Horizontal geometry quality from NAV-DOP.",
    "dop_nDOP": "North-axis geometry quality from NAV-DOP.",
    "dop_eDOP": "East-axis geometry quality from NAV-DOP.",
    "pos_ecefX": "Receiver X position in Earth-centered coordinates.",
    "pos_ecefY": "Receiver Y position in Earth-centered coordinates.",
    "pos_ecefZ": "Receiver Z position in Earth-centered coordinates.",
    "pos_pAcc": "Estimated position uncertainty from NAV-POSECEF.",
    "Spectrum RF-1": "RF power near GPS L1. Broad power rise can indicate jamming.",
    "Spectrum RF-2": "RF power near GPS L2. Broad power rise can indicate jamming.",
}


def _feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def _feature_reason(feature_name: str) -> str:
    return FEATURE_OPERATOR_REASON.get(feature_name, "Important GNSS signal or navigation input for this prediction.")

# Confidence thresholds — tune these on your val set.
# Anything below ALERT_THRESHOLD is flagged as "uncertain".

ALERT_THRESHOLD = 0.65


class GNSSDetector:
    """
    Wraps a trained HybridGNSSCNN checkpoint for single-sample or
    batch inference.

    Parameters
    ----------
    checkpoint_path : path to the .pt file saved by train_cnn.py
    device          : "cpu", "cuda", or "auto"  (default: auto)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Re-create model with same hyper-params used during training
        args = ckpt.get("args", {})
        self.model = HybridGNSSCNN(
            in_channels = GNSSDataset.N_RF_BLOCKS,
            scalar_dim  = 30,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.scaler = ckpt["scaler"]   # fitted StandardScaler
        print(f"Loaded checkpoint (epoch {ckpt['epoch']}, "
              f"val_macro_f1={ckpt['val_macro_f1']:.4f})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, spectrum: np.ndarray,
                       scalar: np.ndarray) -> dict:
        """
        Parameters
        ----------
        spectrum : (2, 256)  or (n_rf_blocks, n_bins)  -- raw dB values
        scalar   : (30,)     -- raw (un-scaled) feature vector

        Returns
        -------
        dict with keys:
            label         : str  ("Clean" / "Spoofing" / "Jamming")
            class_id      : int
            confidence    : float  (probability of predicted class)
            probabilities : dict   {class_name: float}
            uncertain     : bool   (True if confidence < ALERT_THRESHOLD)
        """
        spec_t, scal_t = self._preprocess(
            spectrum[None], scalar[None]    # add batch dim
        )
        with torch.no_grad():
            logits = self.model(spec_t, scal_t)
            probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        class_id   = int(probs.argmax())
        confidence = float(probs[class_id])

        return {
            "label":         CLASS_NAMES[class_id],
            "class_id":      class_id,
            "confidence":    confidence,
            "probabilities": {CLASS_NAMES[i]: float(p)
                              for i, p in enumerate(probs)},
            "uncertain":     confidence < ALERT_THRESHOLD,
        }

    def predict_single_with_importance(self, spectrum: np.ndarray,
                                       scalar: np.ndarray) -> dict:
        """Return the normal prediction plus a gradient-based attribution."""
        spec_t, scal_t = self._preprocess(spectrum[None], scalar[None], requires_grad=True)

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(spec_t, scal_t)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        class_id = int(probs.argmax())
        confidence = float(probs[class_id])

        logits[0, class_id].backward()

        scalar_grad = (scal_t.grad.detach().abs() * scal_t.detach().abs()).squeeze(0).cpu().numpy()
        spectrum_grad = (spec_t.grad.detach().abs() * spec_t.detach().abs()).squeeze(0).cpu().numpy()

        scalar_entries = [
            {
                "name": _feature_label(FEATURE_NAMES[idx]),
                "value": float(score),
                "reason": _feature_reason(FEATURE_NAMES[idx]),
            }
            for idx, score in enumerate(scalar_grad)
        ]
        if scalar_entries:
            scalar_peak = max(entry["value"] for entry in scalar_entries) or 1.0
            for entry in scalar_entries:
                entry["value"] = float(entry["value"] / scalar_peak)

        spectrum_block_names = ["Spectrum RF-1", "Spectrum RF-2"]
        spectrum_scores = spectrum_grad.mean(axis=-1)
        spectrum_peak = float(np.max(spectrum_scores)) if spectrum_scores.size else 1.0
        if spectrum_peak == 0.0:
            spectrum_peak = 1.0
        spectrum_entries = [
            {
                "name": _feature_label(spectrum_block_names[idx]),
                "value": float(score / spectrum_peak),
                "reason": _feature_reason(spectrum_block_names[idx]),
            }
            for idx, score in enumerate(spectrum_scores)
        ]

        combined_entries = sorted(
            scalar_entries + spectrum_entries,
            key=lambda item: item["value"],
            reverse=True,
        )[:6]

        return {
            "label":         CLASS_NAMES[class_id],
            "class_id":      class_id,
            "confidence":    confidence,
            "probabilities": {CLASS_NAMES[i]: float(p)
                              for i, p in enumerate(probs)},
            "uncertain":     confidence < ALERT_THRESHOLD,
            "feature_importance": {
                "top_features": combined_entries,
                "scalar_features": scalar_entries,
                "spectrum_features": spectrum_entries,
            },
        }

    def predict_batch(self, spectra: np.ndarray,
                      scalars: np.ndarray) -> list[dict]:
        """
        Parameters
        ----------
        spectra : (N, 2, 256)
        scalars : (N, 30)

        Returns list of result dicts (same format as predict_single).
        """
        spec_t, scal_t = self._preprocess(spectra, scalars)
        with torch.no_grad():
            logits = self.model(spec_t, scal_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for p in probs:
            class_id   = int(p.argmax())
            confidence = float(p[class_id])
            results.append({
                "label":         CLASS_NAMES[class_id],
                "class_id":      class_id,
                "confidence":    confidence,
                "probabilities": {CLASS_NAMES[i]: float(v)
                                  for i, v in enumerate(p)},
                "uncertain":     confidence < ALERT_THRESHOLD,
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, spectra: np.ndarray,
                    scalars: np.ndarray,
                    requires_grad: bool = False):
        """Normalise spectrum + apply fitted scaler to scalars."""
        # spectrum: z-score per sample per RF block
        spec = spectra.astype(np.float32)
        mean = spec.mean(axis=-1, keepdims=True)
        std  = spec.std(axis=-1, keepdims=True) + 1e-6
        spec = (spec - mean) / std

        # pad / truncate spectrum bins
        N = spec.shape[0]
        out_spec = np.zeros(
            (N, GNSSDataset.N_RF_BLOCKS, GNSSDataset.N_SPECTRUM_BINS),
            dtype=np.float32,
        )
        n_blocks = min(spec.shape[1], GNSSDataset.N_RF_BLOCKS)
        n_bins   = min(spec.shape[2], GNSSDataset.N_SPECTRUM_BINS)
        out_spec[:, :n_blocks, :n_bins] = spec[:, :n_blocks, :n_bins]

        # scalar: StandardScaler
        scal = self.scaler.transform(scalars.astype(np.float32))

        spec_t = torch.from_numpy(out_spec).to(self.device)
        scal_t = torch.from_numpy(scal).to(self.device)
        if requires_grad:
            spec_t.requires_grad_(True)
            scal_t.requires_grad_(True)
        return spec_t, scal_t


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_checkpoint.pt>")
        sys.exit(1)

    detector = GNSSDetector(sys.argv[1])

    # Fake a single sample
    fake_spectrum = np.random.randn(2, 256).astype(np.float32)
    fake_scalar   = np.random.randn(30).astype(np.float32)
    result = detector.predict_single(fake_spectrum, fake_scalar)
    print("\nSingle sample result:")
    for k, v in result.items():
        print(f"  {k}: {v}")