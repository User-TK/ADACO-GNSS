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
                    scalars: np.ndarray):
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