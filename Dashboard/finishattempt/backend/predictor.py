import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from schemas import GNSSInput, PredictionOutput
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectrum_cnn import HybridGNSSCNN

LABEL_NAMES = {0: "Clean", 1: "Spoofing", 2: "Jamming"}
CHECKPOINT  = Path(__file__).parent.parent / "models" / "cnn_best_gpu_run1.pt"

class GNSSPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Predictor using device: {self.device}")

        # load checkpoint once at startup
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)

        self.scaler: StandardScaler = ckpt["scaler"]

        self.model = HybridGNSSCNN(in_channels=2, scalar_dim=30).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print("Model loaded and ready")

    def _normalise_spectrum(self, spec: np.ndarray) -> np.ndarray:
        # z-score per block — same as GNSSDataset._normalise_spectrum
        mean = spec.mean(axis=-1, keepdims=True)
        std  = spec.std(axis=-1,  keepdims=True) + 1e-6
        return (spec - mean) / std

    def predict(self, data: GNSSInput) -> PredictionOutput:
        # build scalar vector (1, 30)
        scalar = np.array(
            data.nav_pvt + data.nav_clock + data.nav_dop + data.nav_posecef,
            dtype=np.float32
        ).reshape(1, -1)
        scalar = self.scaler.transform(scalar)

        # build spectrum (1, 2, 256)
        s1 = np.array(data.spectrum_01, dtype=np.float32)
        s2 = np.array(data.spectrum_02, dtype=np.float32)
        spec = np.stack([s1, s2], axis=0)[np.newaxis]  # (1, 2, 256)
        spec = self._normalise_spectrum(spec)

        # run inference
        with torch.no_grad():
            spec_t   = torch.from_numpy(spec).to(self.device)
            scalar_t = torch.from_numpy(scalar).to(self.device)
            logits   = self.model(spec_t, scalar_t)
            probs    = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            label    = int(probs.argmax())

        return PredictionOutput(
            label=label,
            label_name=LABEL_NAMES[label],
            confidence=float(probs[label]),
            probabilities=probs.tolist(),
        )