"""
train_cnn.py
Full training pipeline for HybridGNSSCNN.

Mirrors the exact train/test splits used in your RF baseline so results
are directly comparable.

Usage
-----
    python train_cnn.py --h5 /path/to/gnss_dataset.h5 --epochs 40
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.serialization

# local import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.attack_labeler import compute_labels_batch
from spectrum_cnn import HybridGNSSCNN, FocalLoss


# -----------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------

class GNSSDataset(Dataset):
    """
    Returns one sample per second:
        spectrum : (2, 256)   -- MON-SPAN rf blocks, zero-padded to 256 bins
        scalar   : (30,)      -- PVT/clock/DOP/POSECEF
        label    : int        -- 0=Clean, 1=Spoofing, 2=Jamming
    """

    N_SPECTRUM_BINS = 256     # pad / truncate all spectrum arrays to this
    N_RF_BLOCKS     = 2       # u-blox F9 has 2 RF paths

    def __init__(self, h5_path: str, mask: np.ndarray,
                 scalar_scaler: StandardScaler | None = None,
                 fit_scaler: bool = False):
        """
        Parameters
        ----------
        h5_path      : path to your HDF5 file
        mask         : boolean index array selecting which rows to use
        scalar_scaler: a fitted StandardScaler (pass None on first call with
                       fit_scaler=True, then reuse the returned scaler)
        fit_scaler   : if True, fit a new scaler on this subset
        """
        with h5py.File(h5_path, "r") as hf:
            # --- scalar features (same 30 as RF baseline) ---
            scalar = np.hstack([
                hf["nav_pvt"][:],       # (N, 15)
                hf["nav_clock"][:],     # (N, 4)
                hf["nav_dop"][:],       # (N, 7)
                hf["nav_posecef"][:],   # (N, 4)
            ])[mask].astype(np.float32)

            # --- spectrum (MON-SPAN) ---
            # Expected shape in HDF5: (N, n_rf_blocks, bins)
            # If your H5 stores it differently, adjust the slice below.
            if "spectrum_01" in hf:
                s1 = hf["spectrum_01"][:][mask].astype(np.float32) # (N,32)
                s2 = hf["spectrum_02"][:][mask].astype(np.float32)
                # stack into (N, 2, 256) to match the expected input
                raw_spec = np.stack([s1,s2], axis=1)
                spec = self._normalise_spectrum(raw_spec)
            else:
                # Fallback: zero spectrum (model degrades to scalar-only)
                spec = np.zeros(
                    (mask.sum(), self.N_RF_BLOCKS, self.N_SPECTRUM_BINS),
                    dtype=np.float32,
                )

            days  = hf["day"][:][mask]
            hours = hf["hour"][:][mask]

        # --- labels ---
        N = mask.sum()
        seconds = np.zeros(N, dtype=np.int32)
        unique_days = np.unique(days)
        for d in unique_days:
          for h in range(24):
              idx = np.where((days==d) & (hours==h))[0]
              if len(idx):
                seconds[idx] = np.arange(len(idx))
        self.labels = compute_labels_batch(days, hours, seconds).astype(np.int64)

        # --- normalise scalars ---
        if fit_scaler:
            scalar_scaler = StandardScaler()
            scalar_scaler.fit(scalar)
        if scalar_scaler is not None:
            scalar = scalar_scaler.transform(scalar)
        self.scaler = scalar_scaler

        self.scalar  = torch.from_numpy(scalar)
        self.spectrum = torch.from_numpy(spec)

    # ------------------------------------------------------------------
    def _normalise_spectrum(self, raw: np.ndarray) -> np.ndarray:
        """
        Ensure shape (N, N_RF_BLOCKS, N_SPECTRUM_BINS).
        Also normalises each spectrum to zero-mean / unit-std so the
        CNN sees relative shape rather than absolute dB values
        (absolute levels vary with AGC / PGA gain).
        """
        N = raw.shape[0]

        # Handle variable number of RF blocks
        n_blocks = min(raw.shape[1], self.N_RF_BLOCKS) if raw.ndim == 3 else 1
        n_bins   = raw.shape[-1]

        out = np.zeros((N, self.N_RF_BLOCKS, self.N_SPECTRUM_BINS),
                       dtype=np.float32)

        bins_to_copy = min(n_bins, self.N_SPECTRUM_BINS)
        if raw.ndim == 3:
            out[:, :n_blocks, :bins_to_copy] = raw[:, :n_blocks, :bins_to_copy]
        else:
            out[:, 0, :bins_to_copy] = raw[:, :bins_to_copy]

        # per-sample, per-block z-score
        mean = out.mean(axis=-1, keepdims=True)
        std  = out.std(axis=-1, keepdims=True) + 1e-6
        out  = (out - mean) / std

        return out

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrum[idx], self.scalar[idx], self.labels[idx]


# -----------------------------------------------------------------------
# 2. Training helpers
# -----------------------------------------------------------------------

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Returns a sampler that oversamples minority classes so every mini-batch
    has a roughly balanced distribution.  This is on top of Focal Loss —
    using both is belt-and-braces for a 19:1 imbalance ratio.
    """
    classes, counts = np.unique(labels, return_counts=True)
    class_weights   = 1.0 / counts
    sample_weights  = class_weights[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(labels),
        replacement = True,
    )


def compute_class_weights(labels: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for Focal Loss alpha parameter."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = torch.zeros(3)
    for c, cnt in zip(classes, counts):
        weights[c] = len(labels) / (len(classes) * cnt)
    return weights.to(device)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for spec, scalar, y in loader:
            spec, scalar, y = spec.to(device), scalar.to(device), y.to(device)
            logits = model(spec, scalar)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probs   = torch.cat(all_probs).numpy()
    return preds, targets, probs


# -----------------------------------------------------------------------
# 3. Main training loop
# -----------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- build masks (identical logic to your RF baseline) ---
    print("Loading HDF5 metadata...")
    with h5py.File(args.h5, "r") as hf:
      days  = hf["day"][:]
      hours = hf["hour"][:]
      N     = len(days)

    seconds = np.zeros(N, dtype=np.int32)
    for d in np.unique(days):
      for h in range(24):
        idx = np.where((days==d) & (hours==h))[0]
        if len(idx):
          seconds[idx] = np.arange(len(idx))

    labels = compute_labels_batch(days, hours, seconds)
    is_spoofing = (labels == 1)
    is_jamming  = (labels == 2)

    CLEAN_TRAIN_DAYS = [b"12", b"13", b"14", b"15", b"16", b"17", b"18",
                        b"19", b"20", b"21", b"22", b"23", b"24", b"25",
                        b"26", b"27", b"28"]
    CLEAN_TEST_DAYS  = [b"29", b"30"]

    clean_train = np.isin(days, CLEAN_TRAIN_DAYS)
    clean_test  = np.isin(days, CLEAN_TEST_DAYS)
    spoof_train = is_spoofing & np.isin(hours, [12, 13, 14])
    spoof_test  = is_spoofing & np.isin(hours, [15, 16])
    jam_hour17_idx = np.where(is_jamming & (hours == 17))[0]

    # 70% train, 30% test — adjust ratio as needed
    split = int(0.7 * len(jam_hour17_idx))
    jam_train_idx = jam_hour17_idx[:split]
    jam_test_idx  = jam_hour17_idx[split:]

    # rebuild masks
    jam_train = np.zeros(N, dtype=bool)
    jam_test  = np.zeros(N, dtype=bool)
    jam_train[jam_train_idx] = True
    jam_test[jam_test_idx]   = True

    # also include the small hour 16 jamming in test
    jam_test = jam_test | (is_jamming & (hours == 16))

    train_mask = clean_train | spoof_train | jam_train
    test_mask  = clean_test  | spoof_test  | jam_test

    # Split off 10% of train for validation
    train_idx = np.where(train_mask)[0]
    train_labels_subset = labels[train_idx]

    # stratified split — keeps class proportions in both train and val
    train_idx_final, val_idx = train_test_split(
      train_idx,
      test_size=0.1,
      random_state=42,
      stratify=train_labels_subset   # <-- this is the key fix
    )
    val_mask = np.zeros(N, dtype=bool)
    val_mask[val_idx] = True
    train_mask_final = np.zeros(N, dtype=bool)
    train_mask_final[train_idx_final] = True

    print("Building datasets...")
    train_ds = GNSSDataset(args.h5, train_mask_final, fit_scaler=True)
    val_ds   = GNSSDataset(args.h5, val_mask,   scalar_scaler=train_ds.scaler)
    test_ds  = GNSSDataset(args.h5, test_mask,  scalar_scaler=train_ds.scaler)

    sampler  = make_weighted_sampler(train_ds.labels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=1, pin_memory=True)
    test_dl = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=1, pin_memory=True)

    # --- model, loss, optimiser ---
    model = HybridGNSSCNN(in_channels=GNSSDataset.N_RF_BLOCKS,scalar_dim=30).to(device)

    alpha     = compute_class_weights(train_ds.labels, device)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimiser  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing: warms up for 5 epochs then decays smoothly
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
      optimiser,
      max_lr = args.lr,
      epochs = args.epochs,
      steps_per_epoch = len(train_dl),
      pct_start = 0.1,
    )

    # --- training loop ---
    best_val_f1   = 0.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
      # this is to make sure the model doesn't have to spend time restabilizing after first epoch. There is a big drop in the loss
      model.train()
      running_loss = 0.0

      for spec, scalar, y in train_dl:
        spec, scalar, y = spec.to(device), scalar.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(spec, scalar)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()
        running_loss += loss.item() * y.size(0)

      train_loss = running_loss / len(train_ds)

      # validation
      preds, targets, probs = evaluate(model, val_dl, device)
      report = classification_report(
        targets, preds, labels=[0, 1, 2],
        target_names=["Clean", "Spoofing", "Jamming"],
        output_dict=True, zero_division=0,
      )
      macro_f1 = report["macro avg"]["f1-score"]

      print(f"Epoch {epoch:3d}/{args.epochs}  "
          f"loss={train_loss:.4f}  val_macro_f1={macro_f1:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}")
      

      # --- early stopping + best model checkpoint ---
      if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        torch.save({
          "epoch":        epoch,
          "model_state":  model.state_dict(),
          "scaler":       train_ds.scaler,
          "val_macro_f1": best_val_f1,
          "args":         vars(args),
        }, args.save_path)
        patience_left = args.patience
        print(f"  ✓ Saved best model (val_macro_f1={best_val_f1:.4f})")
      else:
        patience_left -= 1
        if patience_left == 0:
          print(f"Early stopping at epoch {epoch}")
          break

    # --- final evaluation on test set ---
    print("\n=== Loading best checkpoint for test evaluation ===")
    #torch.serialization.add_safe_globals([StandardScaler])
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    preds, targets, probs = evaluate(model, test_dl, device)

    print("\n=== Test Classification Report ===")
    print(classification_report(targets, preds, labels=[0, 1, 2], target_names=["Clean", "Spoofing", "Jamming"], zero_division=0))

    print("=== Confusion Matrix (Clean=0, Spoofing=1, Jamming=2) ===")
    print(confusion_matrix(targets, preds, labels=[0, 1, 2]))

    n_classes_present = len(np.unique(targets))
    if n_classes_present > 2:
      auc = roc_auc_score(targets, probs, multi_class="ovr")
      print(f"\nAUC-ROC (OvR): {auc:.4f}")

    return model


# -----------------------------------------------------------------------
# 4. CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train HybridGNSSCNN")
  parser.add_argument("--h5",         required=True,
                      help="Path to gnss_dataset.h5")
  parser.add_argument("--epochs",     type=int,   default=40)
  parser.add_argument("--batch-size", type=int,   default=256,
                      dest="batch_size")
  parser.add_argument("--lr",         type=float, default=3e-4)
  parser.add_argument("--patience",   type=int,   default=8,
                      help="Early-stopping patience (epochs)")
  parser.add_argument("--save-path",  default="cnn_best.pt",
                      dest="save_path")
  args = parser.parse_args()
  train(args)