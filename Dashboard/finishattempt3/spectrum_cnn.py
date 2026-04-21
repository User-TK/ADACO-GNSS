"""
spectrum_cnn.py
Hybrid CNN for GNSS spoofing/jamming detection.

Input streams
-------------
spectrum : (B, n_rf_blocks, 256)  -- MON-SPAN, 2 RF paths, 256 bins each
scalar   : (B, 30)                -- nav_pvt(15) + nav_clock(4) + nav_dop(7) + nav_posecef(4)

Output
------
logits   : (B, 3)                 -- 0=Clean, 1=Spoofing, 2=Jamming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building Blocks
# ------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """
    1-D residual block: Conv -> BN -> ReLU -> Conv -> BN + skip.
    Keeps spatial dimension the same (no downsampling here; pooling
    is done explicitly between blocks).
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)          # skip connection


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Tells the model *which frequency bands* matter most.
    This is particularly useful because spoofing is centred on GPS L1 (1575 MHz)
    and jamming is broadband — they leave very different spectral fingerprints.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global avg over time/freq axis
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        w = self.gate(x).unsqueeze(-1)   # (B, C, 1)
        return x * w


# ------------------------------------------------------------------
# Spectrum Branch  (processes MON-SPAN data)
# ------------------------------------------------------------------

class SpectrumBranch(nn.Module):
    """
    Input : (B, n_rf_blocks, 256)   e.g. (B, 2, 256)
    Output: (B, 256)  flattened feature vector
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()

        # Stem: expand channels quickly
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32ch, L/2
        self.stage1 = nn.Sequential(
            ResBlock1d(32, kernel_size=5),
            ChannelAttention(32),
            nn.MaxPool1d(2),           # 256 -> 128
        )

        # Stage 2: 64ch, L/4
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResBlock1d(64, kernel_size=3),
            ChannelAttention(64),
            nn.MaxPool1d(2),           # 128 -> 64
        )

        # Stage 3: 128ch, L/8
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResBlock1d(128, kernel_size=3),
            ChannelAttention(128),
            nn.AdaptiveAvgPool1d(2),   # -> (B, 128, 2)
        )

        self.out_features = 256        # 128 * 2

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.stem(spectrum)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x.flatten(1)            # (B, 256)


# ------------------------------------------------------------------
# Scalar Branch  (processes PVT / clock / DOP / POSECEF features)
# ------------------------------------------------------------------

class ScalarBranch(nn.Module):
    """
    Input : (B, 30)   -- the same 30 features used by the RF baseline
    Output: (B, 64)

    Using a small MLP with residual connections so gradients don't
    vanish, and LayerNorm instead of BN (works better for small dims).
    """

    def __init__(self, in_features: int = 30):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_features = 64

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        x = self.proj(scalar)
        return self.relu(self.block(x) + x)      # residual



# Fusion Head
# ----------

class FusionHead(nn.Module):
    """
    Fuses spectrum features (256) + scalar features (64) = 320 dims.
    Outputs logits for 3 classes.
    """

    def __init__(self, spectrum_dim: int = 256, scalar_dim: int = 64,
                 n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        fused = spectrum_dim + scalar_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, spectrum_feat: torch.Tensor,
                scalar_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spectrum_feat, scalar_feat], dim=1)
        return self.head(x)


# Full Model
# ----

class HybridGNSSCNN(nn.Module):
    """
    Spectrum-CNN + Scalar-MLP hybrid for GNSS threat detection.

    Parameters
    ----------
    in_channels  : number of RF blocks in MON-SPAN (default 2)
    scalar_dim   : number of scalar input features  (default 30)
    n_classes    : 3  (Clean / Spoofing / Jamming)
    dropout      : dropout rate in fusion head
    """

    def __init__(self, in_channels: int = 2, scalar_dim: int = 30,
                 n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.spectrum_branch = SpectrumBranch(in_channels)
        self.scalar_branch   = ScalarBranch(scalar_dim)
        self.fusion          = FusionHead(
            spectrum_dim = self.spectrum_branch.out_features,
            scalar_dim   = self.scalar_branch.out_features,
            n_classes    = n_classes,
            dropout      = dropout,
        )

    def forward(self, spectrum: torch.Tensor,
                scalar: torch.Tensor) -> torch.Tensor:
        """
        spectrum : (B, in_channels, 256)
        scalar   : (B, scalar_dim)
        returns    (B, n_classes) logits
        """
        s_feat = self.spectrum_branch(spectrum)
        x_feat = self.scalar_branch(scalar)
        return self.fusion(s_feat, x_feat)

    # ------------------------------------------------------------------
    # Convenience: spectrum-only forward (when scalar unavailable)
    # ------------------------------------------------------------------
    def forward_spectrum_only(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Falls back to zero scalar features — useful at inference time
        if only spectrum data is available."""
        B = spectrum.size(0)
        zeros = torch.zeros(B, self.scalar_branch.proj[0].in_features,
                            device=spectrum.device)
        return self.forward(spectrum, zeros)


# ------------------------------------------------------------------
# Focal Loss  (critical for the 19:1 class imbalance in this dataset)
# ------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy negatives so the model focuses on
    hard-to-classify attack windows.  gamma=2 is standard.

    Also accepts per-class weights (alpha) to handle the severe
    clean >> spoofing >> jamming imbalance in the dataset.
    """

    def __init__(self, alpha: torch.Tensor | None = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)   # (n_classes,) or None
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             reduction="none")          # (B,)
        pt = torch.exp(-ce)                             # probability of true class
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    model = HybridGNSSCNN(in_channels=2, scalar_dim=30)
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}")

    # dummy forward pass
    B = 8
    spec   = torch.randn(B, 2, 256)
    scalar = torch.randn(B, 30)
    out    = model(spec, scalar)
    print(f"Output shape: {out.shape}")   # expect (8, 3)