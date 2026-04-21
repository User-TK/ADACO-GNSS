"""Multi-modal GNSS spoofing/jamming detection pipeline.

This module is structured as a practical reference implementation for:

* loading GNSS HDF5 telemetry
* feature engineering and normalization
* leakage-safe day-based train/validation/test splits
* sliding temporal window construction
* an advanced multi-branch PyTorch model
* training and evaluation utilities

The code is dependency-light at import time. Heavy dependencies such as
``h5py`` and ``torch`` are imported lazily so the file can still be inspected
or linted in environments that do not have the full ML stack installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    WeightedRandomSampler = None
    Dataset = object  # type: ignore


HDF5_KEYS = (
    "nav_pvt",
    "nav_clock",
    "nav_dop",
    "nav_posecef",
    "span_meta",
    "spectrum_01",
    "spectrum_02",
    "nav_sat",
    "rxm_rawx",
    "day",
    "hour",
    "label",
)


def _require_numpy() -> Any:
    if np is None:
        raise ImportError("numpy is required for the GNSS modeling pipeline")
    return np


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py is required to load the GNSS HDF5 dataset")
    return h5py


def _require_torch() -> Any:
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("torch is required to train the neural network model")
    return torch


def log_message(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def _cache_metadata(
    data_path: str | os.PathLike[str],
    window_size: int,
    stride: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    path = Path(data_path)
    stat = path.stat()
    return {
        "data_path": str(path.resolve()),
        "data_mtime": stat.st_mtime,
        "data_size": stat.st_size,
        "window_size": int(window_size),
        "stride": int(stride),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
    }


def _cache_matches(cache_meta: Mapping[str, Any], expected_meta: Mapping[str, Any]) -> bool:
    for key, expected_value in expected_meta.items():
        cached_value = cache_meta.get(key)
        if isinstance(expected_value, float):
            if cached_value is None or abs(float(cached_value) - expected_value) > 1e-9:
                return False
        else:
            if cached_value != expected_value:
                return False
    return True


def save_preprocessed_cache(cache_path: str | os.PathLike[str], built: Mapping[str, Any], cache_meta: Mapping[str, Any], *, verbose: bool = True) -> None:
    np_mod = _require_numpy()
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_message(f"[cache] saving preprocessed cache to {path}", verbose)

    preprocessor: PreprocessorState = built["preprocessor"]
    processed = built["processed"]
    payload: Dict[str, Any] = {
        "cache_meta": np_mod.asarray(json.dumps(dict(cache_meta))),
        "scalar_mean": np_mod.asarray(preprocessor.scalar.mean),
        "scalar_std": np_mod.asarray(preprocessor.scalar.std),
        "spectrum_01_mean": np_mod.asarray(preprocessor.spectrum_01.mean),
        "spectrum_01_std": np_mod.asarray(preprocessor.spectrum_01.std),
        "spectrum_02_mean": np_mod.asarray(preprocessor.spectrum_02.mean),
        "spectrum_02_std": np_mod.asarray(preprocessor.spectrum_02.std),
        "nav_sat_mean": np_mod.asarray(preprocessor.nav_sat.mean),
        "nav_sat_std": np_mod.asarray(preprocessor.nav_sat.std),
        "rxm_rawx_mean": np_mod.asarray(preprocessor.rxm_rawx.mean),
        "rxm_rawx_std": np_mod.asarray(preprocessor.rxm_rawx.std),
        "processed_scalar": np_mod.asarray(processed["scalar"]),
        "processed_spectrum_01": np_mod.asarray(processed["spectrum_01"]),
        "processed_spectrum_02": np_mod.asarray(processed["spectrum_02"]),
        "processed_nav_sat": np_mod.asarray(processed["nav_sat"]),
        "processed_rxm_rawx": np_mod.asarray(processed["rxm_rawx"]),
        "processed_day": np_mod.asarray(processed["day"]),
        "processed_hour": np_mod.asarray(processed["hour"]),
        "processed_label": np_mod.asarray(processed["label"]),
        "train_starts": np_mod.asarray(built["train_starts"]),
        "val_starts": np_mod.asarray(built["val_starts"]),
        "test_starts": np_mod.asarray(built["test_starts"]),
    }
    np_mod.savez_compressed(path, **payload)
    log_message(f"[cache] wrote {path.stat().st_size / (1024 * 1024):.1f} MiB", verbose)


def load_preprocessed_cache(cache_path: str | os.PathLike[str], expected_meta: Mapping[str, Any], *, verbose: bool = True) -> Optional[Dict[str, Any]]:
    np_mod = _require_numpy()
    path = Path(cache_path)
    if not path.exists():
        return None

    log_message(f"[cache] checking cache file: {path}", verbose)
    with np_mod.load(path, allow_pickle=False) as data:
        cache_meta = json.loads(str(data["cache_meta"]))
        if not _cache_matches(cache_meta, expected_meta):
            log_message("[cache] cache mismatch; rebuilding preprocessing", verbose)
            return None

        preprocessor = PreprocessorState(
            scalar=StandardizationStats(mean=data["scalar_mean"], std=data["scalar_std"]),
            spectrum_01=StandardizationStats(mean=data["spectrum_01_mean"], std=data["spectrum_01_std"]),
            spectrum_02=StandardizationStats(mean=data["spectrum_02_mean"], std=data["spectrum_02_std"]),
            nav_sat=StandardizationStats(mean=data["nav_sat_mean"], std=data["nav_sat_std"]),
            rxm_rawx=StandardizationStats(mean=data["rxm_rawx_mean"], std=data["rxm_rawx_std"]),
        )
        processed = {
            "scalar": data["processed_scalar"],
            "spectrum_01": data["processed_spectrum_01"],
            "spectrum_02": data["processed_spectrum_02"],
            "nav_sat": data["processed_nav_sat"],
            "rxm_rawx": data["processed_rxm_rawx"],
            "day": data["processed_day"],
            "hour": data["processed_hour"],
            "label": data["processed_label"],
        }
        log_message("[cache] loaded cached preprocessing output", verbose)
        return {
            "preprocessor": preprocessor,
            "processed": processed,
            "train_starts": data["train_starts"],
            "val_starts": data["val_starts"],
            "test_starts": data["test_starts"],
        }


def _as_float_array(array: Any) -> Any:
    np_mod = _require_numpy()
    return np_mod.asarray(array, dtype=np_mod.float32)


def _safe_nan_to_num(array: Any) -> Any:
    np_mod = _require_numpy()
    return np_mod.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _standardize(array: Any, mean: Any, std: Any) -> Any:
    np_mod = _require_numpy()
    return (array - mean) / np_mod.maximum(std, 1e-6)


def _concat_features(parts: Sequence[Any]) -> Any:
    np_mod = _require_numpy()
    return np_mod.concatenate([_as_float_array(part) for part in parts], axis=-1)


def _flatten_with_feature_axis(array: Any) -> Any:
    np_mod = _require_numpy()
    array = _as_float_array(array)
    return array.reshape(-1, array.shape[-1])


def _spectral_entropy(power: Any) -> Any:
    np_mod = _require_numpy()
    power = _safe_nan_to_num(_as_float_array(power))
    shifted = power - power.min(axis=-1, keepdims=True)
    shifted = shifted + 1e-8
    prob = shifted / shifted.sum(axis=-1, keepdims=True)
    entropy = -(prob * np_mod.log(prob + 1e-12)).sum(axis=-1)
    max_entropy = np_mod.log(prob.shape[-1])
    return entropy / np_mod.maximum(max_entropy, 1e-6)


def _spectral_feature_matrix(spectrum: Any) -> Any:
    np_mod = _require_numpy()
    spec = _safe_nan_to_num(_as_float_array(spectrum))
    mean = spec.mean(axis=-1)
    std = spec.std(axis=-1)
    min_value = spec.min(axis=-1)
    max_value = spec.max(axis=-1)
    peak_index = spec.argmax(axis=-1).astype(np_mod.float32)
    peak_value = np_mod.take_along_axis(spec, peak_index.astype(np_mod.int64)[..., None], axis=-1)[..., 0]
    peak_ratio = peak_value / np_mod.maximum(spec.sum(axis=-1), 1e-6)
    entropy = _spectral_entropy(spec)
    dynamic_range = max_value - min_value
    return _concat_features(
        [
            mean[..., None],
            std[..., None],
            min_value[..., None],
            max_value[..., None],
            peak_index[..., None],
            peak_value[..., None],
            peak_ratio[..., None],
            entropy[..., None],
            dynamic_range[..., None],
        ]
    )


def _aggregate_axis_stats(array: Any, axis: int) -> Any:
    np_mod = _require_numpy()
    arr = _safe_nan_to_num(_as_float_array(array))
    mean = arr.mean(axis=axis)
    std = arr.std(axis=axis)
    min_value = arr.min(axis=axis)
    max_value = arr.max(axis=axis)
    return _concat_features([mean, std, min_value, max_value])


def _encode_time_features(day: Any, hour: Any) -> Any:
    np_mod = _require_numpy()
    day_arr = _as_float_array(day)
    hour_arr = _as_float_array(hour)
    hour_angle = 2.0 * math.pi * (hour_arr % 24.0) / 24.0
    return _concat_features(
        [
            day_arr[..., None],
            (hour_arr / 24.0)[..., None],
            np_mod.sin(hour_angle)[..., None],
            np_mod.cos(hour_angle)[..., None],
        ]
    )


def load_hdf5_arrays(
    hdf5_path: str | os.PathLike[str],
    keys: Sequence[str] = HDF5_KEYS,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load the required datasets into memory as NumPy arrays."""

    np_mod = _require_numpy()
    h5py_mod = _require_h5py()

    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(path)

    log_message(f"[load] opening dataset: {path}", verbose)

    with h5py_mod.File(path, "r") as handle:
        missing = [key for key in keys if key not in handle]
        if missing:
            raise KeyError(f"Missing HDF5 keys: {missing}")
        arrays = {}
        for key in keys:
            value = np_mod.asarray(handle[key][...])
            arrays[key] = value
            log_message(f"[load] {key}: shape={value.shape}, dtype={value.dtype}", verbose)

    log_message(f"[load] loaded {len(arrays)} datasets", verbose)
    return arrays


def build_scalar_feature_matrix(arrays: Mapping[str, Any]) -> Any:
    """Create the per-second feature vector used by the scalar branch."""

    np_mod = _require_numpy()

    nav_parts = [
        arrays["nav_pvt"],
        arrays["nav_clock"],
        arrays["nav_dop"],
        arrays["nav_posecef"],
        arrays["span_meta"],
    ]
    nav_scalar = _concat_features(nav_parts)
    sat_stats = _aggregate_axis_stats(arrays["nav_sat"], axis=1)
    rawx_stats = _aggregate_axis_stats(arrays["rxm_rawx"], axis=1)
    spectrum_01_stats = _spectral_feature_matrix(arrays["spectrum_01"])
    spectrum_02_stats = _spectral_feature_matrix(arrays["spectrum_02"])
    time_features = _encode_time_features(arrays["day"], arrays["hour"])

    scalar_features = _concat_features(
        [
            nav_scalar,
            sat_stats,
            rawx_stats,
            spectrum_01_stats,
            spectrum_02_stats,
            time_features,
        ]
    )
    return np_mod.asarray(_safe_nan_to_num(scalar_features), dtype=np_mod.float32)


@dataclass
class StandardizationStats:
    mean: Any
    std: Any


@dataclass
class PreprocessorState:
    scalar: StandardizationStats
    spectrum_01: StandardizationStats
    spectrum_02: StandardizationStats
    nav_sat: StandardizationStats
    rxm_rawx: StandardizationStats


def _fit_standardization_stats(array: Any) -> StandardizationStats:
    np_mod = _require_numpy()
    arr = _safe_nan_to_num(_as_float_array(array))
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np_mod.maximum(std, 1e-6)
    return StandardizationStats(mean=mean.astype(np_mod.float32), std=std.astype(np_mod.float32))


def fit_preprocessor(
    arrays: Mapping[str, Any],
    sample_indices: Optional[Sequence[int]] = None,
    *,
    verbose: bool = True,
) -> PreprocessorState:
    """Fit z-score statistics on a training subset.

    If ``sample_indices`` is provided, the scalar features are estimated from
    those timesteps only. Otherwise the entire arrays are used.
    """

    np_mod = _require_numpy()
    log_message("[prep] fitting preprocessing statistics", verbose)
    if sample_indices is not None:
        indices = np_mod.asarray(sample_indices, dtype=np_mod.int64)
        log_message(f"[prep] fitting on {indices.size} training timesteps", verbose)
        scalar_source = {key: np_mod.asarray(value)[indices] for key, value in arrays.items() if key in HDF5_KEYS}
    else:
        log_message("[prep] fitting on all available timesteps", verbose)
        scalar_source = arrays

    scalar_stats = _fit_standardization_stats(build_scalar_feature_matrix(scalar_source))
    spectrum_01_stats = _fit_standardization_stats(arrays["spectrum_01"])
    spectrum_02_stats = _fit_standardization_stats(arrays["spectrum_02"])
    nav_sat_stats = _fit_standardization_stats(_flatten_with_feature_axis(arrays["nav_sat"]))
    rxm_rawx_stats = _fit_standardization_stats(_flatten_with_feature_axis(arrays["rxm_rawx"]))
    return PreprocessorState(
        scalar=scalar_stats,
        spectrum_01=spectrum_01_stats,
        spectrum_02=spectrum_02_stats,
        nav_sat=nav_sat_stats,
        rxm_rawx=rxm_rawx_stats,
    )


def apply_preprocessor(arrays: Mapping[str, Any], state: PreprocessorState, *, verbose: bool = True) -> Dict[str, Any]:
    np_mod = _require_numpy()
    log_message("[prep] applying preprocessing transforms", verbose)
    scalar = _standardize(build_scalar_feature_matrix(arrays), state.scalar.mean, state.scalar.std)
    spectrum_01 = _standardize(_safe_nan_to_num(_as_float_array(arrays["spectrum_01"])), state.spectrum_01.mean, state.spectrum_01.std)
    spectrum_02 = _standardize(_safe_nan_to_num(_as_float_array(arrays["spectrum_02"])), state.spectrum_02.mean, state.spectrum_02.std)

    nav_sat = _safe_nan_to_num(_as_float_array(arrays["nav_sat"]))
    nav_sat = _standardize(nav_sat.reshape(-1, nav_sat.shape[-1]), state.nav_sat.mean, state.nav_sat.std).reshape(nav_sat.shape)

    rxm_rawx = _safe_nan_to_num(_as_float_array(arrays["rxm_rawx"]))
    rxm_rawx = _standardize(rxm_rawx.reshape(-1, rxm_rawx.shape[-1]), state.rxm_rawx.mean, state.rxm_rawx.std).reshape(rxm_rawx.shape)

    return {
        "scalar": np_mod.asarray(scalar, dtype=np_mod.float32),
        "spectrum_01": np_mod.asarray(spectrum_01, dtype=np_mod.float32),
        "spectrum_02": np_mod.asarray(spectrum_02, dtype=np_mod.float32),
        "nav_sat": np_mod.asarray(nav_sat, dtype=np_mod.float32),
        "rxm_rawx": np_mod.asarray(rxm_rawx, dtype=np_mod.float32),
        "day": np_mod.asarray(arrays["day"]),
        "hour": np_mod.asarray(arrays["hour"]),
        "label": np_mod.asarray(arrays["label"], dtype=np_mod.int64),
    }


def _segment_boundaries(day_values: Any) -> Any:
    np_mod = _require_numpy()
    day_arr = np_mod.asarray(day_values)
    if day_arr.ndim != 1:
        raise ValueError("day values must be one-dimensional")
    change_points = np_mod.flatnonzero(np_mod.concatenate(([True], day_arr[1:] != day_arr[:-1], [True])))
    return change_points


def build_window_starts(day_values: Any, window_size: int, stride: int) -> Any:
    """Return valid window start indices that do not cross day boundaries."""

    np_mod = _require_numpy()
    day_arr = np_mod.asarray(day_values)
    if day_arr.ndim != 1:
        raise ValueError("day values must be one-dimensional")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    starts: List[int] = []
    boundaries = _segment_boundaries(day_arr)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segment_length = int(right - left)
        if segment_length < window_size:
            continue
        segment_starts = range(left, right - window_size + 1, stride)
        starts.extend(segment_starts)
    return np_mod.asarray(starts, dtype=np_mod.int64)


def window_labels(labels: Any, starts: Any, window_size: int) -> Any:
    np_mod = _require_numpy()
    label_arr = np_mod.asarray(labels)
    start_arr = np_mod.asarray(starts, dtype=np_mod.int64)
    values = [int(label_arr[start : start + window_size].max()) for start in start_arr]
    return np_mod.asarray(values, dtype=np_mod.int64)


def split_days(day_values: Any, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 13) -> Tuple[Any, Any, Any]:
    """Leakage-safe split that assigns whole days to train/val/test."""

    np_mod = _require_numpy()
    unique_days = np_mod.unique(np_mod.asarray(day_values))
    unique_days = unique_days.copy()
    rng = np_mod.random.default_rng(seed)
    rng.shuffle(unique_days)

    n_days = int(unique_days.shape[0])
    if n_days == 0:
        raise ValueError("no day values were provided")

    test_count = 0
    val_count = 0
    if n_days >= 3:
        test_count = max(1, int(round(n_days * test_ratio)))
        test_count = min(test_count, n_days - 2)
        val_count = max(1, int(round(n_days * val_ratio)))
        val_count = min(val_count, n_days - test_count - 1)

    test_days = unique_days[:test_count]
    val_days = unique_days[test_count : test_count + val_count]
    train_days = unique_days[test_count + val_count :]
    if train_days.size == 0:
        train_days = unique_days[: max(1, n_days - 1)]
    return train_days, val_days, test_days


def filter_window_starts_by_days(day_values: Any, starts: Any, allowed_days: Sequence[Any]) -> Any:
    np_mod = _require_numpy()
    day_arr = np_mod.asarray(day_values)
    start_arr = np_mod.asarray(starts, dtype=np_mod.int64)
    allowed = np_mod.asarray(allowed_days)
    if allowed.size == 0:
        return np_mod.asarray([], dtype=np_mod.int64)
    mask = np_mod.isin(day_arr[start_arr], allowed)
    return start_arr[mask]


class WindowedGNSSDataset(Dataset):
    """Sliding window dataset that slices the raw arrays on demand."""

    def __init__(
        self,
        arrays: Mapping[str, Any],
        window_starts: Sequence[int],
        window_size: int = 20,
        preprocess_state: Optional[PreprocessorState] = None,
    ) -> None:
        self.arrays = arrays
        self.window_starts = _require_numpy().asarray(window_starts, dtype=_require_numpy().int64)
        self.window_size = int(window_size)
        self.preprocess_state = preprocess_state

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def _slice(self, name: str, start: int) -> Any:
        np_mod = _require_numpy()
        value = np_mod.asarray(self.arrays[name][start : start + self.window_size])
        return value

    def __getitem__(self, index: int) -> Dict[str, Any]:
        np_mod = _require_numpy()
        start = int(self.window_starts[index])

        scalar = self._slice("scalar", start)
        spectrum_01 = self._slice("spectrum_01", start)
        spectrum_02 = self._slice("spectrum_02", start)
        nav_sat = self._slice("nav_sat", start)
        rxm_rawx = self._slice("rxm_rawx", start)
        label = int(np_mod.asarray(self.arrays["label"])[start : start + self.window_size].max())

        sample = {
            "scalar": scalar.astype(np_mod.float32),
            "spectrum_01": spectrum_01.astype(np_mod.float32),
            "spectrum_02": spectrum_02.astype(np_mod.float32),
            "nav_sat": nav_sat.astype(np_mod.float32),
            "rxm_rawx": rxm_rawx.astype(np_mod.float32),
            "label": np_mod.asarray(label, dtype=np_mod.float32),
        }

        if torch is not None:
            sample = {key: torch.as_tensor(value) for key, value in sample.items()}
        return sample


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: List[Any] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            previous = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Any) -> Any:
        return self.network(inputs)


class SpectrumCNNEncoder(nn.Module):
    def __init__(self, input_length: int = 256, embedding_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, inputs: Any) -> Any:
        batch_shape = inputs.shape[:-1]
        x = inputs.reshape(-1, 1, inputs.shape[-1])
        x = self.conv(x)
        x = self.projection(x)
        return x.reshape(*batch_shape, -1)


class SetAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, inputs: Any) -> Any:
        batch_shape = inputs.shape[:-2]
        item_count = inputs.shape[-2]
        feature_dim = inputs.shape[-1]
        x = inputs.reshape(-1, item_count, feature_dim)
        x = self.item_encoder(x)
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=-2)
        return pooled.reshape(*batch_shape, -1)


class StepFusionEncoder(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        spectrum_embedding_dim: int = 128,
        set_embedding_dim: int = 128,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        self.scalar_encoder = MLPBlock(scalar_dim, [256, 256, fused_dim], dropout=0.15)
        self.spectrum_encoder = SpectrumCNNEncoder(input_length=256, embedding_dim=spectrum_embedding_dim)
        self.satellite_encoder = SetAttentionEncoder(input_dim=10, embedding_dim=set_embedding_dim)
        self.rawx_encoder = SetAttentionEncoder(input_dim=11, embedding_dim=set_embedding_dim)
        total_dim = fused_dim + spectrum_embedding_dim * 2 + set_embedding_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.15),
        )

    def forward(self, scalar: Any, spectrum_01: Any, spectrum_02: Any, nav_sat: Any, rxm_rawx: Any) -> Any:
        scalar_embedding = self.scalar_encoder(scalar)
        spectrum_01_embedding = self.spectrum_encoder(spectrum_01)
        spectrum_02_embedding = self.spectrum_encoder(spectrum_02)
        nav_sat_embedding = self.satellite_encoder(nav_sat)
        rxm_rawx_embedding = self.rawx_encoder(rxm_rawx)
        fused = torch.cat(
            [scalar_embedding, spectrum_01_embedding, spectrum_02_embedding, nav_sat_embedding, rxm_rawx_embedding],
            dim=-1,
        )
        return self.fusion(fused)


class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        step_embedding_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff_dim: int = 512,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.step_encoder = StepFusionEncoder(
            scalar_dim=scalar_dim,
            spectrum_embedding_dim=128,
            set_embedding_dim=128,
            fused_dim=step_embedding_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=step_embedding_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.head = nn.Sequential(
            nn.Linear(step_embedding_dim, step_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(step_embedding_dim // 2, 1),
        )

    def forward(self, scalar: Any, spectrum_01: Any, spectrum_02: Any, nav_sat: Any, rxm_rawx: Any) -> Any:
        # Inputs are shaped [batch, time, ...].
        batch_size, time_steps = scalar.shape[0], scalar.shape[1]
        scalar_flat = scalar.reshape(batch_size * time_steps, scalar.shape[-1])
        spec1_flat = spectrum_01.reshape(batch_size * time_steps, spectrum_01.shape[-1])
        spec2_flat = spectrum_02.reshape(batch_size * time_steps, spectrum_02.shape[-1])
        nav_sat_flat = nav_sat.reshape(batch_size * time_steps, nav_sat.shape[-2], nav_sat.shape[-1])
        rxm_rawx_flat = rxm_rawx.reshape(batch_size * time_steps, rxm_rawx.shape[-2], rxm_rawx.shape[-1])

        step_embeddings = self.step_encoder(scalar_flat, spec1_flat, spec2_flat, nav_sat_flat, rxm_rawx_flat)
        step_embeddings = step_embeddings.reshape(batch_size, time_steps, -1)
        temporal_embeddings = self.temporal_encoder(step_embeddings)
        pooled = temporal_embeddings.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def binary_confusion_matrix(y_true: Any, y_prob: Any, threshold: float = 0.5) -> Dict[str, int]:
    np_mod = _require_numpy()
    y_true_arr = np_mod.asarray(y_true).astype(np_mod.int64)
    y_pred_arr = (np_mod.asarray(y_prob) >= threshold).astype(np_mod.int64)
    tp = int(np_mod.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np_mod.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np_mod.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np_mod.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_classification_metrics(y_true: Any, y_prob: Any, threshold: float = 0.5) -> Dict[str, float]:
    np_mod = _require_numpy()
    y_true_arr = np_mod.asarray(y_true).astype(np_mod.int64)
    y_prob_arr = np_mod.asarray(y_prob).astype(np_mod.float64)
    cm = binary_confusion_matrix(y_true_arr, y_prob_arr, threshold=threshold)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    roc_auc = binary_roc_auc(y_true_arr, y_prob_arr)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def find_best_threshold(y_true: Any, y_prob: Any, step: float = 0.01) -> Tuple[float, Dict[str, float]]:
    """Return the threshold in [0.05, 0.95] that maximizes F1."""

    np_mod = _require_numpy()
    y_true_arr = np_mod.asarray(y_true)
    y_prob_arr = np_mod.asarray(y_prob)
    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        default_metrics = binary_classification_metrics(y_true_arr, y_prob_arr, threshold=0.5)
        return 0.5, default_metrics

    best_threshold = 0.5
    best_metrics = binary_classification_metrics(y_true_arr, y_prob_arr, threshold=best_threshold)
    best_f1 = float(best_metrics["f1"])

    threshold = 0.05
    while threshold <= 0.95 + 1e-9:
        metrics = binary_classification_metrics(y_true_arr, y_prob_arr, threshold=threshold)
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_threshold = float(threshold)
            best_metrics = metrics
        threshold += step

    return best_threshold, best_metrics


def binary_roc_auc(y_true: Any, y_score: Any) -> float:
    np_mod = _require_numpy()
    y_true_arr = np_mod.asarray(y_true).astype(np_mod.int64)
    y_score_arr = np_mod.asarray(y_score).astype(np_mod.float64)
    pos = int((y_true_arr == 1).sum())
    neg = int((y_true_arr == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np_mod.argsort(y_score_arr)
    ranks = np_mod.empty_like(order, dtype=np_mod.float64)
    ranks[order] = np_mod.arange(1, y_score_arr.shape[0] + 1, dtype=np_mod.float64)
    rank_sum = float(ranks[y_true_arr == 1].sum())
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def compute_pos_weight(labels: Any) -> float:
    np_mod = _require_numpy()
    label_arr = np_mod.asarray(labels).astype(np_mod.int64)
    positives = int((label_arr == 1).sum())
    negatives = int((label_arr == 0).sum())
    return float(negatives / max(positives, 1))


def _collate_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required to collate training batches")
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def make_dataloader(
    dataset: Any,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    *,
    verbose: bool = True,
    name: str = "dataset",
    sampler: Optional[Any] = None,
) -> Any:
    _require_torch()
    log_message(
        f"[data] creating dataloader for {name}: size={len(dataset)}, batch_size={batch_size}, shuffle={shuffle}, workers={num_workers}",
        verbose,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_batch,
    )


def compute_window_labels_from_dataset(dataset: Any) -> Any:
    starts = _require_numpy().asarray(dataset.window_starts)
    return window_labels(dataset.arrays["label"], starts, dataset.window_size)


def make_balanced_sampler(dataset: Any, *, verbose: bool = True, name: str = "train") -> Any:
    """Oversample minority-class windows through weighted random sampling."""

    torch_mod = _require_torch()
    if WeightedRandomSampler is None:
        raise ImportError("torch WeightedRandomSampler is unavailable")

    labels = _require_numpy().asarray(compute_window_labels_from_dataset(dataset), dtype=_require_numpy().int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    pos_sample_weight = negatives / max(positives, 1)
    sample_weights = _require_numpy().where(labels == 1, pos_sample_weight, 1.0).astype(_require_numpy().float32)
    log_message(
        f"[data] {name} sampler labels: positives={positives}, negatives={negatives}, pos_sample_weight={pos_sample_weight:.4f}",
        verbose,
    )
    return WeightedRandomSampler(
        weights=torch_mod.as_tensor(sample_weights, dtype=torch_mod.float32),
        num_samples=int(sample_weights.shape[0]),
        replacement=True,
    )


class FocalWithLogitsLoss(nn.Module):
    """Binary focal loss operating directly on logits."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: Any, targets: Any) -> Any:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        loss = focal * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    *,
    verbose: bool = True,
    epoch: int = 0,
    log_every: int = 25,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    _require_torch()
    model.train()
    total_loss = 0.0
    total_items = 0
    batch_count = 0

    log_message(f"[train] epoch {epoch}: starting training loop", verbose)

    for batch in dataloader:
        batch_count += 1
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["scalar"].to(device),
            batch["spectrum_01"].to(device),
            batch["spectrum_02"].to(device),
            batch["nav_sat"].to(device),
            batch["rxm_rawx"].to(device),
        )
        labels = batch["label"].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if verbose and (batch_count == 1 or batch_count % log_every == 0):
            avg_loss = total_loss / max(total_items, 1)
            print(f"[train] epoch {epoch} batch {batch_count}: loss={float(loss.item()):.5f}, running_avg_loss={avg_loss:.5f}", flush=True)

        if max_batches is not None and batch_count >= max_batches:
            log_message(f"[train] epoch {epoch}: stopping early after {batch_count} batches (max_batches={max_batches})", verbose)
            break

    final_loss = total_loss / max(total_items, 1)
    log_message(f"[train] epoch {epoch}: finished {batch_count} batches, loss={final_loss:.5f}", verbose)
    return {"loss": final_loss}


@torch.no_grad() if torch is not None else (lambda fn: fn)  # type: ignore[misc]
def evaluate_model(
    model: Any,
    dataloader: Any,
    device: Any,
    threshold: float = 0.5,
    *,
    verbose: bool = True,
    name: str = "eval",
    max_batches: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, Any]:
    _require_torch()
    model.eval()
    probabilities: List[Any] = []
    targets: List[Any] = []
    batch_count = 0

    log_message(f"[{name}] starting evaluation", verbose)

    for batch in dataloader:
        batch_count += 1
        logits = model(
            batch["scalar"].to(device),
            batch["spectrum_01"].to(device),
            batch["spectrum_02"].to(device),
            batch["nav_sat"].to(device),
            batch["rxm_rawx"].to(device),
        )
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(batch["label"].detach().cpu())

        if verbose and batch_count % 25 == 0:
            print(f"[{name}] processed {batch_count} batches", flush=True)

        if max_batches is not None and batch_count >= max_batches:
            log_message(f"[{name}] stopping early after {batch_count} batches (max_batches={max_batches})", verbose)
            break

    y_prob = torch.cat(probabilities).numpy() if probabilities else _require_numpy().array([])
    y_true = torch.cat(targets).numpy() if targets else _require_numpy().array([])
    metrics = binary_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["threshold"] = float(threshold)
    if return_raw:
        metrics["_y_true"] = y_true
        metrics["_y_prob"] = y_prob
    log_message(f"[{name}] done: f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}", verbose)
    return metrics


def train_model(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: Optional[str] = None,
    verbose: bool = True,
    log_every: int = 25,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    tune_threshold: bool = False,
) -> Dict[str, Any]:
    """Train the model with early stopping on validation F1."""

    torch_mod = _require_torch()
    device_obj = torch_mod.device(device or ("cuda" if torch_mod.cuda.is_available() else "cpu"))
    model = model.to(device_obj)

    log_message(f"[train] using device: {device_obj}", verbose)
    log_message(f"[train] epochs={epochs}, lr={learning_rate}, weight_decay={weight_decay}, patience={patience}", verbose)
    if max_train_batches is not None:
        log_message(f"[train] max_train_batches per epoch={max_train_batches}", verbose)
    if max_val_batches is not None:
        log_message(f"[train] max_val_batches per epoch={max_val_batches}", verbose)

    train_dataset = getattr(train_loader, "dataset", None)
    if train_dataset is not None and hasattr(train_dataset, "window_starts") and hasattr(train_dataset, "arrays"):
        pos_weight = compute_pos_weight(compute_window_labels_from_dataset(train_dataset))
    else:
        train_labels = []
        for batch in train_loader:
            train_labels.append(batch["label"].detach().cpu() if torch_mod.is_tensor(batch["label"]) else torch_mod.as_tensor(batch["label"]))
        pos_weight = compute_pos_weight(torch_mod.cat(train_labels).numpy()) if train_labels else 1.0
    log_message(f"[train] positive-class weight: {pos_weight:.4f}", verbose)

    if loss_type == "focal":
        criterion = FocalWithLogitsLoss(gamma=focal_gamma, alpha=focal_alpha)
        log_message(f"[train] loss=focal (gamma={focal_gamma}, alpha={focal_alpha})", verbose)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch_mod.tensor(pos_weight, device=device_obj))
        log_message("[train] loss=bce_with_pos_weight", verbose)
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch_mod.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = None
    best_val_f1 = -float("inf")
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        log_message(f"[train] epoch {epoch}/{epochs} starting", verbose)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device_obj,
            verbose=verbose,
            epoch=epoch,
            log_every=log_every,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device_obj,
            verbose=verbose,
            name=f"val epoch {epoch}",
            max_batches=max_val_batches,
            return_raw=tune_threshold,
        )

        if tune_threshold:
            y_true = val_metrics.pop("_y_true", _require_numpy().array([]))
            y_prob = val_metrics.pop("_y_prob", _require_numpy().array([]))
            tuned_threshold, tuned_metrics = find_best_threshold(y_true, y_prob)
            tuned_metrics["threshold"] = float(tuned_threshold)
            val_metrics = tuned_metrics
            log_message(
                f"[train] tuned validation threshold={tuned_threshold:.2f} with f1={val_metrics['f1']:.4f}",
                verbose,
            )

        scheduler.step(val_metrics["f1"])

        record = {"epoch": float(epoch), **{f"train_{k}": float(v) for k, v in train_metrics.items()}, **{f"val_{k}": float(v) for k, v in val_metrics.items()}}
        history.append(record)

        log_message(
            f"[train] epoch {epoch} summary: train_loss={train_metrics['loss']:.5f}, val_f1={val_metrics['f1']:.4f}, val_recall={val_metrics['recall']:.4f}, val_precision={val_metrics['precision']:.4f}",
            verbose,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_threshold = float(val_metrics.get("threshold", 0.5))
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
            log_message(f"[train] new best model at epoch {epoch} with val_f1={best_val_f1:.4f}", verbose)
        else:
            no_improve += 1
            log_message(f"[train] no improvement for {no_improve} epoch(s)", verbose)
            if no_improve >= patience:
                log_message(f"[train] early stopping triggered at epoch {epoch}", verbose)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "device": str(device_obj),
    }


def build_datasets(
    arrays: Mapping[str, Any],
    window_size: int = 20,
    stride: int = 1,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Construct day-safe train/validation/test datasets."""

    np_mod = _require_numpy()
    log_message(f"[split] building window starts: window_size={window_size}, stride={stride}", verbose)
    window_starts = build_window_starts(arrays["day"], window_size=window_size, stride=stride)
    train_days, val_days, test_days = split_days(arrays["day"], val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    log_message(f"[split] window count: {window_starts.size}", verbose)
    log_message(f"[split] train days={train_days.tolist() if hasattr(train_days, 'tolist') else train_days}", verbose)
    log_message(f"[split] val days={val_days.tolist() if hasattr(val_days, 'tolist') else val_days}", verbose)
    log_message(f"[split] test days={test_days.tolist() if hasattr(test_days, 'tolist') else test_days}", verbose)

    train_starts = filter_window_starts_by_days(arrays["day"], window_starts, train_days)
    val_starts = filter_window_starts_by_days(arrays["day"], window_starts, val_days)
    test_starts = filter_window_starts_by_days(arrays["day"], window_starts, test_days)

    log_message(f"[split] train windows={train_starts.size}, val windows={val_starts.size}, test windows={test_starts.size}", verbose)

    train_sample_indices = np_mod.flatnonzero(np_mod.isin(arrays["day"], train_days))
    preprocessor = fit_preprocessor(arrays, sample_indices=train_sample_indices, verbose=verbose)
    processed = apply_preprocessor(arrays, preprocessor, verbose=verbose)

    log_message(
        f"[split] feature shapes: scalar={processed['scalar'].shape}, spectrum_01={processed['spectrum_01'].shape}, spectrum_02={processed['spectrum_02'].shape}, nav_sat={processed['nav_sat'].shape}, rxm_rawx={processed['rxm_rawx'].shape}",
        verbose,
    )

    return {
        "preprocessor": preprocessor,
        "processed": processed,
        "train_dataset": WindowedGNSSDataset(processed, train_starts, window_size=window_size),
        "val_dataset": WindowedGNSSDataset(processed, val_starts, window_size=window_size),
        "test_dataset": WindowedGNSSDataset(processed, test_starts, window_size=window_size),
        "train_starts": train_starts,
        "val_starts": val_starts,
        "test_starts": test_starts,
    }


def attach_cached_datasets(built: MutableMapping[str, Any], window_size: int) -> MutableMapping[str, Any]:
    """Ensure cached bundles include dataset objects expected by main()."""

    if "train_dataset" in built and "val_dataset" in built and "test_dataset" in built:
        return built

    processed = built["processed"]
    built["train_dataset"] = WindowedGNSSDataset(processed, built["train_starts"], window_size=window_size)
    built["val_dataset"] = WindowedGNSSDataset(processed, built["val_starts"], window_size=window_size)
    built["test_dataset"] = WindowedGNSSDataset(processed, built["test_starts"], window_size=window_size)
    return built


def plot_predictions_vs_time(
    y_true: Any,
    y_prob: Any,
    window_days: Any,
    *,
    title: str = "GNSS attack detection over time",
) -> None:
    """Visualize model probabilities against the corresponding window day index."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting") from exc

    np_mod = _require_numpy()
    days = np_mod.asarray(window_days)
    truth = np_mod.asarray(y_true)
    prob = np_mod.asarray(y_prob)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(prob, label="predicted attack probability", linewidth=1.5)
    ax1.plot(truth.astype(float), label="ground truth", alpha=0.6)
    ax1.set_xlabel("window index")
    ax1.set_ylabel("label / probability")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(days, color="black", alpha=0.12, label="day")
    ax2.set_ylabel("day")
    plt.tight_layout()
    plt.show()


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a multi-modal GNSS spoofing/jamming detector")
    parser.add_argument("--data", type=str, default="datasets/gnss_dataset.h5", help="Path to the GNSS HDF5 file")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window length in seconds")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride in seconds")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--log-every", type=int, default=25, help="Print a batch update every N training batches")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap training batches per epoch for faster runs")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches per epoch for faster runs")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Cap test batches during final evaluation")
    parser.add_argument("--cache-file", type=str, default=".cache/gnss_preprocessed.npz", help="Path to a cached preprocessing bundle")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading and writing the preprocessing cache")
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        choices=["pos-weight", "sampler", "focal", "sampler+focal"],
        default="pos-weight",
        help="Class imbalance strategy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma when focal is enabled")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Optional focal alpha in [0,1]")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune classification threshold on validation data")
    args = parser.parse_args(argv)

    verbose = not args.quiet

    set_seed(args.seed)
    log_message("[main] starting GNSS training run", verbose)
    log_message(f"[main] data={args.data}", verbose)
    log_message(f"[main] window_size={args.window_size}, stride={args.stride}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}", verbose)
    if args.max_train_batches is not None or args.max_val_batches is not None or args.max_test_batches is not None:
        log_message(
            f"[main] batch caps: train={args.max_train_batches}, val={args.max_val_batches}, test={args.max_test_batches}",
            verbose,
        )
    if not args.no_cache:
        log_message(f"[main] cache file={args.cache_file}", verbose)
    log_message(f"[main] imbalance strategy={args.imbalance_strategy}, tune_threshold={args.tune_threshold}", verbose)

    cache_meta = _cache_metadata(args.data, args.window_size, args.stride, 0.15, 0.15, args.seed)
    built = None
    if not args.no_cache:
        built = load_preprocessed_cache(args.cache_file, cache_meta, verbose=verbose)

    if built is None:
        arrays = load_hdf5_arrays(args.data, verbose=verbose)
        built = build_datasets(arrays, window_size=args.window_size, stride=args.stride, seed=args.seed, verbose=verbose)
        if not args.no_cache:
            save_preprocessed_cache(args.cache_file, built, cache_meta, verbose=verbose)
    else:
        log_message("[main] using cached preprocessing output", verbose)
        built = attach_cached_datasets(built, window_size=args.window_size)

    train_sampler = None
    if "sampler" in args.imbalance_strategy:
        train_sampler = make_balanced_sampler(built["train_dataset"], verbose=verbose, name="train")

    train_loader = make_dataloader(
        built["train_dataset"],
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        verbose=verbose,
        name="train",
        sampler=train_sampler,
    )
    val_loader = make_dataloader(built["val_dataset"], batch_size=args.batch_size, shuffle=False, verbose=verbose, name="validation")
    test_loader = make_dataloader(built["test_dataset"], batch_size=args.batch_size, shuffle=False, verbose=verbose, name="test")

    model = TemporalTransformerClassifier(scalar_dim=built["processed"]["scalar"].shape[-1])
    log_message(f"[main] model scalar input dim={built['processed']['scalar'].shape[-1]}", verbose)
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        verbose=verbose,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        loss_type="focal" if "focal" in args.imbalance_strategy else "bce",
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        tune_threshold=args.tune_threshold,
    )
    test_metrics = evaluate_model(
        result["model"],
        test_loader,
        torch.device(result["device"]),
        threshold=float(result.get("best_threshold", 0.5)),
        verbose=verbose,
        name="test",
        max_batches=args.max_test_batches,
    ) if torch is not None else {}

    log_message("[main] training complete", verbose)
    print(f"Best epoch: {result['best_epoch']}", flush=True)
    print(f"Best validation F1: {result['best_val_f1']:.4f}", flush=True)
    print(f"Best threshold: {result.get('best_threshold', 0.5):.2f}", flush=True)
    if test_metrics:
        print("Test metrics:", flush=True)
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())