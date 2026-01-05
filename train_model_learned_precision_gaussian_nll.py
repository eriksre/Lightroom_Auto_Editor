#!/usr/bin/env python3
"""
Train a neural network to predict Lightroom editing parameters from image features.
Optimized for Apple Silicon (M4 Pro) with MPS backend.

This variant uses a correlation-aware multivariate Gaussian negative
log-likelihood with a *learned global precision matrix* (inverse covariance).
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict

from models import LightroomMLP


# Circular (angular) columns that need sin/cos encoding
# These are hue values in degrees (0-360) where 0° and 360° are the same
CIRCULAR_COLUMNS = [
    "ColorGradeGlobalHue",
    "ColorGradeShadowHue",
    "ColorGradeMidtoneHue",
    "ColorGradeHighlightHue",
]

# Color grading saturation columns where 0 is a valid/meaningful value (not "untouched")
COLOR_GRADING_SAT_COLUMNS = [
    "ColorGradeGlobalSat",
    "ColorGradeShadowSat",
    "ColorGradeMidtoneSat",
    "ColorGradeHighlightSat",
]

# Target columns to predict
TARGET_COLUMNS = [
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", 
    "Whites2012", "Blacks2012", "Texture", "Clarity2012", "Dehaze",
    "Vibrance", "Saturation", "IncrementalTemperature", "IncrementalTint",
    "HueAdjustmentRed", "HueAdjustmentOrange", "HueAdjustmentYellow",
    "HueAdjustmentGreen", "HueAdjustmentAqua", "HueAdjustmentBlue",
    "HueAdjustmentPurple", "HueAdjustmentMagenta",
    "SaturationAdjustmentRed", "SaturationAdjustmentOrange", 
    "SaturationAdjustmentYellow", "SaturationAdjustmentGreen",
    "SaturationAdjustmentAqua", "SaturationAdjustmentBlue",
    "SaturationAdjustmentPurple", "SaturationAdjustmentMagenta",
    "LuminanceAdjustmentRed", "LuminanceAdjustmentOrange",
    "LuminanceAdjustmentYellow", "LuminanceAdjustmentGreen",
    "LuminanceAdjustmentAqua", "LuminanceAdjustmentBlue",
    "LuminanceAdjustmentPurple", "LuminanceAdjustmentMagenta",
    "ColorGradeBlending", "ColorGradeGlobalHue", "ColorGradeGlobalSat",
    "ColorGradeGlobalLum", "ColorGradeShadowHue", "ColorGradeShadowSat",
    "ColorGradeShadowLum", "ColorGradeMidtoneHue", "ColorGradeMidtoneSat",
    "ColorGradeMidtoneLum", "ColorGradeHighlightHue", "ColorGradeHighlightSat",
    "ColorGradeHighlightLum", "ShadowTint", "RedHue", "RedSaturation",
    "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation"
]

# Non-feature columns in the features dataframe
NON_FEATURE_COLS = ["image_path", "file_path", "image_id", "folder"]

# Camera model encoding
UNKNOWN_CAMERA = "__unknown__"


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LightroomDataset(Dataset):
    """Dataset for Lightroom image features and edit parameters with NaN masking."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, masks: np.ndarray):
        """
        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples, n_targets) - NaN replaced with 0
            masks: Boolean mask array (n_samples, n_targets) - True where target is valid
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.masks = torch.FloatTensor(masks)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx], self.masks[idx]


def compute_masked_covariance(
    targets: np.ndarray,
    masks: np.ndarray
) -> np.ndarray:
    """
    Compute a covariance matrix using only valid (masked-in) target pairs.
    """
    n_targets = targets.shape[1]
    means = np.zeros(n_targets, dtype=np.float32)

    for i in range(n_targets):
        valid = masks[:, i].astype(bool)
        if valid.any():
            means[i] = targets[valid, i].mean()

    cov = np.zeros((n_targets, n_targets), dtype=np.float32)
    for i in range(n_targets):
        valid_i = masks[:, i].astype(bool)
        for j in range(i, n_targets):
            valid = valid_i & masks[:, j].astype(bool)
            n_valid = int(valid.sum())
            if n_valid < 2:
                cov_ij = 1.0 if i == j else 0.0
            else:
                vi = targets[valid, i] - means[i]
                vj = targets[valid, j] - means[j]
                cov_ij = float(np.dot(vi, vj) / (n_valid - 1))
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij

    return cov


def compute_target_precision(
    targets: np.ndarray,
    masks: np.ndarray,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Compute a stabilized precision matrix for Mahalanobis loss.
    """
    cov = compute_masked_covariance(targets, masks)
    cov = 0.5 * (cov + cov.T)

    # Pairwise (masked) covariance is not guaranteed to be PSD; project to SPD so the
    # quadratic form rᵀPr is non-negative and training can't "cheat" via negative modes.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None).astype(np.float32)
    cov_spd = (eigvecs * eigvals_clipped) @ eigvecs.T
    cov_spd = 0.5 * (cov_spd + cov_spd.T)

    precision = np.linalg.inv(cov_spd).astype(np.float32)
    precision = 0.5 * (precision + precision.T)
    return precision


def masked_mahalanobis_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    precision: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mahalanobis loss using only valid (masked-in) target values.
    """
    residual = (predictions - targets) * masks
    projected = residual @ precision
    per_sample = (projected * residual).sum(dim=1)
    valid_counts = masks.sum(dim=1).clamp(min=1)
    per_sample = per_sample / valid_counts
    return per_sample.mean()


class LearnedGlobalPrecision(nn.Module):
    """
    Learn a global SPD precision matrix P via a Cholesky factorization:
      P = L Lᵀ, where L is lower-triangular with positive diagonal.
    """

    def __init__(
        self,
        n_targets: int,
        *,
        init_precision: np.ndarray | None = None,
        min_diag: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_targets = int(n_targets)
        self.min_diag = float(min_diag)

        self.raw_diag = nn.Parameter(torch.zeros(self.n_targets, dtype=torch.float32))
        self.raw_lower = nn.Parameter(torch.zeros((self.n_targets, self.n_targets), dtype=torch.float32))

        if init_precision is not None:
            init_precision = np.asarray(init_precision, dtype=np.float32)
            if init_precision.shape != (self.n_targets, self.n_targets):
                raise ValueError(
                    f"init_precision shape {init_precision.shape} != ({self.n_targets}, {self.n_targets})"
                )
            L0 = np.linalg.cholesky(init_precision)
            diag0 = np.diag(L0).copy()
            diag0 = np.clip(diag0, a_min=self.min_diag, a_max=None)
            with torch.no_grad():
                self.raw_diag.copy_(torch.from_numpy(np.log(diag0)))
                self.raw_lower.copy_(torch.from_numpy(np.tril(L0, k=-1)))

    def precision_matrix(self) -> torch.Tensor:
        diag = torch.exp(self.raw_diag) + self.min_diag
        L = torch.tril(self.raw_lower, diagonal=-1) + torch.diag(diag)
        return L @ L.T


def masked_gaussian_nll_with_learned_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    precision: torch.Tensor,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Correlation-aware Gaussian NLL with missing targets.

    For each sample, uses the observed target subset S (from masks) and computes:
      0.5 * ( r_Sᵀ P_SS r_S - log|P_SS| )
    where P is a learned global precision and r = ŷ - y.

    Notes:
    - This respects cross-target correlation via P_SS.
    - It remains well-posed because the log-determinant term prevents trivial scaling.
    """
    mask_bool = masks > 0.5
    batch_size, n_targets = mask_bool.shape
    if n_targets != precision.shape[0] or n_targets != precision.shape[1]:
        raise ValueError(
            f"precision shape {tuple(precision.shape)} does not match n_targets={n_targets}"
        )

    residual = predictions - targets

    # Group identical masks so we can do one Cholesky per unique subset in the batch.
    if n_targets <= 63:
        shifts = (1 << torch.arange(n_targets, device=masks.device, dtype=torch.int64))
        keys = (mask_bool.to(torch.int64) * shifts).sum(dim=1)
    else:
        keys = torch.arange(batch_size, device=masks.device, dtype=torch.int64)

    total_loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    total_count = 0

    # Use sort-based grouping (stable on MPS) rather than torch.unique + equality masks.
    sorted_keys, perm = torch.sort(keys)
    change = torch.nonzero(sorted_keys[1:] != sorted_keys[:-1], as_tuple=False).squeeze(1) + 1
    boundaries = torch.cat(
        [
            torch.zeros(1, device=keys.device, dtype=change.dtype),
            change,
            torch.tensor([batch_size], device=keys.device, dtype=change.dtype),
        ]
    )

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start_i = int(start.item())
        end_i = int(end.item())
        if end_i <= start_i:
            continue
        idx = perm[start_i:end_i]
        if idx.numel() == 0:
            continue
        obs = torch.nonzero(mask_bool[idx[0]], as_tuple=False).squeeze(1)
        k = int(obs.numel())
        if k == 0:
            continue

        r = residual.index_select(0, idx).index_select(1, obs)  # (n_group, k)
        P_sub = precision.index_select(0, obs).index_select(1, obs)

        P_sub = P_sub + jitter * torch.eye(k, device=predictions.device, dtype=predictions.dtype)
        L = torch.linalg.cholesky(P_sub)  # lower-triangular, P_sub = L Lᵀ

        # Quadratic form q = rᵀ P r = || Lᵀ r ||² = || r L ||²
        z = r @ L
        q = (z * z).sum(dim=1)

        logdetP = 2.0 * torch.log(torch.diagonal(L)).sum()
        per_sample = 0.5 * (q - logdetP) / float(k)

        total_loss = total_loss + per_sample.sum()
        total_count += int(idx.numel())

    if total_count == 0:
        return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    return total_loss / float(total_count)


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe from parquet or csv, with fallback."""
    path = Path(path)
    
    # Try parquet first
    if path.suffix == '.parquet':
        try:
            return pd.read_parquet(path)
        except ImportError:
            # Fall back to CSV if pyarrow not installed
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                print(f"  (Using CSV fallback: {csv_path})")
                return pd.read_csv(csv_path)
            raise
    elif path.suffix == '.csv':
        return pd.read_csv(path)
    else:
        # Try parquet, then csv
        parquet_path = path.with_suffix('.parquet')
        csv_path = path.with_suffix('.csv')
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except ImportError:
                pass
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"Could not find parquet or csv for: {path}")


def encode_camera_models(
    camera_models: np.ndarray,
    known_cameras: list = None
) -> Tuple[np.ndarray, list]:
    """
    One-hot encode camera models.
    
    Args:
        camera_models: Array of camera model strings
        known_cameras: List of known camera names (for inference). 
                       If None, learns from data (for training).
    
    Returns:
        one_hot: numpy array of shape (n_samples, n_cameras)
        camera_list: list of camera names in order
    """
    # Clean camera names (handle NaN/empty)
    camera_models = np.array([
        str(c).strip() if pd.notna(c) and str(c).strip() else UNKNOWN_CAMERA 
        for c in camera_models
    ])
    
    if known_cameras is None:
        # Training mode: learn the camera list
        unique_cameras = sorted(set(camera_models) - {UNKNOWN_CAMERA})
        camera_list = unique_cameras + [UNKNOWN_CAMERA]
    else:
        camera_list = known_cameras
    
    # Create one-hot encoding
    n_samples = len(camera_models)
    n_cameras = len(camera_list)
    one_hot = np.zeros((n_samples, n_cameras), dtype=np.float32)
    
    camera_to_idx = {cam: idx for idx, cam in enumerate(camera_list)}
    unknown_idx = camera_to_idx.get(UNKNOWN_CAMERA, n_cameras - 1)
    
    for i, cam in enumerate(camera_models):
        idx = camera_to_idx.get(cam, unknown_idx)
        one_hot[i, idx] = 1.0
    
    return one_hot, camera_list


def extract_date_from_folder(folder: str) -> str:
    """Extract date (YYYY-MM-DD) from folder path."""
    if pd.isna(folder):
        return "unknown"
    # Match patterns like "2025-03-15" in the folder path
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(folder))
    if match:
        return match.group(1)
    return "unknown"


def expand_circular_targets(
    targets: np.ndarray,
    masks: np.ndarray,
    target_names: list
) -> Tuple[np.ndarray, np.ndarray, list, Dict[str, Tuple[str, str]]]:
    """
    Convert circular hue columns from degrees to (sin, cos) pairs.
    
    This handles the circular nature of hue values where 0° and 360° are the same.
    By encoding as (sin, cos), the model learns in a space where angular proximity
    is preserved (e.g., 5° and 355° are close together).
    
    Args:
        targets: Target array (n_samples, n_targets)
        masks: Mask array (n_samples, n_targets)
        target_names: List of target column names
        
    Returns:
        new_targets: Expanded target array with sin/cos columns
        new_masks: Expanded mask array  
        new_target_names: Updated column names
        circular_info: Dict mapping original column name to (sin_name, cos_name)
    """
    new_targets_list = []
    new_masks_list = []
    new_names = []
    circular_info = {}
    
    for i, name in enumerate(target_names):
        if name in CIRCULAR_COLUMNS:
            # Convert degrees to radians
            radians = np.deg2rad(targets[:, i])
            sin_vals = np.sin(radians)
            cos_vals = np.cos(radians)
            
            # Both sin and cos inherit the same mask as original
            new_targets_list.append(sin_vals.reshape(-1, 1))
            new_targets_list.append(cos_vals.reshape(-1, 1))
            new_masks_list.append(masks[:, i].reshape(-1, 1))
            new_masks_list.append(masks[:, i].reshape(-1, 1))
            
            sin_name = f"{name}_sin"
            cos_name = f"{name}_cos"
            new_names.extend([sin_name, cos_name])
            circular_info[name] = (sin_name, cos_name)
        else:
            new_targets_list.append(targets[:, i].reshape(-1, 1))
            new_masks_list.append(masks[:, i].reshape(-1, 1))
            new_names.append(name)
    
    new_targets = np.hstack(new_targets_list)
    new_masks = np.hstack(new_masks_list)
    
    return new_targets, new_masks, new_names, circular_info


def load_and_prepare_data(
    features_path: str = "image_features.parquet",
    edits_path: str = "processed_images.parquet"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, np.ndarray, list, np.ndarray, Dict[str, Tuple[str, str]]]:
    """
    Load and merge feature and edit data.
    
    Returns:
        features: numpy array of shape (n_samples, n_features)
        targets: numpy array of shape (n_samples, n_targets) - NaN replaced with 0
        target_masks: boolean array of shape (n_samples, n_targets) - True where valid
        feature_names: list of feature column names
        target_names: list of target column names (with circular targets expanded to _sin/_cos)
        camera_one_hot: numpy array of shape (n_samples, n_cameras)
        camera_list: list of camera model names
        dates: numpy array of date strings for each sample
        circular_info: dict mapping original hue column names to (sin_name, cos_name) tuples
    """
    print("Loading data...")
    
    # Load dataframes (with CSV fallback if parquet fails)
    features_df = load_dataframe(features_path)
    edits_df = load_dataframe(edits_path)
    
    print(f"  Features: {len(features_df)} samples, {len(features_df.columns)} columns")
    print(f"  Edits: {len(edits_df)} samples, {len(edits_df.columns)} columns")
    
    # Determine merge key (prefer image_id, fallback to file_path)
    if "image_id" in features_df.columns and "image_id" in edits_df.columns:
        merge_key = "image_id"
    elif "file_path" in features_df.columns and "file_path" in edits_df.columns:
        merge_key = "file_path"
    else:
        raise ValueError(
            f"No common merge key found. Features cols: {features_df.columns.tolist()[:10]}, "
            f"Edits cols: {edits_df.columns.tolist()[:10]}"
        )
    
    print(f"  Merging on: {merge_key}")
    
    # Determine which target columns are available
    available_targets = [col for col in TARGET_COLUMNS if col in edits_df.columns]
    if len(available_targets) < len(TARGET_COLUMNS):
        missing = set(TARGET_COLUMNS) - set(available_targets)
        print(f"  Warning: Missing {len(missing)} target columns: {list(missing)[:5]}...")
    
    # Include camera_model and folder in merge if available
    merge_cols = [merge_key] + available_targets
    if "camera_model" in edits_df.columns:
        merge_cols.append("camera_model")
    if "folder" in edits_df.columns:
        merge_cols.append("folder")
    
    merged_df = features_df.merge(edits_df[merge_cols], on=merge_key, how="inner")
    print(f"  Merged: {len(merged_df)} samples")
    
    if len(merged_df) == 0:
        raise ValueError(f"No matching samples found after merge on {merge_key}.")
    
    # Get feature columns (all columns except non-feature and target columns)
    exclude_cols = set(NON_FEATURE_COLS) | set(available_targets) | {"camera_model"}
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Extract features and targets (keep NaN info before replacing)
    features = merged_df[feature_cols].values.astype(np.float32)
    targets_raw = merged_df[available_targets].values.astype(np.float32)
    
    # Create mask: True where target is valid (not NaN AND not 0)
    # 0 typically means "didn't touch this slider" rather than "explicitly set to 0"
    target_masks = ~np.isnan(targets_raw) & (targets_raw != 0)

    # For color grading saturation columns, 0 IS a valid/meaningful value
    # Override those columns to include 0 (only exclude NaN)
    for col_name in COLOR_GRADING_SAT_COLUMNS:
        if col_name in available_targets:
            col_idx = available_targets.index(col_name)
            target_masks[:, col_idx] = ~np.isnan(targets_raw[:, col_idx])
    
    # Count valid values per target
    valid_counts = target_masks.sum(axis=0)
    print(f"\n  Target validity (non-NaN and non-zero counts):")
    for name, count in zip(available_targets, valid_counts):
        pct = 100 * count / len(targets_raw)
        if pct < 100:
            print(f"    {name}: {count}/{len(targets_raw)} ({pct:.1f}%)")
    
    # Replace NaN with 0 for computation (masked out during loss)
    targets = np.nan_to_num(targets_raw, nan=0.0)
    
    # Expand circular hue columns to sin/cos pairs
    circular_in_targets = [c for c in CIRCULAR_COLUMNS if c in available_targets]
    if circular_in_targets:
        print(f"\n  Converting circular hue columns to sin/cos pairs:")
        print(f"    {circular_in_targets}")
        targets, target_masks, available_targets, circular_info = expand_circular_targets(
            targets, target_masks, available_targets
        )
        print(f"    New target dimensions: {targets.shape}")
    else:
        circular_info = {}
    
    # Handle NaN in features (replace with 0)
    features = np.nan_to_num(features, nan=0.0)
    
    # Extract dates for folder-based splitting
    if "folder" in merged_df.columns:
        dates = np.array([extract_date_from_folder(f) for f in merged_df["folder"].values])
    else:
        print("  Warning: 'folder' column not found, using random split")
        dates = np.array(["unknown"] * len(merged_df))
    
    # Encode camera models
    if "camera_model" in merged_df.columns:
        camera_one_hot, camera_list = encode_camera_models(merged_df["camera_model"].values)
        print(f"  Camera models: {camera_list}")
    else:
        print("  Warning: camera_model not found in data, skipping camera encoding")
        camera_one_hot = np.zeros((len(features), 1), dtype=np.float32)
        camera_list = [UNKNOWN_CAMERA]
    
    print(f"\n  Feature dimensions: {features.shape}")
    print(f"  Camera one-hot dimensions: {camera_one_hot.shape}")
    print(f"  Target dimensions: {targets.shape}")
    print(f"  Valid target ratio: {target_masks.mean():.1%}")
    
    return features, targets, target_masks, feature_cols, available_targets, camera_one_hot, camera_list, dates, circular_info


def split_by_date(
    dates: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data by date to avoid leakage between train and validation sets.
    
    Args:
        dates: Array of date strings for each sample
        val_split: Fraction of dates to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_idx: Indices for training set
        val_idx: Indices for validation set
    """
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(dates)
    n_val_dates = max(1, int(len(unique_dates) * val_split))
    
    # Shuffle and select validation dates
    shuffled_dates = rng.permutation(unique_dates)
    val_dates = set(shuffled_dates[:n_val_dates])
    
    # Create index arrays
    train_idx = np.array([i for i, d in enumerate(dates) if d not in val_dates])
    val_idx = np.array([i for i, d in enumerate(dates) if d in val_dates])
    
    print(f"\n  Date-based split:")
    print(f"    Total dates: {len(unique_dates)}")
    print(f"    Train dates: {len(unique_dates) - n_val_dates} → {len(train_idx)} samples")
    print(f"    Val dates: {n_val_dates} → {len(val_idx)} samples")
    print(f"    Val dates: {sorted(val_dates)[:5]}{'...' if len(val_dates) > 5 else ''}")
    
    return train_idx, val_idx


def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    train_normalized = (train_features - mean) / std
    
    if val_features is not None:
        val_normalized = (val_features - mean) / std
    else:
        val_normalized = None
    
    return train_normalized, val_normalized, mean, std


def normalize_targets(
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    val_targets: np.ndarray = None,
    val_masks: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize targets to zero mean and unit variance.
    Only computes statistics on valid (non-NaN) values.
    
    This is crucial when targets have different scales (e.g., Temperature ~5000 vs Exposure ~0.5).
    Without normalization, the loss would be dominated by large-scale targets.
    """
    n_targets = train_targets.shape[1]
    mean = np.zeros(n_targets, dtype=np.float32)
    std = np.ones(n_targets, dtype=np.float32)
    
    for i in range(n_targets):
        valid_mask = train_masks[:, i].astype(bool)
        if valid_mask.sum() > 0:
            valid_values = train_targets[valid_mask, i]
            mean[i] = valid_values.mean()
            std[i] = valid_values.std()
            if std[i] == 0:
                std[i] = 1  # Avoid division by zero
    
    train_normalized = (train_targets - mean) / std
    
    if val_targets is not None:
        val_normalized = (val_targets - mean) / std
    else:
        val_normalized = None
    
    return train_normalized, val_normalized, mean, std


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision_module: LearnedGlobalPrecision,
) -> float:
    """Train for one epoch with masked Gaussian NLL (learned precision)."""
    model.train()
    precision_module.train()
    total_loss = 0.0
    total_samples = 0
    
    for features, targets, masks in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        precision = precision_module.precision_matrix()
        loss = masked_gaussian_nll_with_learned_precision(
            predictions, targets, masks, precision
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(features)
        total_samples += len(features)
    
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    precision_module: LearnedGlobalPrecision,
) -> float:
    """Evaluate the model with masked Gaussian NLL (learned precision)."""
    model.eval()
    precision_module.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for features, targets, masks in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            predictions = model(features)
            precision = precision_module.precision_matrix()
            loss = masked_gaussian_nll_with_learned_precision(
                predictions, targets, masks, precision
            )
            total_loss += loss.item() * len(features)
            total_samples += len(features)
    
    return total_loss / total_samples


def train(
    features_path: str = "image_features.parquet",
    edits_path: str = "processed_images.parquet",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dims: list = None,
    dropout: float = 0.2,
    val_split: float = 0.2,
    save_path: str = "model_weights/lightroom_model_learned_precision.pt",
    seed: int = 42,
    split_seed: int | None = None,
) -> nn.Module:
    """
    Main training function.
    
    Args:
        features_path: Path to features parquet file
        edits_path: Path to edits parquet file
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        val_split: Fraction of dates to use for validation
        save_path: Path to save the trained model
        seed: Random seed for model init + training
        split_seed: Random seed for train/val split (defaults to `seed`)
    
    Returns:
        Trained model
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    effective_split_seed = seed if split_seed is None else split_seed
    
    # Ensure model_weights directory exists
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data (now includes masks, dates, and circular info)
    features, targets, target_masks, feature_names, target_names, camera_one_hot, camera_list, dates, circular_info = load_and_prepare_data(
        features_path, edits_path
    )
    
    # Concatenate image features with camera one-hot encoding
    # Camera one-hot doesn't need normalization (already 0/1)
    features_with_camera = np.concatenate([features, camera_one_hot], axis=1)
    print(f"  Combined features (image + camera): {features_with_camera.shape}")
    
    # Split by date to avoid leakage
    train_idx, val_idx = split_by_date(dates, val_split=val_split, seed=effective_split_seed)
    
    # Split data
    train_features = features_with_camera[train_idx]
    train_targets = targets[train_idx]
    train_masks = target_masks[train_idx]
    val_features = features_with_camera[val_idx]
    val_targets = targets[val_idx]
    val_masks = target_masks[val_idx]
    
    # Also split the raw image features (without camera) for normalization stats
    train_img_features = features[train_idx]
    val_img_features = features[val_idx]
    train_camera = camera_one_hot[train_idx]
    val_camera = camera_one_hot[val_idx]
    
    print(f"\nTrain samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Normalize only image features (camera one-hot stays as-is)
    train_img_norm, val_img_norm, feature_mean, feature_std = normalize_features(
        train_img_features, val_img_features
    )
    
    # Recombine normalized image features with camera one-hot
    train_features_norm = np.concatenate([train_img_norm, train_camera], axis=1)
    val_features_norm = np.concatenate([val_img_norm, val_camera], axis=1)
    
    # Normalize targets (only on valid values)
    train_targets_norm, val_targets_norm, target_mean, target_std = normalize_targets(
        train_targets, train_masks, val_targets, val_masks
    )
    
    print(f"Target normalization applied (scales now balanced)")

    # Initialize a learnable precision matrix from the empirical (masked) estimate.
    target_precision_init = compute_target_precision(train_targets_norm, train_masks)
    precision_module = LearnedGlobalPrecision(
        len(target_names), init_precision=target_precision_init
    ).to(device)
    
    # Create datasets and dataloaders (now with masks)
    train_dataset = LightroomDataset(train_features_norm, train_targets_norm, train_masks)
    val_dataset = LightroomDataset(val_features_norm, val_targets_norm, val_masks)
    
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=data_gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (input_dim now includes camera one-hot)
    input_dim = train_features_norm.shape[1]
    output_dim = len(target_names)
    
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]
    
    model = LightroomMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": 1e-4},
            {"params": precision_module.parameters(), "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, precision_module)
        val_loss = evaluate(model, val_loader, device, precision_module)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'target_mean': target_mean,
                'target_std': target_std,
                'target_precision_init': target_precision_init,
                'learned_precision_raw_diag': precision_module.raw_diag.detach().cpu().numpy(),
                'learned_precision_raw_lower': precision_module.raw_lower.detach().cpu().numpy(),
                'learned_precision_min_diag': float(precision_module.min_diag),
                'feature_names': feature_names,
                'target_names': target_names,
                'hidden_dims': hidden_dims,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'camera_list': camera_list,  # For encoding camera at inference time
                'circular_info': circular_info,  # For reconstructing hue from sin/cos
            }, save_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    # Load best model
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def train_ensemble(
    *,
    seeds: list[int],
    features_path: str = "image_features.parquet",
    edits_path: str = "processed_images.parquet",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dims: list | None = None,
    dropout: float = 0.2,
    val_split: float = 0.2,
    save_path_template: str = "model_weights/lightroom_model_learned_precision_seed{seed}.pt",
    split_seed: int | None = None,
) -> pd.DataFrame:
    """
    Train multiple models (different seeds) and report aggregate validation stats.

    Returns a DataFrame with per-seed best validation loss and save path.
    """
    # Ensure model_weights directory exists
    first_save_path = save_path_template.format(seed=seeds[0])
    Path(first_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for seed in seeds:
        save_path = save_path_template.format(seed=seed)
        print(f"\n=== Training seed={seed} → {save_path} ===")
        train(
            features_path=features_path,
            edits_path=edits_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            dropout=dropout,
            val_split=val_split,
            save_path=save_path,
            seed=seed,
            split_seed=split_seed,
        )
        checkpoint = torch.load(save_path, weights_only=False)
        rows.append(
            {
                "seed": seed,
                "save_path": save_path,
                "best_val_loss": float(checkpoint.get("val_loss", np.nan)),
                "best_epoch": int(checkpoint.get("epoch", -1)),
            }
        )

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if len(df) > 0:
        print("\nEnsemble summary:")
        print(df.to_string(index=False))
        if df["best_val_loss"].notna().any():
            mean = float(df["best_val_loss"].mean())
            std = float(df["best_val_loss"].std(ddof=1)) if len(df) > 1 else 0.0
            print(f"\nBest val loss: mean={mean:.4f} std={std:.4f} (n={len(df)})")
    return df


if __name__ == "__main__":
    # ===== CONFIGURE HERE =====
    # Train 1 model: set a single seed (e.g. [71])
    # Train N models (ensemble): add more seeds (e.g. [71, 72, 73, 74, 75])
    seeds = [69]

    # If you want the train/val split to stay identical across seeds, set split_seed.
    # If None, each model uses its own seed for splitting too.
    split_seed = 42

    features_path = "image_features.parquet"
    edits_path = "processed_images.parquet"
    epochs = 500
    batch_size = 256
    learning_rate = 1e-3
    hidden_dims = [2048, 1024, 1024, 512, 57]
    dropout = 0.3
    val_split = 0.2

    # Single-model output path (only used when len(seeds)==1)
    save_path = "model_weights/lightroom_model_learned_precision.pt"

    # Ensemble output template (used when len(seeds)>1)
    save_path_template = "model_weights/lightroom_model_learned_precision_seed{seed}.pt"
    # ==========================

    if len(seeds) <= 0:
        raise ValueError("Please provide at least one seed in `seeds`.")

    if len(seeds) == 1:
        train(
            features_path=features_path,
            edits_path=edits_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            dropout=dropout,
            val_split=val_split,
            save_path=save_path,
            seed=seeds[0],
            split_seed=split_seed,
        )
    else:
        train_ensemble(
            seeds=seeds,
            features_path=features_path,
            edits_path=edits_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            dropout=dropout,
            val_split=val_split,
            save_path_template=save_path_template,
            split_seed=split_seed,
        )

# 
