#!/usr/bin/env python3
"""
Predict Lightroom edits for a single image.

Usage:
    python predict_edits.py <image_path>
    python predict_edits.py "/Users/eriksreinfelds/Pictures/2025/2025-12-09/DSC08353.ARW"
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List

from feature_pipeline import extract_all_features
from models import LightroomMLP

# Camera model encoding
UNKNOWN_CAMERA = "__unknown__"


def sincos_to_degrees(sin_val: float, cos_val: float) -> float:
    """
    Convert sin/cos pair back to angle in degrees [0, 360).
    
    Uses atan2 which handles all quadrants correctly.
    """
    # If the vector magnitude is near-zero, the angle is effectively undefined.
    # This can happen when averaging disagreeing ensemble members.
    if (sin_val * sin_val + cos_val * cos_val) < 1e-12:
        return 0.0
    radians = np.arctan2(sin_val, cos_val)
    degrees = np.rad2deg(radians)
    # Normalize to [0, 360)
    return float(degrees % 360)


def reconstruct_circular_targets(
    predictions: dict,
    circular_info: Dict[str, Tuple[str, str]]
) -> dict:
    """
    Reconstruct original hue columns from sin/cos predictions.
    
    Args:
        predictions: Dict of target_name -> value (includes _sin and _cos columns)
        circular_info: Dict mapping original hue name to (sin_name, cos_name)
        
    Returns:
        Updated predictions dict with hue columns reconstructed and sin/cos removed
    """
    result = predictions.copy()
    
    for original_name, (sin_name, cos_name) in circular_info.items():
        if sin_name in result and cos_name in result:
            sin_val = result.pop(sin_name)
            cos_val = result.pop(cos_name)
            result[original_name] = sincos_to_degrees(sin_val, cos_val)
    
    return result


def encode_camera_model(camera_model: Optional[str], camera_list: list) -> np.ndarray:
    """
    One-hot encode a single camera model.
    
    Args:
        camera_model: Camera model name (e.g., "ILCE-6700", "X-H2S")
        camera_list: List of known camera names from training
        
    Returns:
        One-hot encoded array of shape (n_cameras,)
    """
    # Clean camera name
    if camera_model is None or not str(camera_model).strip():
        camera_model = UNKNOWN_CAMERA
    else:
        camera_model = str(camera_model).strip()
    
    n_cameras = len(camera_list)
    one_hot = np.zeros(n_cameras, dtype=np.float32)
    
    camera_to_idx = {cam: idx for idx, cam in enumerate(camera_list)}
    unknown_idx = camera_to_idx.get(UNKNOWN_CAMERA, n_cameras - 1)
    
    idx = camera_to_idx.get(camera_model, unknown_idx)
    one_hot[idx] = 1.0
    
    return one_hot


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: str = "model_weights/lightroom_model.pt") -> Tuple[nn.Module, dict]:
    """Load a trained model from disk."""
    checkpoint = torch.load(path, weights_only=False)
    
    model = LightroomMLP(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        output_dim=checkpoint['output_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_models(paths: List[str]) -> List[Tuple[nn.Module, dict]]:
    """Load multiple trained models from disk."""
    models = []
    for path in paths:
        model, checkpoint = load_model(path)
        models.append((model, checkpoint))
    return models


def denormalize_targets(
    predictions: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray
) -> np.ndarray:
    """Convert normalized predictions back to original scale."""
    return predictions * target_std + target_mean


def predict(
    model: nn.Module,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray = None,
    target_std: np.ndarray = None,
    device: torch.device = None,
    camera_features: bool = False
) -> np.ndarray:
    """
    Make predictions for new features.
    
    Args:
        model: Trained model
        features: Feature array of shape (n_samples, n_features + n_cameras if camera_features)
        feature_mean: Feature mean from training (for image features only)
        feature_std: Feature std from training (for image features only)
        target_mean: Target mean from training (for denormalization)
        target_std: Target std from training (for denormalization)
        device: Device to run on
        camera_features: If True, last columns are camera one-hot (don't normalize them)
    
    Returns:
        Predictions array of shape (n_samples, n_targets) in original scale
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    model.eval()
    
    # Normalize features (but not camera one-hot)
    n_img_features = len(feature_mean)
    
    if camera_features and features.shape[1] > n_img_features:
        # Split into image features and camera one-hot
        img_features = features[:, :n_img_features]
        camera_one_hot = features[:, n_img_features:]
        
        # Normalize only image features
        img_features_norm = (img_features - feature_mean) / feature_std
        
        # Recombine
        features_norm = np.concatenate([img_features_norm, camera_one_hot], axis=1)
    else:
        # No camera features, normalize everything
        features_norm = (features - feature_mean) / feature_std
    
    features_tensor = torch.FloatTensor(features_norm).to(device)
    
    with torch.no_grad():
        predictions = model(features_tensor)
    
    predictions_np = predictions.cpu().numpy()
    
    # Denormalize predictions if target stats provided
    if target_mean is not None and target_std is not None:
        predictions_np = denormalize_targets(predictions_np, target_mean, target_std)
    
    return predictions_np


# Valid Lightroom parameter ranges
PARAM_RANGES = {
    # Special ranges
    "IncrementalTemperature": (-3000, 3000),  # Delta from As Shot
    "IncrementalTint": (-50, 50),              # Delta from As Shot
    "Exposure2012": (-5, 5),
    # Color Grading Hue (0-360)
    "ColorGradeGlobalHue": (0, 360),
    "ColorGradeShadowHue": (0, 360),
    "ColorGradeMidtoneHue": (0, 360),
    "ColorGradeHighlightHue": (0, 360),
    # Color Grading Saturation (0-100)
    "ColorGradeGlobalSat": (0, 100),
    "ColorGradeShadowSat": (0, 100),
    "ColorGradeMidtoneSat": (0, 100),
    "ColorGradeHighlightSat": (0, 100),
    # Color Grading Blending (0-100)
    "ColorGradeBlending": (0, 100),
    # Everything else is -100 to 100 (default)
}
DEFAULT_RANGE = (-100, 100)


def clamp_to_range(name: str, value: float) -> int | float:
    """Clamp a value to its valid Lightroom range and round appropriately."""
    min_val, max_val = PARAM_RANGES.get(name, DEFAULT_RANGE)
    clamped = max(min_val, min(max_val, value))
    
    # Exposure gets 2 decimal places, everything else is integer
    if name == "Exposure2012":
        return round(clamped, 2)
    return round(clamped)


def predict_edits(
    image_path: str, 
    model_path: str = "model_weights/lightroom_model.pt",
    model_paths: Optional[List[str]] = None,
    camera_model: Optional[str] = None
) -> dict:
    """
    Predict Lightroom edit parameters for an image.
    
    Args:
        image_path: Path to the image file (RAW or standard format)
        model_path: Path to the trained model checkpoint (single-model mode)
        model_paths: Optional list of model checkpoints (ensemble mode). If provided,
                     predictions are averaged across models (in target space, including
                     hue sin/cos if present) before reconstruction/clamping.
        camera_model: Camera model name (e.g., "ILCE-6700", "X-H2S", "Canon EOS 600D")
                      If None, uses "unknown" camera encoding.
        
    Returns:
        Dictionary mapping parameter names to predicted values
    """
    if model_paths is None and "," in model_path:
        model_paths = [p.strip() for p in model_path.split(",") if p.strip()]

    if model_paths:
        print(f"Loading {len(model_paths)} models...", file=sys.stderr)
        loaded = load_models(model_paths)
    else:
        print(f"Loading model from {model_path}...", file=sys.stderr)
        loaded = [load_model(model_path)]
    
    print(f"Extracting features from {image_path}...", file=sys.stderr)
    features = extract_all_features(image_path)

    # Reshape for batch prediction (1 sample) - image features only
    img_features_array = features.combined.reshape(1, -1)

    # Sanity checks across ensemble members (target ordering, camera encoding)
    base_checkpoint = loaded[0][1]
    base_target_names = base_checkpoint["target_names"]
    base_circular_info = base_checkpoint.get("circular_info", {})
    base_camera_list = base_checkpoint.get("camera_list")
    base_has_camera = base_camera_list is not None

    for _, ckpt in loaded[1:]:
        if ckpt["target_names"] != base_target_names:
            raise ValueError("Ensemble models have different target_names; cannot average.")
        if ckpt.get("circular_info", {}) != base_circular_info:
            raise ValueError("Ensemble models have different circular_info; cannot average.")
        has_camera = ckpt.get("camera_list") is not None
        if has_camera != base_has_camera:
            raise ValueError("Cannot ensemble-mix models with and without camera encoding.")
        if base_has_camera and ckpt.get("camera_list") != base_camera_list:
            raise ValueError("Ensemble models have different camera_list; cannot average.")

    # Build camera one-hot once (if needed)
    if base_has_camera:
        camera_one_hot = encode_camera_model(camera_model, base_camera_list)
        if camera_model:
            if camera_model in base_camera_list:
                print(f"Using camera: {camera_model}", file=sys.stderr)
            else:
                print(f"Unknown camera '{camera_model}', using fallback encoding", file=sys.stderr)
                print(f"  Known cameras: {[c for c in base_camera_list if c != UNKNOWN_CAMERA]}", file=sys.stderr)
    else:
        camera_one_hot = None
        if camera_model is not None:
            print("WARNING: Model was trained without camera encoding; camera_model ignored.", file=sys.stderr)

    # Predict per-model (each model uses its own normalization stats), then average
    preds = []
    print("Predicting edits...", file=sys.stderr)
    for i, (model, checkpoint) in enumerate(loaded):
        if 'target_mean' not in checkpoint or 'target_std' not in checkpoint:
            print("WARNING: A model was trained without target normalization; predictions may be mis-scaled.", file=sys.stderr)

        features_array = img_features_array
        if camera_one_hot is not None:
            features_array = np.concatenate([img_features_array, camera_one_hot.reshape(1, -1)], axis=1)

        if features_array.shape[1] != checkpoint["input_dim"]:
            raise ValueError(
                f"Feature dimension mismatch for model {i}: got {features_array.shape[1]}, "
                f"expected {checkpoint['input_dim']} (did you change the feature pipeline?)."
            )

        pred = predict(
            model=model,
            features=features_array,
            feature_mean=checkpoint['feature_mean'],
            feature_std=checkpoint['feature_std'],
            target_mean=checkpoint.get('target_mean'),
            target_std=checkpoint.get('target_std'),
            camera_features=camera_one_hot is not None,
        )
        preds.append(pred[0])

    predictions_mean = np.mean(np.stack(preds, axis=0), axis=0)

    # Build result dictionary from averaged predictions
    raw_result = {name: value for name, value in zip(base_target_names, predictions_mean)}

    # Reconstruct circular hue values from sin/cos pairs (if present)
    if base_circular_info:
        print(f"Reconstructing hue from sin/cos for: {list(base_circular_info.keys())}", file=sys.stderr)
        raw_result = reconstruct_circular_targets(raw_result, base_circular_info)

    # Apply clamping to all values
    result = {name: clamp_to_range(name, value) for name, value in raw_result.items()}
    return result


def format_edits(edits: dict, image_path: str = None) -> str:
    """Format edits as a nicely aligned, readable string."""
    lines = []
    
    # Group by category with clean parameter names
    categories = {
        "Basic": [
            ("Exposure2012", "Exposure"),
            ("Contrast2012", "Contrast"),
            ("Highlights2012", "Highlights"),
            ("Shadows2012", "Shadows"),
            ("Whites2012", "Whites"),
            ("Blacks2012", "Blacks"),
            ("Texture", "Texture"),
            ("Clarity2012", "Clarity"),
            ("Dehaze", "Dehaze"),
        ],
        "Color": [
            ("IncrementalTemperature", "Temp Δ"),
            ("IncrementalTint", "Tint Δ"),
            ("Vibrance", "Vibrance"),
            ("Saturation", "Saturation"),
        ],
        "HSL Hue": [
            ("HueAdjustmentRed", "Red"),
            ("HueAdjustmentOrange", "Orange"),
            ("HueAdjustmentYellow", "Yellow"),
            ("HueAdjustmentGreen", "Green"),
            ("HueAdjustmentAqua", "Aqua"),
            ("HueAdjustmentBlue", "Blue"),
            ("HueAdjustmentPurple", "Purple"),
            ("HueAdjustmentMagenta", "Magenta"),
        ],
        "HSL Saturation": [
            ("SaturationAdjustmentRed", "Red"),
            ("SaturationAdjustmentOrange", "Orange"),
            ("SaturationAdjustmentYellow", "Yellow"),
            ("SaturationAdjustmentGreen", "Green"),
            ("SaturationAdjustmentAqua", "Aqua"),
            ("SaturationAdjustmentBlue", "Blue"),
            ("SaturationAdjustmentPurple", "Purple"),
            ("SaturationAdjustmentMagenta", "Magenta"),
        ],
        "HSL Luminance": [
            ("LuminanceAdjustmentRed", "Red"),
            ("LuminanceAdjustmentOrange", "Orange"),
            ("LuminanceAdjustmentYellow", "Yellow"),
            ("LuminanceAdjustmentGreen", "Green"),
            ("LuminanceAdjustmentAqua", "Aqua"),
            ("LuminanceAdjustmentBlue", "Blue"),
            ("LuminanceAdjustmentPurple", "Purple"),
            ("LuminanceAdjustmentMagenta", "Magenta"),
        ],
        "Color Grading": [
            ("ColorGradeBlending", "Blending"),
            ("ColorGradeGlobalHue", "Global Hue"),
            ("ColorGradeGlobalSat", "Global Sat"),
            ("ColorGradeGlobalLum", "Global Lum"),
            ("ColorGradeShadowHue", "Shadow Hue"),
            ("ColorGradeShadowSat", "Shadow Sat"),
            ("ColorGradeShadowLum", "Shadow Lum"),
            ("ColorGradeMidtoneHue", "Midtone Hue"),
            ("ColorGradeMidtoneSat", "Midtone Sat"),
            ("ColorGradeMidtoneLum", "Midtone Lum"),
            ("ColorGradeHighlightHue", "Highlight Hue"),
            ("ColorGradeHighlightSat", "Highlight Sat"),
            ("ColorGradeHighlightLum", "Highlight Lum"),
        ],
        "Calibration": [
            ("ShadowTint", "Shadow Tint"),
            ("RedHue", "Red Hue"),
            ("RedSaturation", "Red Sat"),
            ("GreenHue", "Green Hue"),
            ("GreenSaturation", "Green Sat"),
            ("BlueHue", "Blue Hue"),
            ("BlueSaturation", "Blue Sat"),
        ],
    }
    
    if image_path:
        lines.append(f"Image: {image_path}")
        lines.append("")
    
    for category, params in categories.items():
        available = [(key, name) for key, name in params if key in edits]
        if available:
            lines.append(f"┌─ {category} " + "─" * (40 - len(category)))
            for key, name in available:
                value = edits[key]
                # Format value with sign
                if key == "Exposure2012":
                    # Exposure is a float with 2dp
                    if value == 0:
                        val_str = "  0.00"
                    elif value > 0:
                        val_str = f"+{value:5.2f}"
                    else:
                        val_str = f"{value:6.2f}"
                else:
                    # Everything else is integer
                    if value == 0:
                        val_str = "     0"
                    elif value > 0:
                        val_str = f"+{value:5d}"
                    else:
                        val_str = f"{value:6d}"
                lines.append(f"│  {name:<20} {val_str}")
            lines.append("└" + "─" * 44)
            lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # ===== CONFIGURE HERE =====
    image_path = "/Users/eriksreinfelds/Pictures/2025/2025-12-09/DSC08259.ARW"
    # Single-model:
    #model_path = "model_weights/lightroom_model.pt"
    # Ensemble (either option works):
    # - Comma-separated string:
    #   model_path = "model_weights/lightroom_model_seed71.pt,model_weights/lightroom_model_seed72.pt,model_weights/lightroom_model_seed73.pt"
    # - Or explicit list (recommended if paths have commas/spaces):
    model_paths = [
        "model_weights/lightroom_model_seed69.pt"
    ]
    #model_paths = None
    
    # Camera model - set this based on which camera took the image
    # Known cameras: "ILCE-6700" (Sony), "X-H2S" (Fujifilm), "Canon EOS 600D"
    # Set to None if unknown
    camera_model = "ILCE-6700"
    # ==========================
    
    edits = predict_edits(image_path, model_paths=model_paths, camera_model=camera_model)
    
    print("\n")
    print("╔" + "═" * 46 + "╗")
    print("║" + "  PREDICTED LIGHTROOM EDITS".center(46) + "║")
    print("╚" + "═" * 46 + "╝")
    print() 
    print(format_edits(edits, image_path))
