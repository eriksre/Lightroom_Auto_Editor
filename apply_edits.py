#!/usr/bin/env python3
"""
Apply predicted Lightroom edits via XMP sidecar files.

This module generates XMP sidecar files that Lightroom will read when you:
1. Select the image(s) in Lightroom
2. Go to Photo → Read Metadata from Files (or Ctrl/Cmd+Shift+R)

Alternatively, enable "Automatically read XMP metadata from files" in 
Catalog Settings → Metadata tab.

Usage:
    python apply_edits.py <image_path> [--camera <camera_model>]
    
Example:
    python apply_edits.py "/Users/me/Pictures/DSC08353.ARW" --camera "ILCE-6700"
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import uuid

import numpy as np

from predict_edits import (
    predict_edits, format_edits, load_model, load_models, predict,
    encode_camera_model, reconstruct_circular_targets, clamp_to_range, UNKNOWN_CAMERA
)
from feature_pipeline import extract_batch


# XMP template with Camera Raw settings
XMP_TEMPLATE = '''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Lightroom Auto Editor">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
{settings_attributes}>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''


# Mapping from our prediction names to XMP attribute names
# Most are the same, but Temperature/Tint need special handling
PARAM_TO_XMP = {
    # Basic
    "Exposure2012": "crs:Exposure2012",
    "Contrast2012": "crs:Contrast2012",
    "Highlights2012": "crs:Highlights2012",
    "Shadows2012": "crs:Shadows2012",
    "Whites2012": "crs:Whites2012",
    "Blacks2012": "crs:Blacks2012",
    "Texture": "crs:Texture",
    "Clarity2012": "crs:Clarity2012",
    "Dehaze": "crs:Dehaze",
    "Vibrance": "crs:Vibrance",
    "Saturation": "crs:Saturation",
    
    # HSL Hue
    "HueAdjustmentRed": "crs:HueAdjustmentRed",
    "HueAdjustmentOrange": "crs:HueAdjustmentOrange",
    "HueAdjustmentYellow": "crs:HueAdjustmentYellow",
    "HueAdjustmentGreen": "crs:HueAdjustmentGreen",
    "HueAdjustmentAqua": "crs:HueAdjustmentAqua",
    "HueAdjustmentBlue": "crs:HueAdjustmentBlue",
    "HueAdjustmentPurple": "crs:HueAdjustmentPurple",
    "HueAdjustmentMagenta": "crs:HueAdjustmentMagenta",
    
    # HSL Saturation
    "SaturationAdjustmentRed": "crs:SaturationAdjustmentRed",
    "SaturationAdjustmentOrange": "crs:SaturationAdjustmentOrange",
    "SaturationAdjustmentYellow": "crs:SaturationAdjustmentYellow",
    "SaturationAdjustmentGreen": "crs:SaturationAdjustmentGreen",
    "SaturationAdjustmentAqua": "crs:SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue": "crs:SaturationAdjustmentBlue",
    "SaturationAdjustmentPurple": "crs:SaturationAdjustmentPurple",
    "SaturationAdjustmentMagenta": "crs:SaturationAdjustmentMagenta",
    
    # HSL Luminance
    "LuminanceAdjustmentRed": "crs:LuminanceAdjustmentRed",
    "LuminanceAdjustmentOrange": "crs:LuminanceAdjustmentOrange",
    "LuminanceAdjustmentYellow": "crs:LuminanceAdjustmentYellow",
    "LuminanceAdjustmentGreen": "crs:LuminanceAdjustmentGreen",
    "LuminanceAdjustmentAqua": "crs:LuminanceAdjustmentAqua",
    "LuminanceAdjustmentBlue": "crs:LuminanceAdjustmentBlue",
    "LuminanceAdjustmentPurple": "crs:LuminanceAdjustmentPurple",
    "LuminanceAdjustmentMagenta": "crs:LuminanceAdjustmentMagenta",
    
    # Color Grading
    "ColorGradeBlending": "crs:ColorGradeBlending",
    "ColorGradeGlobalHue": "crs:ColorGradeGlobalHue",
    "ColorGradeGlobalSat": "crs:ColorGradeGlobalSat",
    "ColorGradeGlobalLum": "crs:ColorGradeGlobalLum",
    "ColorGradeShadowHue": "crs:ColorGradeShadowHue",
    "ColorGradeShadowSat": "crs:ColorGradeShadowSat",
    "ColorGradeShadowLum": "crs:ColorGradeShadowLum",
    "ColorGradeMidtoneHue": "crs:ColorGradeMidtoneHue",
    "ColorGradeMidtoneSat": "crs:ColorGradeMidtoneSat",
    "ColorGradeMidtoneLum": "crs:ColorGradeMidtoneLum",
    "ColorGradeHighlightHue": "crs:ColorGradeHighlightHue",
    "ColorGradeHighlightSat": "crs:ColorGradeHighlightSat",
    "ColorGradeHighlightLum": "crs:ColorGradeHighlightLum",
    
    # Calibration
    "ShadowTint": "crs:ShadowTint",
    "RedHue": "crs:RedHue",
    "RedSaturation": "crs:RedSaturation",
    "GreenHue": "crs:GreenHue",
    "GreenSaturation": "crs:GreenSaturation",
    "BlueHue": "crs:BlueHue",
    "BlueSaturation": "crs:BlueSaturation",
}

# Parameters that are incremental (delta from As Shot) - these need special handling
INCREMENTAL_PARAMS = {
    "IncrementalTemperature": "crs:Temperature",
    "IncrementalTint": "crs:Tint",
}


def format_xmp_value(name: str, value) -> str:
    """Format a value for XMP attribute (always string)."""
    if name == "Exposure2012":
        # Exposure needs explicit sign
        if value >= 0:
            return f"+{value:.2f}"
        return f"{value:.2f}"
    elif isinstance(value, float):
        # Other floats - check if it's effectively an integer
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def get_as_shot_values(image_path: str) -> dict:
    """
    Extract As Shot Temperature and Tint from a RAW file using exiftool.
    
    Returns dict with 'Temperature' and 'Tint' keys, or empty dict if unavailable.
    """
    import subprocess
    import json
    
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-ColorTemperature", "-Tint", image_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                return {
                    "Temperature": data[0].get("ColorTemperature"),
                    "Tint": data[0].get("Tint"),
                }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return {}


def edits_to_xmp(
    edits: dict,
    image_path: Optional[str] = None,
    include_incremental: bool = True
) -> str:
    """
    Convert predicted edits dictionary to XMP file content.
    
    Args:
        edits: Dictionary of parameter names to values
        image_path: Optional path to source image (used to get As Shot values
                    for Temperature/Tint if include_incremental is True)
        include_incremental: If True, attempts to convert IncrementalTemperature/Tint
                            to absolute values using As Shot from the image
    
    Returns:
        XMP file content as string
    """
    attributes = []
    
    # Handle standard parameters
    for param_name, xmp_attr in PARAM_TO_XMP.items():
        if param_name in edits:
            value = format_xmp_value(param_name, edits[param_name])
            attributes.append(f'    {xmp_attr}="{value}"')
    
    # Handle incremental Temperature/Tint
    if include_incremental and image_path:
        as_shot = get_as_shot_values(image_path)
        
        if "IncrementalTemperature" in edits and as_shot.get("Temperature") is not None:
            try:
                base_temp = float(as_shot["Temperature"])
                delta = float(edits["IncrementalTemperature"])
                absolute_temp = int(base_temp + delta)
                attributes.append(f'    crs:Temperature="{absolute_temp}"')
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
        
        if "IncrementalTint" in edits and as_shot.get("Tint") is not None:
            try:
                base_tint = float(as_shot["Tint"])
                delta = float(edits["IncrementalTint"])
                absolute_tint = int(base_tint + delta)
                attributes.append(f'    crs:Tint="{absolute_tint}"')
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
    
    # Add process version (required for Lightroom to apply settings)
    attributes.append('    crs:ProcessVersion="15.4"')
    attributes.append('    crs:Version="16.1"')
    
    settings_str = "\n".join(attributes)
    return XMP_TEMPLATE.format(settings_attributes=settings_str)


def write_xmp_sidecar(
    edits: dict,
    image_path: str,
    output_path: Optional[str] = None,
    backup_existing: bool = True
) -> str:
    """
    Write predicted edits to an XMP sidecar file.
    
    Args:
        edits: Dictionary of parameter names to values
        image_path: Path to the source image
        output_path: Optional custom output path. If None, creates sidecar next to image.
        backup_existing: If True, backs up existing XMP file before overwriting
        
    Returns:
        Path to the written XMP file
    """
    image_path = Path(image_path)
    
    if output_path:
        xmp_path = Path(output_path)
    else:
        # Create sidecar path: same name as image but with .xmp extension
        xmp_path = image_path.with_suffix(".xmp")
    
    # Backup existing XMP if present
    if backup_existing and xmp_path.exists():
        backup_path = xmp_path.with_suffix(f".xmp.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        xmp_path.rename(backup_path)
        print(f"Backed up existing XMP to: {backup_path}", file=sys.stderr)
    
    # Generate XMP content
    xmp_content = edits_to_xmp(edits, str(image_path), include_incremental=True)
    
    # Write file
    with open(xmp_path, "w", encoding="utf-8") as f:
        f.write(xmp_content)
    
    return str(xmp_path)


def apply_edits(
    image_path: str,
    model_path: str = "lightroom_model.pt",
    model_paths: Optional[List[str]] = None,
    camera_model: Optional[str] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False
) -> dict:
    """
    Predict edits for an image and write them to an XMP sidecar file.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model checkpoint
        model_paths: Optional list of model paths for ensemble
        camera_model: Camera model name for prediction
        output_path: Optional custom XMP output path
        dry_run: If True, only predicts and prints without writing XMP
        
    Returns:
        Dictionary of predicted edits
    """
    # Predict edits
    edits = predict_edits(
        image_path=image_path,
        model_path=model_path,
        model_paths=model_paths,
        camera_model=camera_model
    )
    
    if dry_run:
        print("\n[DRY RUN] Would write the following edits:\n")
        print(format_edits(edits, image_path))
        print("\nXMP content that would be written:")
        print("-" * 50)
        print(edits_to_xmp(edits, image_path))
        return edits
    
    # Write XMP sidecar
    xmp_path = write_xmp_sidecar(edits, image_path, output_path)
    
    print(f"\n✓ XMP sidecar written to: {xmp_path}")
    print("\nTo apply in Lightroom:")
    print("  1. Select the image in Lightroom")
    print("  2. Go to Photo → Read Metadata from Files (Ctrl/Cmd+Shift+R)")
    print("\nPredicted edits:")
    print(format_edits(edits, image_path))
    
    return edits


def batch_apply_edits(
    image_paths: List[str],
    model_path: str = "lightroom_model.pt",
    model_paths: Optional[List[str]] = None,
    camera_model: Optional[str] = None,
    dry_run: bool = False
) -> List[dict]:
    """
    Apply predicted edits to multiple images.
    
    Args:
        image_paths: List of image paths
        model_path: Path to the trained model checkpoint
        model_paths: Optional list of model paths for ensemble
        camera_model: Camera model name for prediction
        dry_run: If True, only predicts without writing XMP
        
    Returns:
        List of edit dictionaries
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"Processing [{i+1}/{len(image_paths)}]: {Path(image_path).name}")
        print('='*60)
        
        try:
            edits = apply_edits(
                image_path=image_path,
                model_path=model_path,
                model_paths=model_paths,
                camera_model=camera_model,
                dry_run=dry_run
            )
            results.append({"path": image_path, "edits": edits, "success": True})
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            results.append({"path": image_path, "error": str(e), "success": False})
    
    # Summary
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {success_count}/{len(image_paths)} images processed successfully")
    if not dry_run:
        print("\nTo apply all in Lightroom:")
        print("  1. Select all processed images")
        print("  2. Go to Photo → Read Metadata from Files (Ctrl/Cmd+Shift+R)")
    print('='*60)
    
    return results


def batch_apply_edits_fast(
    image_paths: List[str],
    model_path: str = "lightroom_model.pt",
    model_paths: Optional[List[str]] = None,
    camera_model: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 32,
    num_workers: int = 8,
) -> List[dict]:
    """
    Apply predicted edits to multiple images with optimized batch processing.
    
    This is MUCH faster than batch_apply_edits() because it:
    1. Loads model(s) once at the start
    2. Uses parallel image loading (RAW decoding in parallel)
    3. Batches DINOv2 feature extraction on GPU
    4. Runs all predictions in a single batch
    
    Args:
        image_paths: List of image paths
        model_path: Path to the trained model checkpoint
        model_paths: Optional list of model paths for ensemble
        camera_model: Camera model name for prediction
        dry_run: If True, only predicts without writing XMP
        batch_size: Batch size for feature extraction (adjust based on GPU memory)
        num_workers: Number of parallel image loading workers
        
    Returns:
        List of result dictionaries with 'path', 'edits', 'success' keys
    """
    import time
    start_time = time.time()
    
    if not image_paths:
        return []
    
    n_images = len(image_paths)
    print(f"\n{'='*60}")
    print(f"FAST BATCH MODE: Processing {n_images} images")
    print('='*60)
    
    # Step 1: Load model(s) once
    print(f"\n[1/4] Loading model(s)...", file=sys.stderr)
    if model_paths:
        loaded = load_models(model_paths)
        print(f"  Loaded {len(model_paths)} models for ensemble", file=sys.stderr)
    else:
        loaded = [load_model(model_path)]
        print(f"  Loaded single model from {model_path}", file=sys.stderr)
    
    # Get checkpoint info from first model
    base_checkpoint = loaded[0][1]
    base_target_names = base_checkpoint["target_names"]
    base_circular_info = base_checkpoint.get("circular_info", {})
    base_camera_list = base_checkpoint.get("camera_list")
    base_has_camera = base_camera_list is not None
    
    # Validate ensemble consistency
    for _, ckpt in loaded[1:]:
        if ckpt["target_names"] != base_target_names:
            raise ValueError("Ensemble models have different target_names")
        if ckpt.get("circular_info", {}) != base_circular_info:
            raise ValueError("Ensemble models have different circular_info")
    
    # Build camera one-hot once
    camera_one_hot = None
    if base_has_camera:
        camera_one_hot = encode_camera_model(camera_model, base_camera_list)
        if camera_model and camera_model in base_camera_list:
            print(f"  Using camera: {camera_model}", file=sys.stderr)
        elif camera_model:
            print(f"  Unknown camera '{camera_model}', using fallback", file=sys.stderr)
    
    # Step 2: Extract features in batch (parallel loading + batched DINOv2)
    print(f"\n[2/4] Extracting features (parallel loading, batched DINOv2)...", file=sys.stderr)
    feature_start = time.time()
    all_features = extract_batch(
        image_paths,
        verbose=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    feature_time = time.time() - feature_start
    print(f"  Feature extraction: {feature_time:.1f}s ({len(all_features)/feature_time:.1f} img/s)", file=sys.stderr)
    
    # Build mapping from path to features
    path_to_features = {f.image_path: f for f in all_features}
    
    # Step 3: Batch prediction
    print(f"\n[3/4] Running predictions...", file=sys.stderr)
    predict_start = time.time()
    
    # Build feature matrix for all images
    successful_paths = []
    feature_rows = []
    for path in image_paths:
        if path in path_to_features:
            features = path_to_features[path]
            row = features.combined.reshape(1, -1)
            if camera_one_hot is not None:
                row = np.concatenate([row, camera_one_hot.reshape(1, -1)], axis=1)
            feature_rows.append(row)
            successful_paths.append(path)
    
    if not feature_rows:
        print("No images successfully processed!", file=sys.stderr)
        return []
    
    features_matrix = np.vstack(feature_rows)
    
    # Run predictions through each model and average
    all_preds = []
    for model, checkpoint in loaded:
        pred = predict(
            model=model,
            features=features_matrix,
            feature_mean=checkpoint['feature_mean'],
            feature_std=checkpoint['feature_std'],
            target_mean=checkpoint.get('target_mean'),
            target_std=checkpoint.get('target_std'),
            camera_features=camera_one_hot is not None,
        )
        all_preds.append(pred)
    
    # Average predictions across ensemble
    predictions_mean = np.mean(np.stack(all_preds, axis=0), axis=0)
    predict_time = time.time() - predict_start
    print(f"  Prediction: {predict_time:.1f}s", file=sys.stderr)
    
    # Step 4: Write XMP files
    print(f"\n[4/4] Writing XMP sidecar files...", file=sys.stderr)
    write_start = time.time()
    
    results = []
    for i, (path, pred_row) in enumerate(zip(successful_paths, predictions_mean)):
        try:
            # Build result dictionary
            raw_result = {name: value for name, value in zip(base_target_names, pred_row)}
            
            # Reconstruct circular hue values
            if base_circular_info:
                raw_result = reconstruct_circular_targets(raw_result, base_circular_info)
            
            # Apply clamping
            edits = {name: clamp_to_range(name, value) for name, value in raw_result.items()}
            
            if dry_run:
                results.append({"path": path, "edits": edits, "success": True})
            else:
                xmp_path = write_xmp_sidecar(edits, path)
                results.append({"path": path, "edits": edits, "xmp_path": xmp_path, "success": True})
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(successful_paths):
                print(f"  Written {i+1}/{len(successful_paths)} XMP files", file=sys.stderr)
                
        except Exception as e:
            print(f"  Error processing {path}: {e}", file=sys.stderr)
            results.append({"path": path, "error": str(e), "success": False})
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    # Summary
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(image_paths) - success_count
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {success_count}/{n_images} images processed")
    if failed_count > 0:
        print(f"  ({failed_count} failed)")
    print(f"\nTiming breakdown:")
    print(f"  Feature extraction: {feature_time:6.1f}s ({len(all_features)/max(feature_time,0.1):.1f} img/s)")
    print(f"  Prediction:         {predict_time:6.1f}s")
    print(f"  XMP writing:        {write_time:6.1f}s")
    print(f"  ─────────────────────────")
    print(f"  Total:              {total_time:6.1f}s ({len(successful_paths)/max(total_time,0.1):.1f} img/s)")
    
    if not dry_run:
        print("\nTo apply all in Lightroom:")
        print("  1. Select all processed images")
        print("  2. Go to Photo → Read Metadata from Files (Ctrl/Cmd+Shift+R)")
    print('='*60)
    
    return results


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - Modify these values directly
    # =========================================================================
    
    # Option 1: Single image path
    # IMAGE_INPUT = "/Users/eriksreinfelds/Pictures/2025/2025-12-11/DSC08353.ARW"
    
    # Option 2: Multiple image paths (list)
    # IMAGE_INPUT = [
    #     "/Users/eriksreinfelds/Pictures/2025/2025-12-11/DSC08353.ARW",
    #     "/Users/eriksreinfelds/Pictures/2025/2025-12-11/DSC08354.ARW",
    # ]
    
    # Option 3: Folder path (will process all RAW files in folder)
    IMAGE_INPUT = "/Users/eriksreinfelds/Pictures/2025/2025-12-08/"
    
    # Camera model (optional, set to None if not needed)
    CAMERA_MODEL = "ILCE-6700" # e.g., "ILCE-6700", "X-H2S", "Canon EOS 600D"
    
    # Model weights - Option A: Single model
    MODEL_PATH = "model_weights/lightroom_model.pt"
    MODEL_PATHS = None  # Set to None for single model
    
    # Model weights - Option B: Ensemble (uncomment to use)
    # MODEL_PATH = "lightroom_model.pt"  # Fallback, not used when MODEL_PATHS is set
    # MODEL_PATHS = [
    #     "model_weights/lightroom_model_fold1.pt",
    #     "model_weights/lightroom_model_fold2.pt",
    #     "model_weights/lightroom_model_fold3.pt",
    # ]
    
    # Dry run mode (True = preview only, False = write XMP files)
    DRY_RUN = False
    
    # Use fast batch mode (recommended for multiple images)
    # Fast mode loads model once and uses parallel feature extraction
    USE_FAST_BATCH = True
    
    # Batch processing settings (for fast mode)
    BATCH_SIZE = 32      # DINOv2 batch size (reduce if running out of GPU memory)
    NUM_WORKERS = 8      # Parallel image loading threads
    
    # Supported RAW file extensions
    RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng", ".orf", ".rw2"}
    
    # =========================================================================
    # EXECUTION - Don't modify below this line
    # =========================================================================
    
    from glob import glob
    
    # Determine image paths based on input type
    if isinstance(IMAGE_INPUT, str):
        input_path = Path(IMAGE_INPUT)
        
        if input_path.is_dir():
            # Folder path - collect all RAW files
            image_paths = []
            for ext in RAW_EXTENSIONS:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))
            image_paths = sorted([str(p) for p in image_paths])
            print(f"Found {len(image_paths)} RAW files in folder: {IMAGE_INPUT}")
        else:
            # Single image path
            image_paths = [str(input_path)]
    elif isinstance(IMAGE_INPUT, list):
        # Multiple image paths
        image_paths = IMAGE_INPUT
    else:
        raise ValueError(f"IMAGE_INPUT must be a string or list, got {type(IMAGE_INPUT)}")
    
    if not image_paths:
        print("No images found to process!")
        sys.exit(1)
    
    # Process images
    if len(image_paths) == 1:
        apply_edits(
            image_path=image_paths[0],
            model_path=MODEL_PATH,
            model_paths=MODEL_PATHS,
            camera_model=CAMERA_MODEL,
            dry_run=DRY_RUN
        )
    elif USE_FAST_BATCH:
        # Fast batch mode: loads model once, parallel feature extraction
        batch_apply_edits_fast(
            image_paths=image_paths,
            model_path=MODEL_PATH,
            model_paths=MODEL_PATHS,
            camera_model=CAMERA_MODEL,
            dry_run=DRY_RUN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    else:
        # Sequential mode (slower, but useful for debugging)
        batch_apply_edits(
            image_paths=image_paths,
            model_path=MODEL_PATH,
            model_paths=MODEL_PATHS,
            camera_model=CAMERA_MODEL,
            dry_run=DRY_RUN
        )

