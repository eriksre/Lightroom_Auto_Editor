#!/usr/bin/env python3
"""
Feature extraction pipeline for the Lightroom auto-editor.
Combines DINOv2 features with traditional image features.
Optimized for speed with batch processing and parallel image loading.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image

import dinov2_features
import traditional_features
from image_loader import load_image


# Default paths
DEFAULT_CATALOGUE_DIR = Path("lightroom catalogue")
DEFAULT_PROCESSED_PARQUET = Path("processed_images.parquet")
DEFAULT_OUTPUT_PARQUET = Path("image_features.parquet")

# Batch processing settings
DINO_BATCH_SIZE = 32  # Adjust based on GPU memory
IMAGE_LOAD_WORKERS = 8  # Parallel image loading threads


class ImageFeatures(NamedTuple):
    """Container for all extracted features from an image."""
    image_path: str
    dinov2: np.ndarray          # 384 dims
    traditional: np.ndarray     # 190 dims (histogram, color, brightness, temp, HSL, edge, LAB)
    combined: np.ndarray        # 574 dims total


def extract_all_features_from_pil(image: Image.Image, image_path: str) -> ImageFeatures:
    """
    Extract all features from an already-loaded PIL Image.
    Avoids loading the image twice (once for DINOv2, once for traditional).
    
    Args:
        image: PIL Image in RGB mode
        image_path: Original path (for metadata)
        
    Returns:
        ImageFeatures named tuple with all feature arrays
    """
    # Extract DINOv2 features (384 dims)
    dino_feats = dinov2_features.extract_features_from_pil(image)
    
    # Extract traditional features (136 dims)
    trad_feats = traditional_features.extract_all_features_from_pil(image)
    
    # Combine all features
    combined = np.concatenate([dino_feats, trad_feats])
    
    return ImageFeatures(
        image_path=image_path,
        dinov2=dino_feats,
        traditional=trad_feats,
        combined=combined
    )
    

def extract_all_features(image_path: str, verbose: bool = False) -> ImageFeatures:
    """
    Extract all features from an image (DINOv2 + traditional).
    
    Args:
        image_path: Path to the image file
        verbose: Print progress to stderr
        
    Returns:
        ImageFeatures named tuple with all feature arrays
    """
    if verbose:
        print(f"Processing: {image_path}", file=sys.stderr)
    
    # Load image once, use for both extractors
    image = load_image(image_path)
    result = extract_all_features_from_pil(image, image_path)
    
    if verbose:
        print(f"  Total features: {len(result.combined)}", file=sys.stderr)
    
    return result


def _load_image_safe(path: str) -> tuple[str, Image.Image | None, str | None]:
    """Load an image, returning (path, image, error_msg)."""
    try:
        return (path, load_image(path), None)
    except Exception as e:
        return (path, None, str(e))


def extract_batch(
    image_paths: list[str],
    verbose: bool = False,
    batch_size: int = DINO_BATCH_SIZE,
    num_workers: int = IMAGE_LOAD_WORKERS,
) -> list[ImageFeatures]:
    """
    Extract features from multiple images with optimized batch processing.
    
    Uses parallel image loading and batched DINOv2 inference for speed.
    
    Args:
        image_paths: List of paths to image files
        verbose: Print progress to stderr
        batch_size: Batch size for DINOv2 inference
        num_workers: Number of parallel image loading workers
        
    Returns:
        List of ImageFeatures for each successfully processed image
    """
    if not image_paths:
        return []
    
    total = len(image_paths)
    results = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = image_paths[batch_start:batch_end]
        
        if verbose:
            print(f"[Batch {batch_start//batch_size + 1}] Loading images {batch_start+1}-{batch_end}/{total}...", file=sys.stderr)
        
        # Load images in parallel (RAW decoding is CPU-intensive)
        loaded_images = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_load_image_safe, p): p for p in batch_paths}
            for future in as_completed(futures):
                path, image, error = future.result()
                if error:
                    if verbose:
                        print(f"  Error loading {path}: {error}", file=sys.stderr)
                else:
                    loaded_images.append((path, image))
        
        if not loaded_images:
            continue
        
        paths, images = zip(*loaded_images)
        
        if verbose:
            print(f"  Extracting DINOv2 features (batch of {len(images)})...", file=sys.stderr)
        
        # Batch DINOv2 inference (major speedup)
        dino_features = dinov2_features.extract_features_batch(list(images), batch_size=batch_size)
        
        if verbose:
            print(f"  Extracting traditional features...", file=sys.stderr)
        
        # Traditional features (CPU-bound, process in parallel)
        trad_features = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            trad_futures = [executor.submit(traditional_features.extract_all_features_from_pil, img) for img in images]
            for future in trad_futures:
                trad_features.append(future.result())
        
        # Combine results
        for path, dino, trad in zip(paths, dino_features, trad_features):
            combined = np.concatenate([dino, trad])
            results.append(ImageFeatures(
                image_path=path,
                dinov2=dino,
                traditional=trad,
                combined=combined
            ))
        
        if verbose:
            print(f"  Processed {len(results)}/{total} images", file=sys.stderr)
    
    return results


def get_feature_names() -> list[str]:
    """
    Get names for all features in the combined vector.
    
    Returns:
        List of feature names in order
    """
    names = []
    
    # DINOv2 features (384)
    for i in range(384):
        names.append(f"dinov2_{i}")
    
    # Traditional features
    names.extend(traditional_features.get_feature_names())
    
    return names


def get_total_feature_count() -> int:
    """Get total number of features in combined vector."""
    return dinov2_features.get_feature_count() + traditional_features.get_feature_count()


def features_to_dict(features: ImageFeatures) -> dict:
    """Convert ImageFeatures to a dictionary (useful for DataFrames)."""
    names = get_feature_names()
    result = {"image_path": features.image_path}
    
    for name, value in zip(names, features.combined):
        result[name] = float(value)
    
    return result


def process_from_parquet(
    n_images: int | None = None,
    input_parquet: Path = DEFAULT_PROCESSED_PARQUET,
    output_parquet: Path = DEFAULT_OUTPUT_PARQUET,
    catalogue_dir: Path = DEFAULT_CATALOGUE_DIR,
    verbose: bool = True,
    batch_size: int = DINO_BATCH_SIZE,
    num_workers: int = IMAGE_LOAD_WORKERS,
) -> pd.DataFrame:
    """
    Extract features for images listed in a parquet file.
    Uses optimized batch processing for speed.
    
    Args:
        n_images: Number of images to process (None = all)
        input_parquet: Path to parquet file with image metadata (needs 'file_path' column)
        output_parquet: Path to save extracted features
        catalogue_dir: Base directory where images are stored
        verbose: Print progress
        batch_size: Batch size for DINOv2 inference
        num_workers: Number of parallel image loading workers
        
    Returns:
        DataFrame with extracted features
    """
    # Load image metadata
    df = pd.read_parquet(input_parquet)
    
    required_cols = ['file_path', 'image_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Parquet file must have columns {required_cols}. Missing: {missing_cols}. Found: {df.columns.tolist()}")
    
    # Limit number of images if specified
    if n_images is not None:
        df = df.head(n_images)
    
    if verbose:
        print(f"Processing {len(df)} images from {input_parquet}", file=sys.stderr)
    
    # Build list of valid paths and their metadata
    valid_paths = []
    path_to_row = {}
    
    for idx, row in df.iterrows():
        file_path = row['file_path']
        full_path = catalogue_dir / file_path
        
        if not full_path.exists():
            if verbose:
                print(f"  Skipping (not found): {full_path}", file=sys.stderr)
            continue
        
        full_path_str = str(full_path)
        valid_paths.append(full_path_str)
        path_to_row[full_path_str] = row
    
    if verbose:
        print(f"Found {len(valid_paths)} valid images to process", file=sys.stderr)
    
    # Extract features using optimized batch processing
    all_features = extract_batch(
        valid_paths,
        verbose=verbose,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Convert to DataFrame rows
    results = []
    for features in all_features:
        row = path_to_row[features.image_path]
        result_dict = features_to_dict(features)
        # Keep original file_path (relative) and image_id for joining with edit data
        result_dict['file_path'] = row['file_path']
        result_dict['image_id'] = row['image_id']
        results.append(result_dict)
    
    # Create DataFrame and save
    features_df = pd.DataFrame(results)
    features_df.to_parquet(output_parquet, index=False)
    features_df.to_csv("image_features.csv", index=False)
    
    if verbose:
        print(f"\nSaved {len(features_df)} feature vectors to {output_parquet}", file=sys.stderr)
    
    return features_df


if __name__ == "__main__":
    # Configuration
    n_images = None  # Set to None to process all
    input_parquet = "processed_images.parquet"
    output_parquet = "image_features.parquet"
    catalogue_dir = "lightroom catalogue"
    
    process_from_parquet(
        n_images=n_images,
        input_parquet=Path(input_parquet),
        output_parquet=Path(output_parquet),
        catalogue_dir=Path(catalogue_dir),
        verbose=True,
    )

