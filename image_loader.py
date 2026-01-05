#!/usr/bin/env python3
"""
Image loading module for the Lightroom auto-editor pipeline.
Handles RAW formats (CR2, ARW, NEF, DNG, etc.) and standard formats (JPEG, PNG, etc.).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import rawpy
from PIL import Image

# RAW file extensions supported by rawpy
RAW_EXTENSIONS = {".cr2", ".cr3", ".arw", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def load_image(image_path: str, target_size: Tuple[int, int] | None = None) -> Image.Image:
    """
    Load an image from file, handling RAW formats.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) to resize to
        
    Returns:
        PIL Image in RGB mode
    """
    ext = Path(image_path).suffix.lower()
    
    if ext in RAW_EXTENSIONS:
        image = _load_raw(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
    
    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image


def _load_raw(image_path: str) -> Image.Image:
    """
    Load a RAW image file using rawpy.
    
    Args:
        image_path: Path to RAW file
        
    Returns:
        PIL Image in RGB mode
    """
    with rawpy.imread(image_path) as raw:
        # postprocess() converts RAW to RGB numpy array
        # use_camera_wb: use camera white balance
        # half_size: faster processing, still good for feature extraction
        # no_auto_bright: preserve original brightness for consistent features
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=True,
            no_auto_bright=True,
            output_bps=8
        )
    return Image.fromarray(rgb)


def load_image_array(image_path: str, target_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image as a numpy array.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) to resize to
        
    Returns:
        numpy array of shape (H, W, 3) with uint8 values [0-255]
    """
    image = load_image(image_path, target_size)
    return np.array(image)


def is_raw_file(image_path: str) -> bool:
    """Check if a file is a RAW image format."""
    return Path(image_path).suffix.lower() in RAW_EXTENSIONS


def get_image_info(image_path: str) -> dict:
    """
    Get basic info about an image without fully loading it.
    
    Returns:
        dict with 'path', 'is_raw', 'extension'
    """
    path = Path(image_path)
    return {
        "path": str(path),
        "filename": path.name,
        "is_raw": is_raw_file(image_path),
        "extension": path.suffix.lower()
    }

