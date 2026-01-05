#!/usr/bin/env python3
"""
DINOv2 feature extraction for the Lightroom auto-editor pipeline.
Optimized for Apple Silicon (M4 Pro) using MPS backend.
"""

import sys
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from image_loader import load_image

# Module-level cache for the model and transform (avoids reloading)
_model = None
_device = None
_transform = None


def get_device():
    """Get the best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model():
    """Get the DINOv2 model, loading it if necessary (cached)."""
    global _model, _device
    
    if _model is None:
        _device = get_device()
        print(f"Loading dinov2_vits14 on {_device}...", file=sys.stderr)
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _model.eval()
        _model = _model.to(_device)
    
    return _model, _device


def get_transform():
    """Get the preprocessing transform for DINOv2 (cached)."""
    global _transform
    
    if _transform is None:
        _transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return _transform


def extract_features(image_path: str) -> np.ndarray:
    """
    Extract DINOv2 features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of shape (384,)
    """
    image = load_image(image_path)
    return extract_features_from_pil(image)


def extract_features_from_pil(image: Image.Image) -> np.ndarray:
    """
    Extract DINOv2 features from a PIL Image (already loaded).
    
    Args:
        image: PIL Image in RGB mode
        
    Returns:
        numpy array of shape (384,)
    """
    model, device = get_model()
    transform = get_transform()
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.cpu().squeeze(0).numpy()


def extract_features_batch(images: List[Image.Image], batch_size: int = 16) -> List[np.ndarray]:
    """
    Extract DINOv2 features from multiple PIL Images in batches.
    Much faster than processing one at a time due to GPU parallelism.
    
    Args:
        images: List of PIL Images in RGB mode
        batch_size: Number of images to process at once (adjust based on GPU memory)
        
    Returns:
        List of numpy arrays, each of shape (384,)
    """
    if not images:
        return []
    
    model, device = get_model()
    transform = get_transform()
    
    all_features = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Transform all images in batch and stack
        batch_tensors = torch.stack([transform(img) for img in batch_images]).to(device)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            batch_features = model(batch_tensors)
        
        # Convert to numpy and add to results
        batch_features_np = batch_features.cpu().numpy()
        all_features.extend([batch_features_np[j] for j in range(len(batch_images))])
    
    return all_features


def get_feature_count() -> int:
    """Get the number of DINOv2 features (384 for vits14)."""
    return 384


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    features = extract_features(image_path)
    
    print(f"Extracted {len(features)} DINOv2 features", file=sys.stderr)
    print(f"Stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}", file=sys.stderr)
    
    # Print features as comma-separated values
    print(",".join(f"{x:.6f}" for x in features))
