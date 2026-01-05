#!/usr/bin/env python3
"""
Traditional feature extraction for the Lightroom auto-editor pipeline.
Extracts histogram, color statistics, and other handcrafted features.
Optimized for speed with vectorized operations.
"""

from typing import List

import numpy as np
from PIL import Image

from image_loader import load_image, load_image_array


def extract_all_features(image_path: str) -> np.ndarray:
    """
    Extract all traditional features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        1D numpy array of all features concatenated
    """
    image = load_image(image_path)
    return extract_all_features_from_pil(image)


def extract_all_features_from_pil(image: Image.Image) -> np.ndarray:
    """
    Extract all traditional features from a PIL Image (already loaded).
    
    Args:
        image: PIL Image in RGB mode
        
    Returns:
        1D numpy array of all features concatenated
    """
    img_array = np.asarray(image, dtype=np.float32)
    return extract_all_features_from_array(img_array)


def extract_all_features_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Extract all traditional features from a numpy array.
    Optimized to compute shared values once.
    
    Args:
        img_array: numpy array of shape (H, W, 3)
        
    Returns:
        1D numpy array of all features concatenated
    """
    # Pre-allocate output array for speed
    # 96 (histogram) + 18 (color stats) + 10 (brightness) + 12 (color dist) + 
    # 14 (temperature) + 24 (HSL color) + 10 (edge/texture) + 6 (LAB) = 190
    features = np.empty(190, dtype=np.float32)
    
    # Normalize once for features that need it
    img_norm = img_array / 255.0
    
    # Extract all features into pre-allocated array
    idx = 0
    
    # Histogram features (96 values)
    idx = _extract_histogram_features_fast(img_array, features, idx)
    
    # Color statistics (18 values)
    idx = _extract_color_statistics_fast(img_array, features, idx)
    
    # Brightness features (10 values)
    idx = _extract_brightness_features_fast(img_array, features, idx)
    
    # Color distribution (12 values)
    idx = _extract_color_distribution_fast(img_norm, features, idx)
    
    # Temperature/white balance features (14 values)
    idx = _extract_temperature_features_fast(img_norm, features, idx)
    
    # HSL per-color analysis (24 values) - for HSL sliders
    idx = _extract_hsl_color_features_fast(img_norm, features, idx)
    
    # Edge/texture features (10 values) - for Texture, Clarity, Dehaze
    idx = _extract_edge_texture_features_fast(img_norm, features, idx)
    
    # LAB color space features (6 values) - perceptual color
    idx = _extract_lab_features_fast(img_norm, features, idx)
    
    return features


def _extract_histogram_features_fast(img_array: np.ndarray, out: np.ndarray, start_idx: int, bins: int = 32) -> int:
    """Extract histogram features directly into output array."""
    idx = start_idx
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum()
        out[idx:idx + bins] = hist
        idx += bins
    return idx


def _extract_color_statistics_fast(img_array: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """Extract color statistics directly into output array."""
    idx = start_idx
    for channel in range(3):
        ch = img_array[:, :, channel].ravel()  # ravel is faster than flatten
        
        mean = np.mean(ch)
        std = np.std(ch)
        
        out[idx] = mean / 255
        out[idx + 1] = std / 255
        out[idx + 2] = np.median(ch) / 255
        out[idx + 3] = ch.min() / 255
        out[idx + 4] = ch.max() / 255
        out[idx + 5] = np.mean(((ch - mean) / (std + 1e-6)) ** 3) if std > 0 else 0.0
        idx += 6
    return idx


def _extract_brightness_features_fast(img_array: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """Extract brightness features directly into output array."""
    # Compute grayscale using vectorized operation
    gray = (0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]).ravel()
    
    mean_brightness = np.mean(gray) / 255
    std_brightness = np.std(gray) / 255
    
    # Compute all percentiles in one call (faster)
    percentiles = np.percentile(gray, [5, 25, 50, 75, 95]) / 255
    
    idx = start_idx
    out[idx] = mean_brightness
    out[idx + 1] = std_brightness
    out[idx + 2:idx + 7] = percentiles
    out[idx + 7] = percentiles[4] - percentiles[0]  # dynamic_range
    out[idx + 8] = percentiles[3] - percentiles[1]  # contrast
    out[idx + 9] = np.mean((gray < 10) | (gray > 245))  # clipping
    
    return idx + 10


def _extract_color_distribution_fast(img_norm: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """Extract color distribution features directly into output array."""
    r = img_norm[:, :, 0].ravel()
    g = img_norm[:, :, 1].ravel()
    b = img_norm[:, :, 2].ravel()
    
    # Color ratios
    total = r + g + b + 1e-6
    
    idx = start_idx
    out[idx] = np.mean(r / total)
    out[idx + 1] = np.mean(g / total)
    out[idx + 2] = np.mean(b / total)
    
    # Channel correlations - use faster covariance-based calculation
    # corrcoef is slow; compute directly
    def fast_corr(x, y):
        n = len(x)
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()
        if sx < 1e-8 or sy < 1e-8:
            return 0.0
        return np.dot(x - mx, y - my) / (n * sx * sy)
    
    out[idx + 3] = fast_corr(r, g)
    out[idx + 4] = fast_corr(r, b)
    out[idx + 5] = fast_corr(g, b)
    
    # Saturation
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = max_rgb - min_rgb
    
    out[idx + 6] = np.mean(saturation)
    out[idx + 7] = np.std(saturation)
    out[idx + 8] = np.mean(r - b)  # warmth
    out[idx + 9] = np.mean(g - 0.5 * (r + b))  # tint
    out[idx + 10] = np.mean(max_rgb)
    out[idx + 11] = np.mean(min_rgb)
    
    return idx + 12


def _extract_temperature_features_fast(img_norm: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """
    Extract temperature/white balance features directly into output array.
    These features are designed to help predict Lightroom's Temperature and Tint sliders.
    
    Key insight: Color temperature affects R-B balance, and varies by tonal range.
    Highlights often reveal the true light source color temperature.
    """
    r = img_norm[:, :, 0].ravel()
    g = img_norm[:, :, 1].ravel()
    b = img_norm[:, :, 2].ravel()
    
    # Compute grayscale for tonal segmentation
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Warmth = R - B (positive = warm, negative = cool)
    warmth = r - b
    
    # Tint = G - 0.5*(R+B) (positive = green, negative = magenta)
    tint = g - 0.5 * (r + b)
    
    idx = start_idx
    
    # Warmth distribution (not just mean - std and percentiles matter)
    out[idx] = np.mean(warmth)
    out[idx + 1] = np.std(warmth)
    warmth_percentiles = np.percentile(warmth, [10, 50, 90])
    out[idx + 2] = warmth_percentiles[0]  # p10
    out[idx + 3] = warmth_percentiles[1]  # p50 (median)
    out[idx + 4] = warmth_percentiles[2]  # p90
    out[idx + 5] = warmth_percentiles[2] - warmth_percentiles[0]  # warmth range
    
    # Tonal-segmented warmth (key for white balance detection)
    # Highlights often reveal true color temperature of light source
    highlight_mask = gray > 0.7
    midtone_mask = (gray >= 0.3) & (gray <= 0.7)
    shadow_mask = gray < 0.3
    
    out[idx + 6] = np.mean(warmth[highlight_mask]) if np.any(highlight_mask) else 0.0
    out[idx + 7] = np.mean(warmth[midtone_mask]) if np.any(midtone_mask) else 0.0
    out[idx + 8] = np.mean(warmth[shadow_mask]) if np.any(shadow_mask) else 0.0
    
    # Neutral area warmth (low saturation pixels should be neutral if WB is correct)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = max_rgb - min_rgb
    neutral_mask = saturation < 0.1  # Low saturation = likely neutral gray
    out[idx + 9] = np.mean(warmth[neutral_mask]) if np.any(neutral_mask) else 0.0
    
    # Tint features (green/magenta axis)
    out[idx + 10] = np.mean(tint)
    out[idx + 11] = np.std(tint)
    out[idx + 12] = np.mean(tint[highlight_mask]) if np.any(highlight_mask) else 0.0
    out[idx + 13] = np.mean(tint[neutral_mask]) if np.any(neutral_mask) else 0.0
    
    return idx + 14


def _rgb_to_hsl_vectorized(img_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB image to HSL. Vectorized for speed.
    
    Args:
        img_norm: Normalized RGB array (0-1), shape (H, W, 3)
        
    Returns:
        H (0-360), S (0-1), L (0-1) as 1D arrays
    """
    r = img_norm[:, :, 0].ravel()
    g = img_norm[:, :, 1].ravel()
    b = img_norm[:, :, 2].ravel()
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    
    # Lightness
    L = (max_c + min_c) / 2
    
    # Saturation
    S = np.zeros_like(L)
    mask = delta > 0
    S[mask] = delta[mask] / (1 - np.abs(2 * L[mask] - 1) + 1e-8)
    S = np.clip(S, 0, 1)
    
    # Hue
    H = np.zeros_like(L)
    
    # Red is max
    mask_r = (max_c == r) & (delta > 0)
    H[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6)
    
    # Green is max
    mask_g = (max_c == g) & (delta > 0)
    H[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8) + 2)
    
    # Blue is max
    mask_b = (max_c == b) & (delta > 0)
    H[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8) + 4)
    
    H = H % 360
    
    return H, S, L


# Lightroom's 8 color hue ranges (in degrees)
# These match Lightroom's HSL panel color definitions
LIGHTROOM_HUE_RANGES = {
    'red':     [(0, 15), (345, 360)],      # Red wraps around 0/360
    'orange':  [(15, 45)],
    'yellow':  [(45, 75)],
    'green':   [(75, 165)],
    'aqua':    [(165, 195)],
    'blue':    [(195, 255)],
    'purple':  [(255, 285)],
    'magenta': [(285, 345)],
}


def _extract_hsl_color_features_fast(img_norm: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """
    Extract per-color HSL features for Lightroom's 8 color channels.
    These directly inform HueAdjustment, SaturationAdjustment, and LuminanceAdjustment sliders.
    
    For each color: pixel_ratio, avg_saturation, avg_luminance (24 features total)
    """
    H, S, L = _rgb_to_hsl_vectorized(img_norm)
    n_pixels = len(H)
    
    idx = start_idx
    
    for color_name in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
        hue_ranges = LIGHTROOM_HUE_RANGES[color_name]
        
        # Build mask for this color's hue range
        mask = np.zeros(n_pixels, dtype=bool)
        for (low, high) in hue_ranges:
            mask |= (H >= low) & (H < high)
        
        # Only consider pixels with some saturation (avoid grays)
        mask &= (S > 0.1)
        
        count = np.sum(mask)
        
        # Pixel ratio (how much of image is this color)
        out[idx] = count / n_pixels if n_pixels > 0 else 0.0
        
        # Average saturation of this color
        out[idx + 1] = np.mean(S[mask]) if count > 0 else 0.0
        
        # Average luminance of this color
        out[idx + 2] = np.mean(L[mask]) if count > 0 else 0.0
        
        idx += 3
    
    return idx


def _extract_edge_texture_features_fast(img_norm: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """
    Extract edge and texture features for Texture, Clarity, and Dehaze sliders.
    Uses gradient-based methods (no OpenCV dependency).
    """
    # Convert to grayscale
    gray = 0.299 * img_norm[:, :, 0] + 0.587 * img_norm[:, :, 1] + 0.114 * img_norm[:, :, 2]
    
    # Compute gradients (Sobel-like)
    # Horizontal gradient
    gx = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    
    # Vertical gradient
    gy = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    
    # Gradient magnitude
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    idx = start_idx
    
    # Edge density (proportion of strong edges)
    edge_threshold = 0.1
    out[idx] = np.mean(grad_mag > edge_threshold)
    
    # Edge strength statistics
    out[idx + 1] = np.mean(grad_mag)
    out[idx + 2] = np.std(grad_mag)
    out[idx + 3] = np.percentile(grad_mag, 90)
    
    # Local contrast (std in local patches - approximated by gradient variance)
    out[idx + 4] = np.var(grad_mag)
    
    # High frequency energy (sum of squared gradients)
    out[idx + 5] = np.mean(grad_mag ** 2)
    
    # Haze estimation using dark channel prior approximation
    # In hazy images, at least one color channel has low values in most patches
    # We use a simplified version: minimum of RGB channels
    dark_channel = np.minimum(np.minimum(img_norm[:, :, 0], img_norm[:, :, 1]), img_norm[:, :, 2])
    
    # Haze metrics
    out[idx + 6] = np.mean(dark_channel)  # Higher = more haze
    out[idx + 7] = np.percentile(dark_channel, 90)  # Upper bound of haze
    
    # Contrast in dark channel (low = uniform haze)
    out[idx + 8] = np.std(dark_channel)
    
    # Atmospheric light estimate (brightness of haziest region)
    # Top 0.1% brightest pixels in dark channel
    top_threshold = np.percentile(dark_channel, 99.9)
    bright_mask = dark_channel >= top_threshold
    if np.any(bright_mask):
        out[idx + 9] = np.mean(gray[bright_mask])
    else:
        out[idx + 9] = np.mean(gray)
    
    return idx + 10


def _rgb_to_lab_vectorized(img_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to LAB color space. Simplified D65 illuminant.
    
    Args:
        img_norm: Normalized RGB array (0-1), shape (H, W, 3)
        
    Returns:
        L (0-100), a (-128 to 128), b (-128 to 128) as 1D arrays
    """
    # Linearize sRGB
    rgb = img_norm.reshape(-1, 3)
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # RGB to XYZ (D65 illuminant)
    # Using sRGB to XYZ matrix
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = rgb_linear @ M.T
    
    # Normalize by D65 white point
    xyz[:, 0] /= 0.95047
    xyz[:, 1] /= 1.00000
    xyz[:, 2] /= 1.08883
    
    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    
    mask = xyz > epsilon
    f = np.where(mask, xyz ** (1/3), (kappa * xyz + 16) / 116)
    
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    
    return L, a, b


def _extract_lab_features_fast(img_norm: np.ndarray, out: np.ndarray, start_idx: int) -> int:
    """
    Extract LAB color space features.
    LAB better represents perceptual color differences than RGB.
    
    L*: Lightness (0-100)
    a*: Green (-) to Red (+)
    b*: Blue (-) to Yellow (+) -- directly related to temperature!
    """
    L, a, b = _rgb_to_lab_vectorized(img_norm)
    
    idx = start_idx
    
    # L channel (lightness) - normalize to 0-1
    out[idx] = np.mean(L) / 100
    out[idx + 1] = np.std(L) / 100
    
    # a channel (green-red) - normalize approximately
    out[idx + 2] = np.mean(a) / 128
    out[idx + 3] = np.std(a) / 128
    
    # b channel (blue-yellow) - normalize approximately
    # This is very relevant for temperature prediction!
    out[idx + 4] = np.mean(b) / 128
    out[idx + 5] = np.std(b) / 128
    
    return idx + 6


# Legacy functions - kept for backward compatibility but less optimized
# Use extract_all_features_from_array for best performance

def extract_histogram_features(img_array: np.ndarray, bins: int = 32) -> np.ndarray:
    """Extract histogram features for each RGB channel."""
    out = np.empty(bins * 3, dtype=np.float32)
    _extract_histogram_features_fast(img_array.astype(np.float32), out, 0, bins)
    return out


def extract_color_statistics(img_array: np.ndarray) -> np.ndarray:
    """Extract per-channel color statistics."""
    out = np.empty(18, dtype=np.float32)
    _extract_color_statistics_fast(img_array.astype(np.float32), out, 0)
    return out


def extract_brightness_features(img_array: np.ndarray) -> np.ndarray:
    """Extract luminance/brightness related features."""
    out = np.empty(10, dtype=np.float32)
    _extract_brightness_features_fast(img_array.astype(np.float32), out, 0)
    return out


def extract_color_distribution(img_array: np.ndarray) -> np.ndarray:
    """Extract color balance and distribution features."""
    out = np.empty(12, dtype=np.float32)
    _extract_color_distribution_fast(img_array.astype(np.float32) / 255.0, out, 0)
    return out


def get_feature_names() -> list[str]:
    """
    Get names for all features in the order they appear.
    Useful for debugging and analysis.
    """
    names = []
    
    # Histogram features (32 bins * 3 channels)
    for ch in ['R', 'G', 'B']:
        for i in range(32):
            names.append(f'hist_{ch}_{i}')
    
    # Color statistics (6 * 3 channels)
    for ch in ['R', 'G', 'B']:
        names.extend([f'{ch}_mean', f'{ch}_std', f'{ch}_median', f'{ch}_min', f'{ch}_max', f'{ch}_skew'])
    
    # Brightness features (10)
    names.extend([
        'brightness_mean', 'brightness_std',
        'brightness_p5', 'brightness_p25', 'brightness_p50', 'brightness_p75', 'brightness_p95',
        'dynamic_range', 'contrast', 'clipping'
    ])
    
    # Color distribution (12)
    names.extend([
        'r_ratio', 'g_ratio', 'b_ratio',
        'rg_corr', 'rb_corr', 'gb_corr',
        'saturation_mean', 'saturation_std',
        'warmth', 'tint',
        'max_rgb_mean', 'min_rgb_mean'
    ])
    
    # Temperature/white balance features (14)
    names.extend([
        'temp_warmth_mean', 'temp_warmth_std',
        'temp_warmth_p10', 'temp_warmth_p50', 'temp_warmth_p90',
        'temp_warmth_range',
        'temp_highlight_warmth', 'temp_midtone_warmth', 'temp_shadow_warmth',
        'temp_neutral_warmth',
        'temp_tint_mean', 'temp_tint_std',
        'temp_highlight_tint', 'temp_neutral_tint'
    ])
    
    # HSL per-color features (24) - for Lightroom's 8 color HSL sliders
    for color in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
        names.extend([
            f'hsl_{color}_ratio',
            f'hsl_{color}_saturation',
            f'hsl_{color}_luminance'
        ])
    
    # Edge/texture features (10) - for Texture, Clarity, Dehaze
    names.extend([
        'edge_density', 'edge_mean', 'edge_std', 'edge_p90',
        'local_contrast_var', 'high_freq_energy',
        'haze_dark_channel_mean', 'haze_dark_channel_p90',
        'haze_dark_channel_std', 'atmospheric_light'
    ])
    
    # LAB color space features (6)
    names.extend([
        'lab_L_mean', 'lab_L_std',
        'lab_a_mean', 'lab_a_std',
        'lab_b_mean', 'lab_b_std'
    ])
    
    return names


def get_feature_count() -> int:
    """Get the total number of traditional features."""
    # 96 (histogram) + 18 (color stats) + 10 (brightness) + 12 (color dist) + 
    # 14 (temperature) + 24 (HSL color) + 10 (edge/texture) + 6 (LAB) = 190
    return 190


if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) < 2:
    #     print(f"Usage: {sys.argv[0]} <image_path>", file=sys.stderr)
    #     sys.exit(1)
    
    #image_path = sys.argv[1]
    image_path = 'lightroom catalogue/2025/2025-07-10/DSC03144.ARW'
    features = extract_all_features(image_path)
    
    print(f"Extracted {len(features)} traditional features", file=sys.stderr)
    print(f"Feature names: {get_feature_names()[:10]}... (showing first 10)", file=sys.stderr)
    
    # Print features as comma-separated values
    print(",".join(f"{x:.6f}" for x in features))

