#!/usr/bin/env python3
"""
Proxy renderer for approximate Lightroom-style tonal adjustments.

This is intentionally lightweight and fast. It aims to be "good enough" for
perceptual search or loss definitions, not a pixel-accurate Lightroom clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


ImageArray = Union[np.ndarray, "torch.Tensor"]

SLIDER_DEFAULTS = {
    "Exposure2012": 0.0,   # stops
    "Contrast2012": 0.0,   # -100..100
    "Highlights2012": 0.0, # -100..100
    "Shadows2012": 0.0,    # -100..100
    "Whites2012": 0.0,     # -100..100
    "Blacks2012": 0.0,     # -100..100
}


@dataclass
class ProxyRenderResult:
    image: ImageArray
    luma_pre: ImageArray
    luma_post: ImageArray
    maps: Optional[Dict[str, ImageArray]] = None


def _is_torch(x: ImageArray) -> bool:
    return torch is not None and torch.is_tensor(x)


def _clip(x: ImageArray, lo: float, hi: float) -> ImageArray:
    if _is_torch(x):
        return x.clamp(lo, hi)
    return np.clip(x, lo, hi)


def _where(cond: ImageArray, a: ImageArray, b: ImageArray) -> ImageArray:
    if _is_torch(cond):
        return torch.where(cond, a, b)
    return np.where(cond, a, b)


def _smoothstep(x: ImageArray, edge0: float, edge1: float) -> ImageArray:
    t = _clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ensure_float01(img: ImageArray) -> ImageArray:
    if _is_torch(img):
        if img.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            img = img.float() / 255.0
        else:
            img = img.float()
        if float(img.max()) > 1.5:
            img = img / 255.0
        return _clip(img, 0.0, 1.0)

    if img.dtype.kind in {"u", "i"}:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if float(img.max()) > 1.5:
            img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _srgb_to_linear(x: ImageArray) -> ImageArray:
    thresh = 0.04045
    if _is_torch(x):
        return _where(x <= thresh, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return np.where(x <= thresh, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: ImageArray) -> ImageArray:
    thresh = 0.0031308
    if _is_torch(x):
        return _where(x <= thresh, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
    return np.where(x <= thresh, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def _compute_luma(rgb_lin: ImageArray) -> ImageArray:
    return (
        0.2126 * rgb_lin[..., 0] +
        0.7152 * rgb_lin[..., 1] +
        0.0722 * rgb_lin[..., 2]
    )


def _apply_exposure(rgb_lin: ImageArray, exposure_stops: float) -> ImageArray:
    if _is_torch(rgb_lin):
        if not torch.is_tensor(exposure_stops):
            exposure_stops = torch.tensor(
                float(exposure_stops),
                device=rgb_lin.device,
                dtype=rgb_lin.dtype,
            )
        scale = torch.pow(
            torch.tensor(2.0, device=rgb_lin.device, dtype=rgb_lin.dtype),
            exposure_stops,
        )
        return rgb_lin * scale
    return rgb_lin * (2.0 ** float(exposure_stops))


def _apply_contrast(luma: ImageArray, amount: float) -> ImageArray:
    # Simple pivoted contrast around 0.5
    k = 1.0 + 1.5 * amount
    return _clip(0.5 + (luma - 0.5) * k, 0.0, 1.0)


def _apply_region(luma: ImageArray, amount: float, weight: ImageArray) -> ImageArray:
    if _is_torch(luma):
        if not torch.is_tensor(amount):
            amount = torch.tensor(float(amount), device=luma.device, dtype=luma.dtype)
        pos = torch.clamp(amount, min=0.0)
        neg = torch.clamp(amount, max=0.0)
        return luma + pos * weight * (1.0 - luma) + neg * weight * luma
    amount = float(amount)
    pos = max(amount, 0.0)
    neg = min(amount, 0.0)
    return luma + pos * weight * (1.0 - luma) + neg * weight * luma


def _get_slider(sliders: Dict[str, float], name: str, like: ImageArray) -> ImageArray:
    val = sliders.get(name, SLIDER_DEFAULTS[name])
    if _is_torch(like):
        if torch.is_tensor(val):
            return val.to(device=like.device, dtype=like.dtype)
        return torch.tensor(float(val), device=like.device, dtype=like.dtype)
    return float(val)


def render_proxy(
    image: ImageArray,
    sliders: Dict[str, float],
    input_space: str = "srgb",
    output_space: str = "srgb",
    return_maps: bool = False,
) -> ImageArray | ProxyRenderResult:
    """
    Render a proxy image using a subset of Lightroom-style tone sliders.

    Args:
        image: HxWx3 RGB image, uint8 or float in [0, 1]
        sliders: Dict of Lightroom slider names to values
        input_space: "srgb" or "linear"
        output_space: "srgb" or "linear"
        return_maps: If True, return ProxyRenderResult with debug maps
    """
    img = _ensure_float01(image)

    if input_space == "srgb":
        rgb_lin = _srgb_to_linear(img)
    elif input_space == "linear":
        rgb_lin = img
    else:
        raise ValueError(f"Unsupported input_space: {input_space}")

    s = {**SLIDER_DEFAULTS, **(sliders or {})}
    exposure = _get_slider(s, "Exposure2012", rgb_lin)
    contrast = _get_slider(s, "Contrast2012", rgb_lin) / 100.0
    highlights = _get_slider(s, "Highlights2012", rgb_lin) / 100.0
    shadows = _get_slider(s, "Shadows2012", rgb_lin) / 100.0
    whites = _get_slider(s, "Whites2012", rgb_lin) / 100.0
    blacks = _get_slider(s, "Blacks2012", rgb_lin) / 100.0

    # Exposure in linear space
    rgb_lin = _apply_exposure(rgb_lin, exposure)
    rgb_lin = _clip(rgb_lin, 0.0, 1.0)

    luma_pre = _compute_luma(rgb_lin)
    luma = luma_pre

    # Contrast then tone regions
    luma = _apply_contrast(luma, contrast)

    shadow_w = 1.0 - _smoothstep(luma, 0.18, 0.5)
    highlight_w = _smoothstep(luma, 0.5, 0.82)
    black_w = 1.0 - _smoothstep(luma, 0.02, 0.2)
    white_w = _smoothstep(luma, 0.8, 0.98)

    luma = _apply_region(luma, shadows * 0.8, shadow_w)
    luma = _apply_region(luma, highlights * 0.8, highlight_w)
    luma = _apply_region(luma, blacks * 0.9, black_w)
    luma = _apply_region(luma, whites * 0.9, white_w)

    luma = _clip(luma, 0.0, 1.0)

    # Preserve chroma by scaling to the adjusted luma
    scale = luma / (luma_pre + 1e-6)
    rgb_lin = rgb_lin * scale[..., None]
    rgb_lin = _clip(rgb_lin, 0.0, 1.0)

    if output_space == "srgb":
        out = _linear_to_srgb(rgb_lin)
    elif output_space == "linear":
        out = rgb_lin
    else:
        raise ValueError(f"Unsupported output_space: {output_space}")

    out = _clip(out, 0.0, 1.0)

    if not return_maps:
        return out

    maps = {
        "shadow_w": shadow_w,
        "highlight_w": highlight_w,
        "black_w": black_w,
        "white_w": white_w,
    }
    return ProxyRenderResult(image=out, luma_pre=luma_pre, luma_post=luma, maps=maps)


def render_proxy_from_path(
    image_path: str,
    sliders: Dict[str, float],
    target_size: tuple[int, int] | None = None,
    return_maps: bool = False,
) -> ImageArray | ProxyRenderResult:
    """Load an image and render the proxy (convenience wrapper)."""
    from image_loader import load_image

    image = load_image(image_path, target_size=target_size)
    img_array = np.asarray(image, dtype=np.float32)
    return render_proxy(img_array, sliders, return_maps=return_maps)
