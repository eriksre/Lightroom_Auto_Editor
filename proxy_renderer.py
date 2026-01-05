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
    "Texture": 0.0,        # -100..100
    "Clarity2012": 0.0,    # -100..100
    "Dehaze": 0.0,         # -100..100
    "Vibrance": 0.0,       # -100..100
    "Saturation": 0.0,     # -100..100
    "IncrementalTemperature": 0.0, # delta Kelvin
    "IncrementalTint": 0.0,        # -150..150
    # HSL - Hue
    "HueAdjustmentRed": 0.0,
    "HueAdjustmentOrange": 0.0,
    "HueAdjustmentYellow": 0.0,
    "HueAdjustmentGreen": 0.0,
    "HueAdjustmentAqua": 0.0,
    "HueAdjustmentBlue": 0.0,
    "HueAdjustmentPurple": 0.0,
    "HueAdjustmentMagenta": 0.0,
    # HSL - Saturation
    "SaturationAdjustmentRed": 0.0,
    "SaturationAdjustmentOrange": 0.0,
    "SaturationAdjustmentYellow": 0.0,
    "SaturationAdjustmentGreen": 0.0,
    "SaturationAdjustmentAqua": 0.0,
    "SaturationAdjustmentBlue": 0.0,
    "SaturationAdjustmentPurple": 0.0,
    "SaturationAdjustmentMagenta": 0.0,
    # HSL - Luminance
    "LuminanceAdjustmentRed": 0.0,
    "LuminanceAdjustmentOrange": 0.0,
    "LuminanceAdjustmentYellow": 0.0,
    "LuminanceAdjustmentGreen": 0.0,
    "LuminanceAdjustmentAqua": 0.0,
    "LuminanceAdjustmentBlue": 0.0,
    "LuminanceAdjustmentPurple": 0.0,
    "LuminanceAdjustmentMagenta": 0.0,
    # Color grading
    "ColorGradeBlending": 50.0,
    "ColorGradeGlobalHue": 0.0,
    "ColorGradeGlobalSat": 0.0,
    "ColorGradeGlobalLum": 0.0,
    "ColorGradeShadowHue": 0.0,
    "ColorGradeShadowSat": 0.0,
    "ColorGradeShadowLum": 0.0,
    "ColorGradeMidtoneHue": 0.0,
    "ColorGradeMidtoneSat": 0.0,
    "ColorGradeMidtoneLum": 0.0,
    "ColorGradeHighlightHue": 0.0,
    "ColorGradeHighlightSat": 0.0,
    "ColorGradeHighlightLum": 0.0,
    # Calibration
    "ShadowTint": 0.0,
    "RedHue": 0.0,
    "RedSaturation": 0.0,
    "GreenHue": 0.0,
    "GreenSaturation": 0.0,
    "BlueHue": 0.0,
    "BlueSaturation": 0.0,
}

HSL_COLOR_CENTERS = {
    "Red": 0.0,
    "Orange": 30.0 / 360.0,
    "Yellow": 60.0 / 360.0,
    "Green": 120.0 / 360.0,
    "Aqua": 180.0 / 360.0,
    "Blue": 240.0 / 360.0,
    "Purple": 270.0 / 360.0,
    "Magenta": 300.0 / 360.0,
}

HSL_WEIGHT_WIDTH = 0.08
HUE_ADJ_MAX_DEG = 30.0
CALIBRATION_HUE_MAX_DEG = 25.0


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


def _temp_to_rgb(temp_k: float) -> np.ndarray:
    temp_k = float(np.clip(temp_k, 1000.0, 40000.0))
    t = temp_k / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        if t <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0
    rgb = np.array([r, g, b], dtype=np.float32)
    return np.clip(rgb / 255.0, 0.0, 1.0)


def _apply_white_balance(
    rgb_lin: ImageArray,
    temp_delta: float,
    tint_delta: float,
    base_temp: float = 6500.0,
) -> ImageArray:
    if abs(float(temp_delta)) < 1e-6 and abs(float(tint_delta)) < 1e-6:
        return rgb_lin
    base_temp = float(base_temp)
    target_temp = base_temp + float(temp_delta)
    target_temp = float(np.clip(target_temp, 2000.0, 12000.0))

    neutral = _temp_to_rgb(base_temp)
    target = _temp_to_rgb(target_temp)
    wb = target / (neutral + 1e-6)

    tint = float(tint_delta)
    g_scale = 1.0 + tint / 200.0
    rb_scale = 1.0 - tint / 400.0
    wb = wb * np.array([rb_scale, g_scale, rb_scale], dtype=np.float32)

    if _is_torch(rgb_lin):
        wb = torch.tensor(wb, device=rgb_lin.device, dtype=rgb_lin.dtype)
    return rgb_lin * wb


def _box_blur_2d(x: ImageArray, radius: int) -> ImageArray:
    if radius <= 0:
        return x
    if _is_torch(x):
        h, w = x.shape[-2], x.shape[-1]
        if min(h, w) <= 1:
            return x
        radius = int(min(radius, min(h, w) - 1))
        import torch.nn.functional as F
        x4 = x[None, None, ...]
        x4 = F.pad(x4, (radius, radius, radius, radius), mode="reflect")
        k = 2 * radius + 1
        out = F.avg_pool2d(x4, kernel_size=k, stride=1)
        return out[0, 0, ...]

    h, w = x.shape[-2], x.shape[-1]
    if min(h, w) <= 1:
        return x
    radius = int(min(radius, min(h, w) - 1))
    pad = radius
    x_pad = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
    k = 2 * radius + 1
    integral = np.pad(x_pad, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    out = total / float(k * k)
    return out[pad:-pad, pad:-pad]


def _apply_local_contrast(
    rgb_lin: ImageArray,
    amount: float,
    radius: int,
    mask: Optional[ImageArray] = None,
) -> ImageArray:
    if abs(float(amount)) < 1e-6:
        return rgb_lin
    luma = _compute_luma(rgb_lin)
    blur = _box_blur_2d(luma, radius)
    detail = luma - blur
    if mask is not None:
        detail = detail * mask
    luma_adj = _clip(luma + detail * amount, 0.0, 1.0)
    scale = luma_adj / (luma + 1e-6)
    return _clip(rgb_lin * scale[..., None], 0.0, 1.0)


def _rgb_to_hsl(rgb: ImageArray) -> tuple[ImageArray, ImageArray, ImageArray]:
    eps = 1e-6
    if _is_torch(rgb):
        maxc, _ = torch.max(rgb, dim=-1)
        minc, _ = torch.min(rgb, dim=-1)
        delta = maxc - minc
        l = (maxc + minc) / 2.0
        s = delta / (1.0 - torch.abs(2.0 * l - 1.0) + eps)
        s = torch.where(delta > eps, s, torch.zeros_like(s))

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        h = torch.zeros_like(l)
        mask = delta > eps
        h_r = torch.remainder((g - b) / (delta + eps), 6.0)
        h_g = ((b - r) / (delta + eps)) + 2.0
        h_b = ((r - g) / (delta + eps)) + 4.0
        h = torch.where((maxc == r) & mask, h_r, h)
        h = torch.where((maxc == g) & mask, h_g, h)
        h = torch.where((maxc == b) & mask, h_b, h)
        h = torch.remainder(h / 6.0, 1.0)
        return h, s, l

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    l = (maxc + minc) / 2.0
    s = delta / (1.0 - np.abs(2.0 * l - 1.0) + eps)
    s = np.where(delta > eps, s, 0.0)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    h = np.zeros_like(l)
    mask = delta > eps
    h_r = np.mod((g - b) / (delta + eps), 6.0)
    h_g = ((b - r) / (delta + eps)) + 2.0
    h_b = ((r - g) / (delta + eps)) + 4.0
    h = np.where((maxc == r) & mask, h_r, h)
    h = np.where((maxc == g) & mask, h_g, h)
    h = np.where((maxc == b) & mask, h_b, h)
    h = np.mod(h / 6.0, 1.0)
    return h, s, l


def _hsl_to_rgb(h: ImageArray, s: ImageArray, l: ImageArray) -> ImageArray:
    eps = 1e-6
    if _is_torch(h):
        def hue_to_rgb(p: ImageArray, q: ImageArray, t: ImageArray) -> ImageArray:
            t = torch.remainder(t, 1.0)
            return torch.where(
                t < (1.0 / 6.0),
                p + (q - p) * 6.0 * t,
                torch.where(
                    t < 0.5,
                    q,
                    torch.where(
                        t < (2.0 / 3.0),
                        p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                        p,
                    ),
                ),
            )

        q = torch.where(l < 0.5, l * (1.0 + s), l + s - l * s)
        p = 2.0 * l - q
        r = hue_to_rgb(p, q, h + 1.0 / 3.0)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1.0 / 3.0)
        rgb = torch.stack([r, g, b], dim=-1)
        return torch.where(s[..., None] > eps, rgb, l[..., None])

    def hue_to_rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < (1.0 / 6.0),
            p + (q - p) * 6.0 * t,
            np.where(
                t < 0.5,
                q,
                np.where(
                    t < (2.0 / 3.0),
                    p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                    p,
                ),
            ),
        )

    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q
    r = hue_to_rgb(p, q, h + 1.0 / 3.0)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1.0 / 3.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.where(s[..., None] > eps, rgb, l[..., None])


def _hue_weight(h: ImageArray, center: float, width: float = HSL_WEIGHT_WIDTH) -> ImageArray:
    dist = _where(h > center, h - center, center - h)
    dist = _where(dist > 0.5, 1.0 - dist, dist)
    return _clip(1.0 - _smoothstep(dist, 0.0, width), 0.0, 1.0)


def _adjust_saturation(s: ImageArray, amount: float, weight: Optional[ImageArray] = None) -> ImageArray:
    if abs(float(amount)) < 1e-6:
        return s
    if weight is None:
        weight = 1.0
    pos = max(float(amount), 0.0)
    neg = min(float(amount), 0.0)
    s = s + pos * weight * (1.0 - s)
    s = s + neg * weight * s
    return _clip(s, 0.0, 1.0)


def _color_from_hsl(h: float, s: float, l: float, like: ImageArray) -> ImageArray:
    h = float(h) % 1.0
    s = float(np.clip(s, 0.0, 1.0))
    l = float(np.clip(l, 0.0, 1.0))
    if s <= 1e-6:
        rgb = np.array([l, l, l], dtype=np.float32)
    else:
        q = l * (1.0 + s) if l < 0.5 else (l + s - l * s)
        p = 2.0 * l - q
        def hue_to_rgb(p_val: float, q_val: float, t_val: float) -> float:
            t_val = t_val % 1.0
            if t_val < 1.0 / 6.0:
                return p_val + (q_val - p_val) * 6.0 * t_val
            if t_val < 0.5:
                return q_val
            if t_val < 2.0 / 3.0:
                return p_val + (q_val - p_val) * (2.0 / 3.0 - t_val) * 6.0
            return p_val
        r = hue_to_rgb(p, q, h + 1.0 / 3.0)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1.0 / 3.0)
        rgb = np.array([r, g, b], dtype=np.float32)
    if _is_torch(like):
        return torch.tensor(rgb, device=like.device, dtype=like.dtype)
    return rgb


def _apply_presence(
    rgb_lin: ImageArray,
    texture: float,
    clarity: float,
    dehaze: float,
) -> ImageArray:
    if abs(float(texture)) > 1e-6:
        rgb_lin = _apply_local_contrast(rgb_lin, texture * 0.4, radius=1)
    if abs(float(clarity)) > 1e-6:
        luma = _compute_luma(rgb_lin)
        mid_w = _smoothstep(luma, 0.1, 0.4) * (1.0 - _smoothstep(luma, 0.6, 0.9))
        rgb_lin = _apply_local_contrast(rgb_lin, clarity * 0.6, radius=3, mask=mid_w)
    if abs(float(dehaze)) > 1e-6:
        rgb_lin = _apply_local_contrast(rgb_lin, dehaze * 0.6, radius=8)
        luma = _compute_luma(rgb_lin)
        luma_adj = _apply_contrast(luma, dehaze * 0.35)
        scale = luma_adj / (luma + 1e-6)
        rgb_lin = _clip(rgb_lin * scale[..., None], 0.0, 1.0)
    return rgb_lin


def _apply_color_mixer(
    rgb_lin: ImageArray,
    vibrance: float,
    saturation: float,
    hsl_values: Dict[str, float],
) -> ImageArray:
    if (
        abs(float(vibrance)) < 1e-6
        and abs(float(saturation)) < 1e-6
        and all(abs(float(v)) < 1e-6 for v in hsl_values.values())
    ):
        return rgb_lin

    h, s, l = _rgb_to_hsl(rgb_lin)
    h_base = h

    if abs(float(vibrance)) > 1e-6:
        skin_weight = _hue_weight(h_base, 30.0 / 360.0, width=0.07)
        s = _clip(s + (1.0 - s) * vibrance * (1.0 - 0.5 * skin_weight), 0.0, 1.0)

    if abs(float(saturation)) > 1e-6:
        s = _adjust_saturation(s, saturation)

    h_shift = 0.0
    for color, center in HSL_COLOR_CENTERS.items():
        weight = _hue_weight(h_base, center)
        hue_delta = hsl_values[f"HueAdjustment{color}"] / 100.0
        sat_delta = hsl_values[f"SaturationAdjustment{color}"] / 100.0
        lum_delta = hsl_values[f"LuminanceAdjustment{color}"] / 100.0
        if abs(float(hue_delta)) > 1e-6:
            h_shift = h_shift + weight * hue_delta * (HUE_ADJ_MAX_DEG / 360.0)
        if abs(float(sat_delta)) > 1e-6:
            s = _adjust_saturation(s, sat_delta, weight)
        if abs(float(lum_delta)) > 1e-6:
            l = _apply_region(l, lum_delta, weight)

    h = (h + h_shift) % 1.0
    return _hsl_to_rgb(h, s, l)


def _color_grade_weights(luma: ImageArray, blend: float) -> tuple[ImageArray, ImageArray, ImageArray]:
    blend = float(np.clip(blend, 0.0, 1.0))
    shadow_end = 0.35 + 0.1 * blend
    highlight_start = 0.65 - 0.1 * blend
    shadow_w = 1.0 - _smoothstep(luma, 0.0, shadow_end)
    highlight_w = _smoothstep(luma, highlight_start, 1.0)
    mid_w = _clip(1.0 - shadow_w - highlight_w, 0.0, 1.0)
    return shadow_w, mid_w, highlight_w


def _apply_color_grade(
    rgb_lin: ImageArray,
    luma: ImageArray,
    hue_deg: float,
    sat: float,
    lum: float,
    weight: ImageArray,
) -> tuple[ImageArray, ImageArray]:
    if not hasattr(weight, "shape"):
        if _is_torch(luma):
            weight = torch.full_like(luma, float(weight))
        else:
            weight = np.full_like(luma, float(weight), dtype=np.float32)
    if abs(float(lum)) > 1e-6:
        luma_adj = _apply_region(luma, lum, weight)
        scale = luma_adj / (luma + 1e-6)
        rgb_lin = _clip(rgb_lin * scale[..., None], 0.0, 1.0)
        luma = luma_adj

    if abs(float(sat)) > 1e-6:
        sat_strength = min(abs(float(sat)), 1.0) * 0.6
        if sat > 0:
            color = _color_from_hsl(hue_deg / 360.0, abs(float(sat)), 0.5, rgb_lin)
            rgb_lin = rgb_lin + weight[..., None] * sat_strength * (color - rgb_lin)
        else:
            rgb_lin = rgb_lin + weight[..., None] * sat_strength * (luma[..., None] - rgb_lin)
        rgb_lin = _clip(rgb_lin, 0.0, 1.0)

    return rgb_lin, luma


def _apply_color_grading(
    rgb_lin: ImageArray,
    sliders: Dict[str, float],
) -> tuple[ImageArray, Dict[str, ImageArray]]:
    shadow_h = float(sliders["ColorGradeShadowHue"])
    shadow_s = float(sliders["ColorGradeShadowSat"]) / 100.0
    shadow_l = float(sliders["ColorGradeShadowLum"]) / 100.0
    mid_h = float(sliders["ColorGradeMidtoneHue"])
    mid_s = float(sliders["ColorGradeMidtoneSat"]) / 100.0
    mid_l = float(sliders["ColorGradeMidtoneLum"]) / 100.0
    high_h = float(sliders["ColorGradeHighlightHue"])
    high_s = float(sliders["ColorGradeHighlightSat"]) / 100.0
    high_l = float(sliders["ColorGradeHighlightLum"]) / 100.0
    glob_h = float(sliders["ColorGradeGlobalHue"])
    glob_s = float(sliders["ColorGradeGlobalSat"]) / 100.0
    glob_l = float(sliders["ColorGradeGlobalLum"]) / 100.0
    blend = float(sliders["ColorGradeBlending"]) / 100.0

    luma = _compute_luma(rgb_lin)
    shadow_w, mid_w, high_w = _color_grade_weights(luma, blend)
    rgb_lin, luma = _apply_color_grade(rgb_lin, luma, shadow_h, shadow_s, shadow_l, shadow_w)
    rgb_lin, luma = _apply_color_grade(rgb_lin, luma, mid_h, mid_s, mid_l, mid_w)
    rgb_lin, luma = _apply_color_grade(rgb_lin, luma, high_h, high_s, high_l, high_w)
    rgb_lin, luma = _apply_color_grade(rgb_lin, luma, glob_h, glob_s, glob_l, 1.0)
    maps = {
        "color_grade_shadow_w": shadow_w,
        "color_grade_mid_w": mid_w,
        "color_grade_highlight_w": high_w,
    }
    return rgb_lin, maps


def _apply_calibration(
    rgb_lin: ImageArray,
    sliders: Dict[str, float],
) -> ImageArray:
    shadow_tint = float(sliders["ShadowTint"]) / 100.0
    red_h = float(sliders["RedHue"]) / 100.0
    red_s = float(sliders["RedSaturation"]) / 100.0
    green_h = float(sliders["GreenHue"]) / 100.0
    green_s = float(sliders["GreenSaturation"]) / 100.0
    blue_h = float(sliders["BlueHue"]) / 100.0
    blue_s = float(sliders["BlueSaturation"]) / 100.0

    if abs(shadow_tint) > 1e-6:
        luma = _compute_luma(rgb_lin)
        shadow_w = 1.0 - _smoothstep(luma, 0.12, 0.45)
        g_scale = 1.0 + shadow_tint * 0.4
        rb_scale = 1.0 - shadow_tint * 0.2
        if _is_torch(rgb_lin):
            wb = torch.tensor([rb_scale, g_scale, rb_scale], device=rgb_lin.device, dtype=rgb_lin.dtype)
        else:
            wb = np.array([rb_scale, g_scale, rb_scale], dtype=np.float32)
        rgb_lin = _clip(rgb_lin * (1.0 + shadow_w[..., None] * (wb - 1.0)), 0.0, 1.0)

    if (
        abs(red_h) < 1e-6 and abs(red_s) < 1e-6 and
        abs(green_h) < 1e-6 and abs(green_s) < 1e-6 and
        abs(blue_h) < 1e-6 and abs(blue_s) < 1e-6
    ):
        return rgb_lin

    h, s, l = _rgb_to_hsl(rgb_lin)
    rgb_base = rgb_lin
    if _is_torch(rgb_base):
        sum_rgb = rgb_base[..., 0] + rgb_base[..., 1] + rgb_base[..., 2]
        weight_r = (rgb_base[..., 0] / (sum_rgb + 1e-6)) * s
        weight_g = (rgb_base[..., 1] / (sum_rgb + 1e-6)) * s
        weight_b = (rgb_base[..., 2] / (sum_rgb + 1e-6)) * s
    else:
        sum_rgb = rgb_base[..., 0] + rgb_base[..., 1] + rgb_base[..., 2]
        weight_r = (rgb_base[..., 0] / (sum_rgb + 1e-6)) * s
        weight_g = (rgb_base[..., 1] / (sum_rgb + 1e-6)) * s
        weight_b = (rgb_base[..., 2] / (sum_rgb + 1e-6)) * s

    hue_scale = CALIBRATION_HUE_MAX_DEG / 360.0
    h = (h + weight_r * red_h * hue_scale) % 1.0
    h = (h + weight_g * green_h * hue_scale) % 1.0
    h = (h + weight_b * blue_h * hue_scale) % 1.0

    s = _adjust_saturation(s, red_s, weight_r)
    s = _adjust_saturation(s, green_s, weight_g)
    s = _adjust_saturation(s, blue_s, weight_b)

    return _hsl_to_rgb(h, s, l)


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


def render_proxy(
    image: ImageArray,
    sliders: Dict[str, float],
    input_space: str = "srgb",
    output_space: str = "srgb",
    return_maps: bool = False,
) -> ImageArray | ProxyRenderResult:
    """
    Render a proxy image using Lightroom-style tone and color sliders.

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
    exposure = float(s["Exposure2012"])
    contrast = float(s["Contrast2012"]) / 100.0
    highlights = float(s["Highlights2012"]) / 100.0
    shadows = float(s["Shadows2012"]) / 100.0
    whites = float(s["Whites2012"]) / 100.0
    blacks = float(s["Blacks2012"]) / 100.0
    texture = float(s["Texture"]) / 100.0
    clarity = float(s["Clarity2012"]) / 100.0
    dehaze = float(s["Dehaze"]) / 100.0
    vibrance = float(s["Vibrance"]) / 100.0
    saturation = float(s["Saturation"]) / 100.0

    temp_delta = float(s["IncrementalTemperature"])
    tint_delta = float(s["IncrementalTint"])
    if sliders:
        if sliders.get("Temperature") is not None:
            temp_delta = float(sliders["Temperature"]) - 6500.0
        if sliders.get("Tint") is not None:
            tint_delta = float(sliders["Tint"])

    # White balance in linear space (approximate, incremental by default).
    rgb_lin = _apply_white_balance(rgb_lin, temp_delta, tint_delta)
    rgb_lin = _clip(rgb_lin, 0.0, 1.0)

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

    # Presence adjustments
    rgb_lin = _apply_presence(rgb_lin, texture, clarity, dehaze)

    # Color adjustments in a gamma-encoded space for closer perceptual behavior.
    rgb_color = _clip(_linear_to_srgb(rgb_lin), 0.0, 1.0)

    hsl_values = {
        f"HueAdjustment{color}": float(s[f"HueAdjustment{color}"])
        for color in HSL_COLOR_CENTERS
    }
    hsl_values.update({
        f"SaturationAdjustment{color}": float(s[f"SaturationAdjustment{color}"])
        for color in HSL_COLOR_CENTERS
    })
    hsl_values.update({
        f"LuminanceAdjustment{color}": float(s[f"LuminanceAdjustment{color}"])
        for color in HSL_COLOR_CENTERS
    })
    rgb_color = _apply_color_mixer(rgb_color, vibrance, saturation, hsl_values)

    color_grade_maps = {}
    if any(
        abs(float(s[key])) > 1e-6
        for key in (
            "ColorGradeGlobalSat", "ColorGradeGlobalLum",
            "ColorGradeShadowSat", "ColorGradeShadowLum",
            "ColorGradeMidtoneSat", "ColorGradeMidtoneLum",
            "ColorGradeHighlightSat", "ColorGradeHighlightLum",
        )
    ):
        rgb_color, color_grade_maps = _apply_color_grading(rgb_color, s)

    rgb_color = _apply_calibration(rgb_color, s)
    rgb_color = _clip(rgb_color, 0.0, 1.0)

    if output_space == "srgb":
        out = rgb_color
    elif output_space == "linear":
        out = _srgb_to_linear(rgb_color)
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
    maps.update(color_grade_maps)
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
