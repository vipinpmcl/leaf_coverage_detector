from __future__ import annotations

import numpy as np
from PIL import Image


def _gray(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def _masked_core(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    p = np.pad(m, 1, constant_values=False)
    return (
        p[:-2, :-2] & p[:-2, 1:-1] & p[:-2, 2:] &
        p[1:-1, :-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
        p[2:, :-2] & p[2:, 1:-1] & p[2:, 2:]
    )


def _laplacian_variance(gray, mask):
    core = _masked_core(mask)
    p = np.pad(gray, 1, mode="reflect")
    lap = (
        p[:-2, 1:-1] + p[2:, 1:-1] +
        p[1:-1, :-2] + p[1:-1, 2:] -
        4.0 * p[1:-1, 1:-1]
    )
    v = lap[core]
    return float(np.var(v)) if v.size else 0.0


def _tenengrad(gray, mask):
    core = _masked_core(mask)
    p = np.pad(gray, 1, mode="reflect")
    gx = (-p[:-2, :-2] + p[:-2, 2:] -
          2*p[1:-1, :-2] + 2*p[1:-1, 2:] -
          p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:] +
          p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:])
    v = (gx*gx + gy*gy)[core]
    return float(np.mean(v)) if v.size else 0.0


def _brenner(gray, mask):
    m = mask.astype(bool)
    valid = m[:, :-2] & m[:, 2:]
    d = gray[:, 2:] - gray[:, :-2]
    v = d[valid]
    return float(np.mean(v*v)) if v.size else 0.0


def _fft_high_frequency_ratio(gray, mask):
    ys, xs = np.where(mask)
    if len(xs) < 16:
        return 0.0
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    crop = gray[y0:y1, x0:x1].copy()
    cm = mask[y0:y1, x0:x1]
    if crop.size < 64:
        return 0.0
    crop[~cm] = crop[cm].mean()
    crop -= crop.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    h, w = power.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy-h/2)/max(h,1))**2 + ((xx-w/2)/max(w,1))**2)
    total = power.sum()
    return float(power[r >= 0.20].sum() / total) if total > 0 else 0.0


def _edge_density(gray, mask):
    core = _masked_core(mask)
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    # Align to the core region.
    mag = np.sqrt(
        ((gray[1:-1, 2:] - gray[1:-1, :-2]) / 2.0)**2 +
        ((gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2.0)**2
    )
    v = mag[core[1:-1, 1:-1]]
    if v.size == 0:
        return 0.0
    return float(np.mean(v > np.percentile(v, 75)))


def calculate_quality_metrics(image: Image.Image, mask: np.ndarray, probability: np.ndarray):
    """Raw metrics; sharpness metrics are computed only over the predicted leaf."""
    h, w = mask.shape
    area = int(mask.sum())
    total = h * w
    coverage = 100.0 * area / max(total, 1)

    if area:
        ys, xs = np.where(mask)
        bw = int(xs.max()-xs.min()+1)
        bh = int(ys.max()-ys.min()+1)
        bbox_area = bw * bh
        bbox_fill = 100.0 * area / max(bbox_area, 1)
        bw_pct = 100.0 * bw / max(w, 1)
        bh_pct = 100.0 * bh / max(h, 1)
        leaf_prob = float(probability[mask].mean())
    else:
        bw = bh = bbox_area = 0
        bbox_fill = bw_pct = bh_pct = leaf_prob = 0.0

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixels = rgb[mask]
    if len(pixels):
        brightness = 0.299*pixels[:,0] + 0.587*pixels[:,1] + 0.114*pixels[:,2]
        mx, mn = pixels.max(1), pixels.min(1)
        saturation = np.where(mx > 0, (mx-mn)/mx, 0)
        brightness_mean = float(brightness.mean())
        brightness_std = float(brightness.std())
        dark_clip = float(np.mean(brightness <= 5)*100)
        bright_clip = float(np.mean(brightness >= 250)*100)
        sat_mean = float(saturation.mean())
    else:
        brightness_mean = brightness_std = dark_clip = bright_clip = sat_mean = 0.0

    gray = _gray(image)

    return {
        "image_width": w,
        "image_height": h,
        "image_area_pixels": total,
        "leaf_area_pixels": area,
        "leaf_coverage_percent": coverage,
        "leaf_bbox_width_pixels": bw,
        "leaf_bbox_height_pixels": bh,
        "leaf_bbox_area_pixels": bbox_area,
        "leaf_bbox_width_percent": bw_pct,
        "leaf_bbox_height_percent": bh_pct,
        "leaf_bbox_fill_percent": bbox_fill,
        "leaf_mean_probability": leaf_prob,
        "masked_laplacian_variance": _laplacian_variance(gray, mask),
        "masked_tenengrad": _tenengrad(gray, mask),
        "masked_brenner": _brenner(gray, mask),
        "masked_fft_high_frequency_ratio": _fft_high_frequency_ratio(gray, mask),
        "masked_edge_density": _edge_density(gray, mask),
        "masked_brightness_mean": brightness_mean,
        "masked_brightness_std": brightness_std,
        "masked_dark_clip_percent": dark_clip,
        "masked_bright_clip_percent": bright_clip,
        "masked_saturation_mean": sat_mean,
    }
