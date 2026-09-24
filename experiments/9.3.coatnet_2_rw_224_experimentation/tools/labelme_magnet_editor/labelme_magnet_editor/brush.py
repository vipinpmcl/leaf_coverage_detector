from __future__ import annotations

import math
import numpy as np


def apply_magnet(points, center, radius, force, mode):
    """
    Deform polygon vertices in-place conceptually and return a new Nx2 array.

    radius: image-pixel radius
    force: 0..1, normalized strength
    mode: 'attract' or 'repel'

    The displacement is proportional to a smooth quadratic falloff:
        influence = (1 - d/r)^2
    and capped by a fraction of radius to prevent explosive movement.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    if pts.size == 0:
        return pts

    center = np.asarray(center, dtype=np.float64)
    delta = center[None, :] - pts
    dist = np.linalg.norm(delta, axis=1)

    mask = dist > 1e-9
    inside = dist < radius

    influence = np.zeros_like(dist)
    influence[inside] = (1.0 - dist[inside] / radius) ** 2

    # Exponential-ish force response gives useful control across slider range.
    strength = float(np.clip(force, 0.0, 1.0))
    strength = strength ** 1.35

    # A brush stroke should move points smoothly, not teleport them.
    max_step = radius * 0.18 * strength

    direction = np.zeros_like(delta)
    direction[mask] = delta[mask] / dist[mask, None]

    sign = 1.0 if mode == "attract" else -1.0
    step = sign * direction * (max_step * influence)[:, None]
    pts += step
    return pts


def brush_weight(distance, radius):
    if radius <= 0 or distance >= radius:
        return 0.0
    return float((1.0 - distance / radius) ** 2)
