from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_mask(path: str | Path, mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask_u8)


def save_overlay(
    path: str | Path,
    image_rgb,
    mask,
    alpha=0.45,
):
    image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    mask_bool = mask > 0

    overlay = image.copy()

    # Green leaf overlay.
    overlay[mask_bool] = (
        0.5 * overlay[mask_bool]
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    result = cv2.addWeighted(
        image,
        1.0 - float(alpha),
        overlay,
        float(alpha),
        0,
    )

    cv2.imwrite(str(path), result)
