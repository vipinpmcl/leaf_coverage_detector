from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygons(
    mask,
    epsilon_ratio=0.002,
    min_area=10000,
):
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < float(min_area):
            continue

        perimeter = float(cv2.arcLength(contour, True))
        epsilon = float(epsilon_ratio) * perimeter

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        if len(approx) < 3:
            continue

        points = [
            [float(point[0][0]), float(point[0][1])]
            for point in approx
        ]

        candidates.append((area, points))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [points for _, points in candidates]
