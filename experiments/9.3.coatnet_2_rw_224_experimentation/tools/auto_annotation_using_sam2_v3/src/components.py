from __future__ import annotations

import cv2
import numpy as np


def connected_components(
    mask,
    min_area=100,
    keep_largest_n=0,
):
    mask_u8 = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    components = []

    for label_id in range(1, n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < int(min_area):
            continue

        component = (labels == label_id).astype(np.uint8)
        components.append((area, component))

    components.sort(key=lambda item: item[0], reverse=True)

    if keep_largest_n > 0:
        components = components[:int(keep_largest_n)]

    return [component for _, component in components]


def remove_small_components(mask, min_area):
    result = np.zeros_like(mask, dtype=np.uint8)

    for component in connected_components(
        mask,
        min_area=min_area,
        keep_largest_n=0,
    ):
        result = np.maximum(result, component)

    return result
