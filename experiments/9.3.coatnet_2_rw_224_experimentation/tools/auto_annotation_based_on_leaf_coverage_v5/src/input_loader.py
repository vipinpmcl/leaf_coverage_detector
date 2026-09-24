from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def discover_prediction_samples(
    input_dir: str | Path,
    original_name: str = "original.jpg",
    mask_name: str = "mask.png",
    skip_directories: list[str] | None = None,
):
    input_dir = Path(input_dir).resolve()
    skip_directories = set(skip_directories or [])

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d not in skip_directories]

        root_path = Path(root)

        if mask_name not in files:
            continue

        mask_path = root_path / mask_name
        image_path = root_path / original_name

        if image_path.exists():
            yield {
                "sample_dir": root_path,
                "image_path": image_path,
                "mask_path": mask_path,
                "error": None,
            }
        else:
            yield {
                "sample_dir": root_path,
                "image_path": None,
                "mask_path": mask_path,
                "error": f"Missing {original_name}",
            }


def read_image(path: str | Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")

    return (mask > 0).astype(np.uint8)
