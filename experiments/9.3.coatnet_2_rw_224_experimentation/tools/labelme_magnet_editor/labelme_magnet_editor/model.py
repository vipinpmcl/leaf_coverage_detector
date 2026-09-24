from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class LabelMeDocument:
    def __init__(self):
        self.data: dict[str, Any] = {}
        self.json_path: Path | None = None
        self.image_path: Path | None = None
        self.image_bgr: np.ndarray | None = None

    @property
    def shapes(self):
        return self.data.get("shapes", [])

    def load(self, json_path: Path):
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.json_path = json_path
        image_path_value = self.data.get("imagePath", "")

        if image_path_value:
            candidate = json_path.parent / image_path_value
        else:
            candidate = json_path.with_suffix(".jpg")

        if not candidate.exists():
            candidates = [
                json_path.with_suffix(ext)
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
            ]
            candidate = next((p for p in candidates if p.exists()), candidate)

        self.image_path = candidate
        self.image_bgr = cv2.imread(str(candidate), cv2.IMREAD_COLOR)

        if self.image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {candidate}")

    def copy_points(self) -> list[list[list[float]]]:
        result = []
        for shape in self.shapes:
            result.append(copy.deepcopy(shape.get("points", [])))
        return result

    def restore_points(self, snapshot):
        for shape, points in zip(self.shapes, snapshot):
            if "points" in shape:
                shape["points"] = copy.deepcopy(points)

    def polygon_shapes(self):
        for index, shape in enumerate(self.shapes):
            if shape.get("shape_type", "polygon") == "polygon":
                yield index, shape

    def save(self, output_path: Path):
        output_path = Path(output_path)
        data = copy.deepcopy(self.data)

        # LabelMe stores imagePath relative to the JSON file.
        if self.image_path:
            try:
                data["imagePath"] = str(self.image_path.relative_to(output_path.parent))
            except ValueError:
                data["imagePath"] = self.image_path.name

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path
