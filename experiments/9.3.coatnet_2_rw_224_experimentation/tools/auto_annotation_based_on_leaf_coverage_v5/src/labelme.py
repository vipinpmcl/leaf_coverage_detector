from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image


def encode_image_base64(image_path: str | Path) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image_bytes = image_path.read_bytes()
    if not image_bytes:
        raise RuntimeError(f"Image file is empty: {image_path}")
    return base64.b64encode(image_bytes).decode("utf-8")


def _shape(label, points, shape_type, description=""):
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": description,
        "shape_type": shape_type,
        "flags": {},
        "mask": None,
    }


def create_labelme_json(
    image_path: str | Path,
    polygons: list[list[list[float]]] | None = None,
    label: str = "leaf",
    description: str = "",
    version: str = "5.11.3",
    embed_image_data: bool = True,
    flags: dict | None = None,
    points: list[list[float]] | None = None,
    point_label: str | None = None,
    point_description: str | None = None,
):
    """Create LabelMe JSON supporting polygon and point annotations."""
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        width, height = image.size

    image_data = encode_image_base64(image_path) if embed_image_data else None

    shapes = []

    for polygon in polygons or []:
        shapes.append(_shape(label, polygon, "polygon", description))

    if points:
        effective_point_label = point_label or label
        effective_point_description = (
            description if point_description is None else point_description
        )

        for point in points:
            if len(point) != 2:
                raise ValueError(
                    f"Point must contain exactly [x, y], got: {point}"
                )

            shapes.append(
                _shape(
                    effective_point_label,
                    [[float(point[0]), float(point[1])]],
                    "point",
                    effective_point_description,
                )
            )

    return {
        "version": version,
        "flags": flags if flags is not None else {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def save_labelme_json(
    output_path: str | Path,
    image_path: str | Path,
    polygons: list[list[list[float]]] | None = None,
    label: str = "leaf",
    description: str = "",
    version: str = "5.11.3",
    embed_image_data: bool = True,
    flags: dict | None = None,
    points: list[list[float]] | None = None,
    point_label: str | None = None,
    point_description: str | None = None,
):
    data = create_labelme_json(
        image_path=image_path,
        polygons=polygons,
        label=label,
        description=description,
        version=version,
        embed_image_data=embed_image_data,
        flags=flags,
        points=points,
        point_label=point_label,
        point_description=point_description,
    )

    if embed_image_data:
        image_data = data["imageData"]
        if not isinstance(image_data, str) or not image_data:
            raise RuntimeError("LabelMe imageData is missing or empty.")
        try:
            base64.b64decode(image_data, validate=True)
        except Exception as exc:
            raise RuntimeError("LabelMe imageData is invalid Base64.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Backward-compatible alias.
write_labelme_json = save_labelme_json
