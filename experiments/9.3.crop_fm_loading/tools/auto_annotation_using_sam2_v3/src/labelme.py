from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image


def encode_image_base64(image_path: str | Path) -> str:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    if not image_bytes:
        raise RuntimeError(
            f"Image file is empty: {image_path}"
        )

    return base64.b64encode(image_bytes).decode("utf-8")


def create_labelme_json(
    image_path: str | Path,
    polygons: list[list[list[float]]],
    label: str = "leaf",
    description: str = "",
    version: str = "5.11.3",
    embed_image_data: bool = True,
    flags: dict | None = None,
):
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        width, height = image.size

    image_data = (
        encode_image_base64(image_path)
        if embed_image_data
        else None
    )

    shapes = []

    for points in polygons:
        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": description,
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
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
    polygons: list[list[list[float]]],
    label: str = "leaf",
    description: str = "",
    version: str = "5.11.3",
    embed_image_data: bool = True,
    flags: dict | None = None,
):
    data = create_labelme_json(
        image_path=image_path,
        polygons=polygons,
        label=label,
        description=description,
        version=version,
        embed_image_data=embed_image_data,
        flags=flags,
    )

    if embed_image_data:
        image_data = data["imageData"]

        if not isinstance(image_data, str):
            raise RuntimeError(
                "LabelMe imageData is not a Base64 string."
            )

        if not image_data:
            raise RuntimeError(
                "LabelMe imageData is empty."
            )

        # Validate the Base64 payload before writing JSON.
        try:
            base64.b64decode(
                image_data,
                validate=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "LabelMe imageData is invalid Base64."
            ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# Backward-compatible alias.
# Older versions of refine_batch.py may import this name.
write_labelme_json = save_labelme_json
