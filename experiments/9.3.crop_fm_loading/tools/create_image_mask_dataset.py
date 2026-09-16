import argparse
import shutil
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def find_json(image_path):
    """Find JSON corresponding to an image."""

    json_path = image_path.with_suffix(".json")

    if json_path.exists():
        return json_path

    return None


def create_mask_from_labelme_json(
    json_path,
    image_width,
    image_height,
):
    """
    Create binary mask from LabelMe JSON.

    Supported:
        polygon
        rectangle
        circle
        line
    """

    mask = Image.new(
        "L",
        (image_width, image_height),
        0,
    )

    draw = ImageDraw.Draw(mask)

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    for shape in shapes:

        shape_type = shape.get(
            "shape_type",
            "polygon",
        )

        points = shape.get(
            "points",
            [],
        )

        if not points:
            continue

        # -----------------------------------------------------
        # Polygon
        # -----------------------------------------------------

        if shape_type == "polygon":

            polygon = [
                (
                    float(point[0]),
                    float(point[1]),
                )
                for point in points
            ]

            if len(polygon) >= 3:
                draw.polygon(
                    polygon,
                    fill=255,
                )

        # -----------------------------------------------------
        # Rectangle
        # -----------------------------------------------------

        elif shape_type == "rectangle":

            if len(points) >= 2:

                x1, y1 = points[0]
                x2, y2 = points[1]

                draw.rectangle(
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    fill=255,
                )

        # -----------------------------------------------------
        # Circle
        # -----------------------------------------------------

        elif shape_type == "circle":

            if len(points) >= 2:

                cx, cy = points[0]
                px, py = points[1]

                radius = (
                    (px - cx) ** 2
                    + (py - cy) ** 2
                ) ** 0.5

                draw.ellipse(
                    [
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                    ],
                    fill=255,
                )

        # -----------------------------------------------------
        # Line
        # -----------------------------------------------------

        elif shape_type == "line":

            if len(points) >= 2:

                line = [
                    (
                        float(point[0]),
                        float(point[1]),
                    )
                    for point in points
                ]

                draw.line(
                    line,
                    fill=255,
                    width=1,
                )

        else:

            print(
                f"WARNING: Unsupported shape type "
                f"'{shape_type}' in {json_path.name}"
            )

    return mask


def collect_images(input_dirs):
    """
    Collect images from multiple input directories.
    """

    image_files = []

    for input_dir in input_dirs:

        print(
            f"Scanning: {input_dir}"
        )

        if not input_dir.exists():

            print(
                f"WARNING: Directory does not exist: "
                f"{input_dir}"
            )

            continue

        if not input_dir.is_dir():

            print(
                f"WARNING: Not a directory: "
                f"{input_dir}"
            )

            continue

        for path in input_dir.iterdir():

            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
            ):
                image_files.append(path)

    return sorted(image_files)


def get_unique_destination(
    destination,
    source_path,
):
    """
    Prevent overwriting files when two input folders
    contain images with the same filename.

    Example:

        image.jpg
        image_1.jpg
        image_2.jpg
    """

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    counter = 1

    while True:

        new_name = (
            f"{stem}_{counter}{suffix}"
        )

        new_path = (
            destination.parent /
            new_name
        )

        if not new_path.exists():
            return new_path

        counter += 1


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a combined image/mask dataset "
            "from multiple input folders."
        )
    )

    # ---------------------------------------------------------
    # Multiple input directories
    # ---------------------------------------------------------

    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Input directory containing images and JSON files. "
            "Can be specified multiple times."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output dataset directory. "
            "images/ and masks/ will be created here."
        ),
    )

    args = parser.parse_args()

    input_dirs = args.input_dir
    output_dir = args.output_dir

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    images_dir = (
        output_dir / "images"
    )

    masks_dir = (
        output_dir / "masks"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Scan input folders
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINED IMAGE + MASK DATASET CREATION")
    print("=" * 70)

    print()
    print("Input directories:")

    for directory in input_dirs:
        print(f"  {directory}")

    print()
    print(
        f"Output directory: {output_dir}"
    )

    print()

    image_files = collect_images(
        input_dirs
    )

    print()
    print(
        f"Total images found: "
        f"{len(image_files)}"
    )

    print()

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    processed = 0
    skipped_no_json = 0
    failed = 0
    duplicate_names = 0

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    for image_path in image_files:

        json_path = find_json(
            image_path
        )

        # -----------------------------------------------------
        # JSON not found
        # -----------------------------------------------------

        if json_path is None:

            print(
                f"SKIP - JSON not found: "
                f"{image_path}"
            )

            skipped_no_json += 1
            continue

        try:

            # -------------------------------------------------
            # Open image
            # -------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                width, height = image.size

            # -------------------------------------------------
            # Create mask
            # -------------------------------------------------

            mask = create_mask_from_labelme_json(
                json_path=json_path,
                image_width=width,
                image_height=height,
            )

            # -------------------------------------------------
            # Destination image
            # -------------------------------------------------

            output_image = (
                images_dir /
                image_path.name
            )

            # -------------------------------------------------
            # Handle duplicate filenames
            # -------------------------------------------------

            if output_image.exists():

                duplicate_names += 1

                output_image = get_unique_destination(
                    output_image,
                    image_path,
                )

            # -------------------------------------------------
            # Mask uses SAME final stem
            # -------------------------------------------------

            output_mask = (
                masks_dir /
                f"{output_image.stem}.png"
            )

            # -------------------------------------------------
            # Copy image
            # -------------------------------------------------

            shutil.copy2(
                image_path,
                output_image,
            )

            # -------------------------------------------------
            # Save mask
            # -------------------------------------------------

            mask.save(
                output_mask
            )

            processed += 1

            print(
                f"OK    {image_path.name}"
                f" -> "
                f"{output_image.name}"
            )

        except Exception as e:

            failed += 1

            print(
                f"ERROR: {image_path}"
            )

            print(
                f"       {e}"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Input directories  : "
        f"{len(input_dirs)}"
    )

    print(
        f"Images found       : "
        f"{len(image_files)}"
    )

    print(
        f"Processed          : "
        f"{processed}"
    )

    print(
        f"No JSON            : "
        f"{skipped_no_json}"
    )

    print(
        f"Failed             : "
        f"{failed}"
    )

    print(
        f"Duplicate names    : "
        f"{duplicate_names}"
    )

    print()

    print(
        f"Images: {images_dir}"
    )

    print(
        f"Masks : {masks_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()