import argparse
import json
import shutil
from pathlib import Path

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

LEAF_TAG = "leaf"
NEGATIVE_TAG = "negative"


def find_json(image_path):
    """Find the LabelMe JSON corresponding to an image."""
    json_path = image_path.with_suffix(".json")
    return json_path if json_path.exists() else None


def load_labelme_json(json_path):
    """Load a LabelMe JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_annotation(json_path):
    """
    Classify an annotated image.

    Rules:
      1. polygon + tag/label == leaf     -> TRAIN_LEAF
      2. point   + tag/label == negative -> TRAIN_NEGATIVE
      3. anything else                  -> TEST

    If an image contains both a leaf polygon and a negative point,
    the leaf annotation takes precedence and the image is TRAIN_LEAF.
    """
    if json_path is None:
        return "TEST", None

    data = load_labelme_json(json_path)
    shapes = data.get("shapes", [])

    has_leaf_polygon = False
    has_negative_point = False

    for shape in shapes:
        shape_type = str(shape.get("shape_type", "")).lower()
        label = str(shape.get("label", "")).strip().lower()

        if shape_type == "polygon" and label == LEAF_TAG:
            has_leaf_polygon = True

        if shape_type == "point" and label == NEGATIVE_TAG:
            has_negative_point = True

    if has_leaf_polygon:
        return "TRAIN_LEAF", data

    if has_negative_point:
        return "TRAIN_NEGATIVE", data

    return "TEST", data


def create_leaf_mask_from_json(json_data, image_width, image_height):
    """
    Create a binary leaf mask.

    Only shapes satisfying:
        shape_type == polygon
        label == leaf

    are included in the mask.

    Mask:
        0   = background
        255 = leaf
    """
    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in json_data.get("shapes", []):
        shape_type = str(shape.get("shape_type", "")).lower()
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points", [])

        if shape_type != "polygon" or label != LEAF_TAG:
            continue

        if len(points) < 3:
            continue

        polygon = [
            (float(point[0]), float(point[1]))
            for point in points
        ]

        draw.polygon(polygon, fill=255)

    return mask


def create_negative_mask(image_width, image_height):
    """Create an all-background mask for a negative image."""
    return Image.new("L", (image_width, image_height), 0)


def collect_images(input_dirs):
    """
    Recursively collect images from all input directories.

    Subdirectories are included.
    """
    image_files = []

    for input_dir in input_dirs:
        print(f"Scanning recursively: {input_dir}")

        if not input_dir.exists():
            print(f"WARNING: Directory does not exist: {input_dir}")
            continue

        if not input_dir.is_dir():
            print(f"WARNING: Not a directory: {input_dir}")
            continue

        for path in input_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ):
                image_files.append(path)

    return sorted(set(image_files))


def get_unique_destination(destination):
    """
    Prevent overwriting when multiple source folders contain the same
    filename.
    """
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1

    while True:
        new_path = destination.parent / f"{stem}_{counter}{suffix}"

        if not new_path.exists():
            return new_path

        counter += 1


def transfer_file(source, destination, operation):
    """Copy or move a file according to the selected operation."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if operation == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def process_train_image(
    image_path,
    json_path,
    json_data,
    train_images_dir,
    train_masks_dir,
    operation,
):
    """Process a leaf or negative training image."""
    with Image.open(image_path) as image:
        width, height = image.size

    output_image = get_unique_destination(
        train_images_dir / image_path.name
    )

    # JSON must use the same final stem as the image.
    output_json = output_image.with_suffix(".json")

    output_mask = train_masks_dir / f"{output_image.stem}.png"

    # Transfer image.
    transfer_file(
        image_path,
        output_image,
        operation,
    )

    # Transfer the corresponding JSON.
    transfer_file(
        json_path,
        output_json,
        operation,
    )

    # Create mask from the original JSON data.
    if json_data is not None:
        mask = create_leaf_mask_from_json(
            json_data,
            width,
            height,
        )
    else:
        mask = create_negative_mask(width, height)

    mask.save(output_mask)

    return output_image, output_json, output_mask


def process_test_image(
    image_path,
    test_images_dir,
    operation,
):
    """Transfer an image that does not belong to the training set."""
    output_image = get_unique_destination(
        test_images_dir / image_path.name
    )

    transfer_file(
        image_path,
        output_image,
        operation,
    )

    return output_image


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a leaf segmentation dataset from LabelMe images.\n\n"
            "Dataset organization:\n"
            "  data/1.train/images\n"
            "  data/1.train/masks\n"
            "  data/2.test/images\n\n"
            "Classification rules:\n"
            "  polygon + label/tag 'leaf'     -> train + leaf mask\n"
            "  point   + label/tag 'negative' -> train + zero mask\n"
            "  all other images               -> test"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Input directory. Can be specified multiple times. "
            "All subdirectories are scanned recursively."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Dataset output directory. The script creates:\n"
            "  1.train/images\n"
            "  1.train/masks\n"
            "  2.test/images"
        ),
    )

    parser.add_argument(
        "--operation",
        choices=["copy", "move"],
        default="copy",
        help=(
            "How source files are handled: copy or move. "
            "Default: copy."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Output structure
    # ---------------------------------------------------------

    train_images_dir = args.output_dir / "1.train" / "images"
    train_masks_dir = args.output_dir / "1.train" / "masks"
    test_images_dir = args.output_dir / "2.test" / "images"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_masks_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 75)
    print("LABELME -> LEAF DATASET CREATION")
    print("=" * 75)

    print()
    print("Input directories:")
    for directory in args.input_dir:
        print(f"  {directory}")

    print()
    print(f"Output directory : {args.output_dir}")
    print(f"Operation        : {args.operation}")

    print()
    print("Rules:")
    print("  polygon + leaf     -> TRAIN (leaf mask)")
    print("  point + negative   -> TRAIN (zero mask)")
    print("  everything else    -> TEST")
    print()

    # ---------------------------------------------------------
    # Scan
    # ---------------------------------------------------------

    image_files = collect_images(args.input_dir)

    print(f"Total images found: {len(image_files)}")
    print()

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    train_leaf = 0
    train_negative = 0
    test_images = 0
    skipped = 0
    failed = 0
    duplicate_names = 0

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):
        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path}"
        )

        try:
            json_path = find_json(image_path)

            # No JSON means it cannot be a leaf/negative annotation.
            # Therefore it goes to TEST.
            if json_path is None:
                output_image = process_test_image(
                    image_path,
                    test_images_dir,
                    args.operation,
                )

                test_images += 1

                print(
                    f"  TEST (no JSON) -> {output_image.name}"
                )
                continue

            json_data = load_labelme_json(json_path)

            category, _ = classify_annotation(json_path)

            # -------------------------------------------------
            # TRAIN: leaf polygon
            # -------------------------------------------------

            if category == "TRAIN_LEAF":
                before = train_images_dir / image_path.name

                output_image, output_json, output_mask = (
                    process_train_image(
                        image_path=image_path,
                        json_path=json_path,
                        json_data=json_data,
                        train_images_dir=train_images_dir,
                        train_masks_dir=train_masks_dir,
                        operation=args.operation,
                    )
                )

                if output_image != before:
                    duplicate_names += 1

                train_leaf += 1

                print(
                    f"  TRAIN/LEAF -> {output_image.name}"
                )
                print(
                    f"    JSON  : {output_json.name}"
                )
                print(
                    f"    MASK  : {output_mask.name}"
                )

            # -------------------------------------------------
            # TRAIN: negative point
            # -------------------------------------------------

            elif category == "TRAIN_NEGATIVE":
                before = train_images_dir / image_path.name

                output_image, output_json, output_mask = (
                    process_train_image(
                        image_path=image_path,
                        json_path=json_path,
                        json_data=None,
                        train_images_dir=train_images_dir,
                        train_masks_dir=train_masks_dir,
                        operation=args.operation,
                    )
                )

                if output_image != before:
                    duplicate_names += 1

                train_negative += 1

                print(
                    f"  TRAIN/NEGATIVE -> {output_image.name}"
                )
                print(
                    f"    JSON  : {output_json.name}"
                )
                print(
                    f"    MASK  : {output_mask.name} (all zero)"
                )

            # -------------------------------------------------
            # TEST
            # -------------------------------------------------

            else:
                output_image = process_test_image(
                    image_path,
                    test_images_dir,
                    args.operation,
                )

                test_images += 1

                print(
                    f"  TEST -> {output_image.name}"
                )

        except Exception as e:
            failed += 1

            print(f"  ERROR: {e}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(f"Images found       : {len(image_files)}")
    print(f"Train - leaf       : {train_leaf}")
    print(f"Train - negative   : {train_negative}")
    print(f"Train total        : {train_leaf + train_negative}")
    print(f"Test               : {test_images}")
    print(f"Failed             : {failed}")
    print(f"Duplicate renamed  : {duplicate_names}")

    print()
    print("Dataset structure:")
    print(f"  {train_images_dir}")
    print(f"  {train_masks_dir}")
    print(f"  {test_images_dir}")

    print()
    print("Important:")
    print("  * Training JSON files are copied/moved next to training images.")
    print("  * Negative training masks are completely black (all zero).")
    print("  * Test images do not get masks or JSON files.")
    print("  * Input subdirectories are scanned recursively.")
    print("=" * 75)


if __name__ == "__main__":
    main()
