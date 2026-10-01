#!/usr/bin/env python3
"""Create all-negative (black) masks for every image in a folder.

Requires Pillow: python -m pip install pillow
Example: python create_negative_masks.py "path/to/grass_backgrounds"
"""

import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Folder containing source images")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Mask output folder (default: IMAGE_DIR/masks)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace masks that already exist",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    if not image_dir.is_dir():
        parser.error(f"Image folder does not exist or is not a directory: {image_dir}")

    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir else image_dir / "masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resolved = output_dir.resolve()

    images = []
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Avoid treating generated masks (or any output tree inside the input)
        # as source images if this script is run again.
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        images.append(path)

    created = skipped = failed = 0
    for image_path in sorted(images):
        relative = image_path.relative_to(image_dir)
        mask_path = (output_dir / relative).with_suffix(".png")
        if mask_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            with Image.open(image_path) as image:
                size = image.size
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", size, color=0).save(mask_path, format="PNG")
            created += 1
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"Failed: {image_path} ({exc})")

    print(f"Images found: {len(images)}")
    print(f"Masks created: {created}")
    print(f"Existing masks skipped: {skipped}")
    print(f"Failures: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
