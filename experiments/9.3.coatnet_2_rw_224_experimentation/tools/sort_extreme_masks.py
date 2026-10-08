#!/usr/bin/env python3
"""Move masks that are at least 95% pure black or pure white.

Example:
    python sort_extreme_masks.py path/to/masks --output-dir path/to/sorted_masks

Files are moved into ``OUTPUT_DIR/mostly_black`` or ``OUTPUT_DIR/mostly_white``
with their subdirectory structure preserved. Pixels are checked after conversion
to 8-bit grayscale; only values exactly 0 and 255 count as pure black and white.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_dir", type=Path, help="Folder containing masks")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Destination folder (default: MASK_DIR/extreme_masks)",
    )
    parser.add_argument(
        "--threshold", type=float, default=95.0,
        help="Minimum percent pure black or pure white pixels (default: 95)",
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir.expanduser().resolve()
    if not mask_dir.is_dir():
        parser.error(f"Mask folder does not exist or is not a directory: {mask_dir}")
    if not 0 < args.threshold <= 100:
        parser.error("--threshold must be greater than 0 and at most 100")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else mask_dir / "extreme_masks"
    )
    black_dir = output_dir / "mostly_black"
    white_dir = output_dir / "mostly_white"
    output_resolved = output_dir.resolve()

    moved_black = moved_white = skipped = failed = 0
    for mask_path in sorted(mask_dir.rglob("*")):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Prevent reprocessing files already in the destination on subsequent runs.
        try:
            mask_path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass

        try:
            with Image.open(mask_path) as image:
                pixels = image.convert("L")
                total = pixels.width * pixels.height
                if total == 0:
                    skipped += 1
                    continue
                histogram = pixels.histogram()
            black_percent = histogram[0] * 100 / total
            white_percent = histogram[255] * 100 / total

            if black_percent >= args.threshold:
                target_root = black_dir
                category = "black"
            elif white_percent >= args.threshold:
                target_root = white_dir
                category = "white"
            else:
                skipped += 1
                continue

            target = target_root / mask_path.relative_to(mask_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Destination already exists: {target}")
            shutil.move(str(mask_path), str(target))
            if category == "black":
                moved_black += 1
            else:
                moved_white += 1
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"Failed: {mask_path} ({exc})")

    print(f"Moved mostly black masks: {moved_black}")
    print(f"Moved mostly white masks: {moved_white}")
    print(f"Skipped (below threshold or empty): {skipped}")
    print(f"Failures: {failed}")
    print(f"Output: {output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
