"""Check that image and mask directories contain matching files.

By default, files are paired by their relative path without the extension. This
allows, for example, ``images/leaf_01.jpg`` to pair with ``masks/leaf_01.png``.
Use ``--by-stem`` when images and masks live in different subdirectory layouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"


def parse_extensions(value: str) -> set[str]:
    extensions = {item.strip().lower() for item in value.split(",") if item.strip()}
    return {ext if ext.startswith(".") else f".{ext}" for ext in extensions}


def collect_files(directory: Path, extensions: set[str]) -> tuple[list[Path], list[Path]]:
    supported: list[Path] = []
    unsupported: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        (supported if path.suffix.lower() in extensions else unsupported).append(path)
    return supported, unsupported


def key_for(path: Path, root: Path, by_stem: bool) -> str:
    relative = path.relative_to(root)
    if by_stem:
        return relative.stem
    return str(relative.with_suffix("")).replace("\\", "/")


def describe(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory containing images")
    parser.add_argument("masks", type=Path, help="Directory containing masks")
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help=f"Comma-separated file extensions to check (default: {DEFAULT_EXTENSIONS})",
    )
    parser.add_argument(
        "--by-stem",
        action="store_true",
        help="Pair by basename stem only, ignoring subdirectory paths",
    )
    parser.add_argument(
        "--remove-unpaired",
        action="store_true",
        help="Delete images without masks and masks without images after listing them",
    )
    args = parser.parse_args()

    images_root = args.images.expanduser().resolve()
    masks_root = args.masks.expanduser().resolve()
    for label, directory in (("Image", images_root), ("Mask", masks_root)):
        if not directory.is_dir():
            parser.error(f"{label} directory does not exist or is not a directory: {directory}")

    extensions = parse_extensions(args.extensions)
    if not extensions:
        parser.error("At least one extension must be provided")

    images, unsupported_images = collect_files(images_root, extensions)
    masks, unsupported_masks = collect_files(masks_root, extensions)
    image_map: dict[str, list[Path]] = defaultdict(list)
    mask_map: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        image_map[key_for(path, images_root, args.by_stem)].append(path)
    for path in masks:
        mask_map[key_for(path, masks_root, args.by_stem)].append(path)

    missing_masks = sorted(image_map.keys() - mask_map.keys())
    missing_images = sorted(mask_map.keys() - image_map.keys())
    duplicate_images = {key: paths for key, paths in image_map.items() if len(paths) > 1}
    duplicate_masks = {key: paths for key, paths in mask_map.items() if len(paths) > 1}

    print(f"Images: {len(images)} supported, {len(unsupported_images)} unsupported")
    print(f"Masks:  {len(masks)} supported, {len(unsupported_masks)} unsupported")
    print(f"Matching rule: {'filename stem' if args.by_stem else 'relative path and filename stem'}")

    if missing_masks:
        print(f"\nImages without masks ({len(missing_masks)}):")
        for key in missing_masks:
            print(f"  {key}  [image: {describe(image_map[key][0], images_root)}]")
    if missing_images:
        print(f"\nMasks without images ({len(missing_images)}):")
        for key in missing_images:
            print(f"  {key}  [mask: {describe(mask_map[key][0], masks_root)}]")

    if args.remove_unpaired and (missing_masks or missing_images):
        removed_images = [path for key in missing_masks for path in image_map[key]]
        removed_masks = [path for key in missing_images for path in mask_map[key]]
        for path in removed_images + removed_masks:
            path.unlink()
        print(
            f"\nRemoved {len(removed_images)} unpaired image(s) and "
            f"{len(removed_masks)} unpaired mask(s)."
        )
        missing_masks = []
        missing_images = []
    if duplicate_images:
        print(f"\nDuplicate image keys ({len(duplicate_images)}):")
        for key, paths in sorted(duplicate_images.items()):
            print(f"  {key}: {', '.join(describe(path, images_root) for path in paths)}")
    if duplicate_masks:
        print(f"\nDuplicate mask keys ({len(duplicate_masks)}):")
        for key, paths in sorted(duplicate_masks.items()):
            print(f"  {key}: {', '.join(describe(path, masks_root) for path in paths)}")

    if unsupported_images or unsupported_masks:
        print("\nUnsupported files (ignored):")
        for path in unsupported_images:
            print(f"  image: {describe(path, images_root)}")
        for path in unsupported_masks:
            print(f"  mask:  {describe(path, masks_root)}")

    if missing_masks or missing_images or duplicate_images or duplicate_masks:
        print("\nResult: inconsistencies found.")
        return 1
    print("\nResult: every image has exactly one matching mask, and every mask has an image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
