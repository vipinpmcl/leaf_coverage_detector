import argparse
import shutil
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


def collect_images(input_dirs, output_dir):
    """Recursively collect supported images, excluding the output tree."""
    images = set()
    output_resolved = output_dir.resolve()
    for input_dir in input_dirs:
        print(f"Scanning recursively: {input_dir}")
        if not input_dir.exists():
            print(f"WARNING: Directory does not exist: {input_dir}")
            continue
        if not input_dir.is_dir():
            print(f"WARNING: Not a directory: {input_dir}")
            continue
        for path in input_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            try:
                path.resolve().relative_to(output_resolved)
                continue
            except ValueError:
                pass
            images.add(path)
    return sorted(images)


def build_mask_index(mask_dirs, output_dir):
    """Index supported mask files by case-insensitive filename stem."""
    index = {}
    output_resolved = output_dir.resolve()
    for mask_dir in mask_dirs:
        print(f"Indexing masks recursively: {mask_dir}")
        if not mask_dir.exists():
            print(f"WARNING: Mask directory does not exist: {mask_dir}")
            continue
        if not mask_dir.is_dir():
            print(f"WARNING: Not a mask directory: {mask_dir}")
            continue
        for path in mask_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            try:
                path.resolve().relative_to(output_resolved)
                continue
            except ValueError:
                pass
            index.setdefault(path.stem.casefold(), []).append(path)
    for candidates in index.values():
        candidates.sort()
    return index


def find_mask(image_path, mask_index):
    """Prefer another same-folder file, then the configured mask directory."""
    same_folder = sorted(
        candidate for candidate in image_path.parent.glob(image_path.stem + ".*")
        if candidate.is_file()
        and candidate != image_path
        and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if same_folder:
        return same_folder[0]
    candidates = [candidate for candidate in mask_index.get(image_path.stem.casefold(), []) if candidate != image_path]
    return candidates[0] if candidates else None


def transfer_file(source, destination, operation):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if operation == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def unique_pair_destinations(image_path, mask_path, images_dir, masks_dir):
    """Choose a shared output stem that cannot overwrite either file."""
    suffix = 0
    while True:
        stem = image_path.stem if suffix == 0 else f"{image_path.stem}_{suffix}"
        image_dest = images_dir / f"{stem}{image_path.suffix}"
        mask_dest = masks_dir / f"{stem}{mask_path.suffix}"
        if not image_dest.exists() and not mask_dest.exists():
            return image_dest, mask_dest, suffix > 0
        suffix += 1


def unique_image_destination(image_path, images_dir):
    suffix = 0
    while True:
        stem = image_path.stem if suffix == 0 else f"{image_path.stem}_{suffix}"
        destination = images_dir / f"{stem}{image_path.suffix}"
        if not destination.exists():
            return destination, suffix > 0
        suffix += 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an image-mask dataset by matching existing masks to images by filename stem.\n\n"
            "A matching mask sends the image and mask to 1.train/images and 1.train/masks.\n"
            "Images without a matching mask go to 2.test/images. JSON files are ignored."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, action="append", required=True,
        help="Image directory; may be repeated. All subdirectories are scanned recursively.",
    )
    parser.add_argument(
        "--mask-dir", type=Path, action="append", default=[],
        help="Mask directory; may be repeated. Masks are matched recursively by filename stem. Same-folder masks are also checked.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Dataset output directory (creates 1.train/images, 1.train/masks, and 2.test/images).",
    )
    parser.add_argument(
        "--operation", choices=("copy", "move"), default="copy",
        help="Copy or move source images and matched masks. Default: copy.",
    )
    args = parser.parse_args()

    train_images_dir = args.output_dir / "1.train" / "images"
    train_masks_dir = args.output_dir / "1.train" / "masks"
    test_images_dir = args.output_dir / "2.test" / "images"
    for directory in (train_images_dir, train_masks_dir, test_images_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("IMAGE-MASK DATASET CREATION")
    print("=" * 75)
    print("Input directories:")
    for directory in args.input_dir:
        print(f"  {directory}")
    print("Mask directories:")
    if args.mask_dir:
        for directory in args.mask_dir:
            print(f"  {directory}")
    else:
        print("  (none; same-folder mask lookup only)")
    print(f"Output directory : {args.output_dir}")
    print(f"Operation        : {args.operation}")
    print("Rules: matching mask -> TRAIN image + mask; no mask -> TEST image")

    image_files = collect_images(args.input_dir, args.output_dir)
    mask_index = build_mask_index(args.mask_dir, args.output_dir)
    print(f"\nTotal images found: {len(image_files)}\n")

    train_images = train_masks = test_images = failed = duplicate_names = 0
    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] {image_path}")
        try:
            mask_path = find_mask(image_path, mask_index)
            if mask_path is not None:
                image_dest, mask_dest, renamed = unique_pair_destinations(
                    image_path, mask_path, train_images_dir, train_masks_dir
                )
                transfer_file(image_path, image_dest, args.operation)
                transfer_file(mask_path, mask_dest, args.operation)
                train_images += 1
                train_masks += 1
                duplicate_names += int(renamed)
                print(f"  TRAIN -> image: {image_dest.name}; mask: {mask_dest.name}")
                if mask_path.stem.casefold() != image_path.stem.casefold():
                    print(f"  Matched mask source: {mask_path}")
            else:
                image_dest, renamed = unique_image_destination(image_path, test_images_dir)
                transfer_file(image_path, image_dest, args.operation)
                test_images += 1
                duplicate_names += int(renamed)
                print(f"  TEST (no matching mask) -> {image_dest.name}")
        except Exception as error:
            failed += 1
            print(f"  ERROR: {error}")

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"Images found       : {len(image_files)}")
    print(f"Train images       : {train_images}")
    print(f"Train masks        : {train_masks}")
    print(f"Test images        : {test_images}")
    print(f"Failed             : {failed}")
    print(f"Duplicate renamed  : {duplicate_names}")
    print("\nDataset structure:")
    print(f"  {train_images_dir}")
    print(f"  {train_masks_dir}")
    print(f"  {test_images_dir}")
    print("\nJSON files are ignored and are not copied.")
    print("Image and matched mask filenames share the same output stem.")
    print("=" * 75)


if __name__ == "__main__":
    main()



