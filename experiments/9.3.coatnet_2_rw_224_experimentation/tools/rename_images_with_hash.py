"""Rename images to <sha256>, optionally renaming matching masks.

Images and masks are matched by their filename stem (case-insensitive). The
mask keeps its own extension but receives the renamed image's new stem.
"""

import argparse
import hashlib
import os
import uuid
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_renames(image_dir: Path, mask_dir: Path | None):
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")

    masks_by_stem = {}
    if mask_dir:
        for path in mask_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                masks_by_stem.setdefault(path.stem.casefold(), []).append(path)

    plan = []
    used_stems = set()
    for image in images:
        # Use the first half of the SHA-256 hex digest to keep filenames compact.
        digest = sha256_file(image)[:32]
        stem = digest
        duplicate_index = 2
        while stem.casefold() in used_stems:
            stem = f"{digest}_{duplicate_index}"
            duplicate_index += 1
        used_stems.add(stem.casefold())
        new_name = f"{stem}{image.suffix}"
        mask = None
        if mask_dir:
            matches = masks_by_stem.get(image.stem.casefold(), [])
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple masks match {image.name}: "
                    + ", ".join(p.name for p in matches)
                )
            if matches:
                mask = matches[0]
        plan.append((image, image.with_name(new_name), mask,
                     mask.with_name(f"{stem}{mask.suffix}") if mask else None))

    # Fail before changing anything if a target would overwrite an unrelated file.
    sources = {p.resolve() for row in plan for p in (row[0], row[2]) if p}
    targets = [p for row in plan for p in (row[1], row[3]) if p]
    target_paths = [p.resolve() for p in targets]
    if len(target_paths) != len(set(target_paths)):
        raise ValueError("Two files would receive the same destination name")
    for target in targets:
        if target.exists() and target.resolve() not in sources:
            raise FileExistsError(f"Destination already exists: {target}")
    return plan


def apply_plan(plan, dry_run=False):
    pairs = [(src, dst) for row in plan for src, dst in ((row[0], row[1]), (row[2], row[3])) if src]
    for src, dst in pairs:
        print(f"{src.name} -> {dst.name}")
    if dry_run or not pairs:
        return

    # Stage all files first so swaps and overlapping old/new names are safe.
    staged = []
    try:
        for src, dst in pairs:
            temporary = src.with_name(f".rename_tmp_{uuid.uuid4().hex}{src.suffix}")
            os.rename(src, temporary)
            staged.append((temporary, dst))
        for temporary, dst in staged:
            os.rename(temporary, dst)
    except Exception:
        # Restore files that have not reached their destination where possible.
        for temporary, dst in reversed(staged):
            if temporary.exists():
                original = next(src for src, candidate in pairs if candidate == dst)
                if not original.exists():
                    os.rename(temporary, original)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Rename images as <SHA-256> and optionally rename matching masks."
    )
    parser.add_argument("image_folder", nargs="?", help="Folder containing images")
    parser.add_argument("--mask-folder", help="Optional folder containing masks")
    parser.add_argument("--dry-run", action="store_true", help="Show planned names without changing files")
    args = parser.parse_args()

    image_value = args.image_folder or input("Image folder path: ").strip().strip('"')
    mask_value = args.mask_folder
    if mask_value is None:
        mask_value = input("Mask folder path (leave blank to skip masks): ").strip().strip('"')
    image_dir = Path(image_value).expanduser().resolve()
    mask_dir = Path(mask_value).expanduser().resolve() if mask_value else None
    if not image_dir.is_dir():
        parser.error(f"Image folder does not exist or is not a directory: {image_dir}")
    if mask_dir and not mask_dir.is_dir():
        parser.error(f"Mask folder does not exist or is not a directory: {mask_dir}")
    try:
        plan = plan_renames(image_dir, mask_dir)
        apply_plan(plan, args.dry_run)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
