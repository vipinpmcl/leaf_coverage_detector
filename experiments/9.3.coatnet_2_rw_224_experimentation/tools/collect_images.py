"""Copy images from a folder tree into one output folder."""

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(source: Path, output: Path, dry_run: bool = False) -> int:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source folder does not exist: {source}")
    if source == output:
        raise ValueError("Output folder must be different from the source folder")

    # Exclude the output tree if it is nested under source, so reruns do not
    # pick up files copied during earlier runs.
    output_is_inside_source = output.is_relative_to(source)
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not (output_is_inside_source and path.is_relative_to(output))
    )

    reserved_names = {p.name.casefold() for p in output.iterdir() if p.is_file()} if output.exists() else set()
    copied = 0
    for image in images:
        name = image.name
        candidate = name
        index = 2
        while candidate.casefold() in reserved_names:
            candidate = f"{image.stem}_{index}{image.suffix}"
            index += 1
        reserved_names.add(candidate.casefold())
        destination = output / candidate
        print(f"{image} -> {destination}")
        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively copy supported images into a single output folder."
    )
    parser.add_argument("source", type=Path, help="Folder to search, including its subfolders")
    parser.add_argument("output", type=Path, help="Folder where images will be collected")
    parser.add_argument("--dry-run", action="store_true", help="List planned copies without copying")
    args = parser.parse_args()

    try:
        count = collect_images(args.source, args.output, args.dry_run)
    except (NotADirectoryError, ValueError) as error:
        parser.error(str(error))
    print(f"{'Found' if args.dry_run else 'Copied'} {count} image(s).")


if __name__ == "__main__":
    main()
