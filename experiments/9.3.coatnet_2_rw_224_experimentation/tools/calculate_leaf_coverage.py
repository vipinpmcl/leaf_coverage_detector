import argparse
from pathlib import Path
import csv
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def calculate_coverage(mask_path):
    """
    Calculate leaf coverage from a binary mask.

    Leaf pixels = pixels with value > 0
    Coverage = leaf pixels / total pixels * 100
    """

    mask = np.array(Image.open(mask_path).convert("L"))

    total_pixels = mask.size
    leaf_pixels = np.count_nonzero(mask > 0)

    coverage_percent = (
        leaf_pixels / total_pixels * 100
        if total_pixels > 0
        else 0.0
    )

    height, width = mask.shape

    return {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "leaf_pixels": leaf_pixels,
        "leaf_coverage_percent": coverage_percent,
    }


def save_coverage_txt(folder, result):
    """
    Save coverage information inside each image prediction folder.
    """

    output_file = folder / "leaf_coverage.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"Leaf Coverage: "
            f"{result['leaf_coverage_percent']:.4f}%\n"
        )

        f.write(
            f"Leaf Pixels: "
            f"{result['leaf_pixels']}\n"
        )

        f.write(
            f"Total Pixels: "
            f"{result['total_pixels']}\n"
        )

        f.write(
            f"Image Width: "
            f"{result['width']}\n"
        )

        f.write(
            f"Image Height: "
            f"{result['height']}\n"
        )

    return output_file


def find_prediction_folders(input_dir):
    """
    Find folders containing mask.png.
    """

    prediction_folders = []

    for folder in input_dir.iterdir():

        if not folder.is_dir():
            continue

        mask_path = folder / "mask.png"

        if mask_path.exists():
            prediction_folders.append(folder)

    return sorted(prediction_folders)


def main():

    parser = argparse.ArgumentParser(
        description="Calculate leaf coverage from prediction masks."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Folder containing per-image prediction folders. "
            "Each folder must contain mask.png."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Folder where the aggregate CSV file "
            "leaf_coverage.csv will be stored."
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # ---------------------------------------------------------
    # Validate input directory
    # ---------------------------------------------------------

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist:")
        print(f"       {input_dir}")
        return

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory:")
        print(f"       {input_dir}")
        return

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV file will automatically be created inside output-dir
    output_csv = output_dir / "leaf_coverage.csv"

    # ---------------------------------------------------------
    # Find prediction folders
    # ---------------------------------------------------------

    prediction_folders = find_prediction_folders(input_dir)

    if not prediction_folders:
        print("No prediction folders containing mask.png were found.")
        return

    print()
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"CSV file        : {output_csv}")
    print(f"Images found    : {len(prediction_folders)}")
    print()

    results = []

    # ---------------------------------------------------------
    # Process each image folder
    # ---------------------------------------------------------

    for folder in prediction_folders:

        mask_path = folder / "mask.png"

        try:

            result = calculate_coverage(mask_path)

            # Save TXT inside the image folder
            txt_path = save_coverage_txt(
                folder,
                result,
            )

            results.append(
                {
                    "image": folder.name,
                    "mask_path": str(mask_path),
                    "width": result["width"],
                    "height": result["height"],
                    "total_pixels": result["total_pixels"],
                    "leaf_pixels": result["leaf_pixels"],
                    "leaf_coverage_percent": (
                        result["leaf_coverage_percent"]
                    ),
                }
            )

            print(
                f"{folder.name:<40} "
                f"{result['leaf_coverage_percent']:>8.4f}%"
            )

            print(
                f"  Saved: {txt_path}"
            )

        except Exception as e:

            print(
                f"ERROR processing {folder.name}: {e}"
            )

    # ---------------------------------------------------------
    # Save aggregate CSV
    # ---------------------------------------------------------

    if results:

        fieldnames = [
            "image",
            "mask_path",
            "width",
            "height",
            "total_pixels",
            "leaf_pixels",
            "leaf_coverage_percent",
        ]

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        coverage_values = [
            r["leaf_coverage_percent"]
            for r in results
        ]

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(
            f"Images processed : {len(results)}"
        )

        print(
            f"Mean coverage    : "
            f"{np.mean(coverage_values):.4f}%"
        )

        print(
            f"Minimum coverage : "
            f"{np.min(coverage_values):.4f}%"
        )

        print(
            f"Maximum coverage : "
            f"{np.max(coverage_values):.4f}%"
        )

        print()
        print(f"CSV saved        : {output_csv}")
        print("=" * 60)

    else:

        print()
        print("No images were successfully processed.")


if __name__ == "__main__":
    main()