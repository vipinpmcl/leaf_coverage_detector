"""
Cluster images based on leaf coverage percentage.

Input:
    leaf_coverage.csv

The CSV must contain:
    image
    leaf_coverage_percent

The script finds the corresponding original images and
copies them into cluster-specific folders.

"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

def find_image(image_dir, image_name):
    """
    Robustly locate an image from the filename stored in the CSV.
    """

    image_name = str(image_name).strip()

    # Remove accidental surrounding quotes
    image_name = image_name.strip('"').strip("'")

    # ---------------------------------------------------------
    # Case 1: CSV contains an absolute path
    # ---------------------------------------------------------

    csv_path = Path(image_name)

    if csv_path.is_absolute() and csv_path.exists():
        return csv_path

    # ---------------------------------------------------------
    # Case 2: Exact filename inside image directory
    # ---------------------------------------------------------

    exact_path = image_dir / image_name

    if exact_path.is_file():
        return exact_path

    # ---------------------------------------------------------
    # Case 3: CSV contains a path, but only filename is needed
    # ---------------------------------------------------------

    filename = Path(image_name).name

    exact_path = image_dir / filename

    if exact_path.is_file():
        return exact_path

    # ---------------------------------------------------------
    # Case 4: Search recursively
    # ---------------------------------------------------------

    matches = list(
        image_dir.rglob(filename)
    )

    for match in matches:

        if (
            match.is_file()
            and match.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            return match

    # ---------------------------------------------------------
    # Case 5: Filename without extension
    # ---------------------------------------------------------

    stem = Path(filename).stem

    for ext in SUPPORTED_EXTENSIONS:

        candidate = image_dir / f"{stem}{ext}"

        if candidate.is_file():
            return candidate

    # Recursive search for stem
    for match in image_dir.rglob("*"):

        if not match.is_file():
            continue

        if match.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if match.stem == stem:
            return match

    return None

def main():

    parser = argparse.ArgumentParser(
        description="Cluster images based on leaf coverage percentage."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to leaf_coverage.csv",
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing original images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cluster folders will be created",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of coverage clusters",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating links",
    )

    args = parser.parse_args()

    csv_path = args.csv
    image_dir = args.image_dir
    output_dir = args.output_dir
    n_clusters = args.n_clusters

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Image path is not a directory: {image_dir}"
        )

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    # ---------------------------------------------------------
    # Load CSV
    # ---------------------------------------------------------

    df = pd.read_csv(csv_path)

    required_columns = [
        "image",
        "leaf_coverage_percent",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' "
                f"not found in CSV."
            )

    # ---------------------------------------------------------
    # Clean coverage values
    # ---------------------------------------------------------

    df["leaf_coverage_percent"] = pd.to_numeric(
        df["leaf_coverage_percent"],
        errors="coerce",
    )

    before_count = len(df)

    df = df.dropna(
        subset=[
            "leaf_coverage_percent"
        ]
    ).copy()

    removed_count = before_count - len(df)

    if removed_count > 0:
        print(
            f"Removed {removed_count} rows "
            f"with invalid coverage values."
        )

    if len(df) < n_clusters:
        raise ValueError(
            f"Only {len(df)} valid images available, "
            f"but n_clusters={n_clusters}."
        )

    # ---------------------------------------------------------
    # Prepare clustering data
    # ---------------------------------------------------------

    X = df[
        ["leaf_coverage_percent"]
    ].values

    # ---------------------------------------------------------
    # K-Means clustering
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("LEAF COVERAGE CLUSTERING")
    print("=" * 60)

    print(
        f"Images        : {len(df)}"
    )

    print(
        f"Coverage min  : "
        f"{df['leaf_coverage_percent'].min():.4f}%"
    )

    print(
        f"Coverage max  : "
        f"{df['leaf_coverage_percent'].max():.4f}%"
    )

    print(
        f"Coverage mean : "
        f"{df['leaf_coverage_percent'].mean():.4f}%"
    )

    print(
        f"Clusters      : {n_clusters}"
    )

    print()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    )

    df["cluster"] = kmeans.fit_predict(X)

    # ---------------------------------------------------------
    # Sort clusters by coverage
    #
    # K-Means cluster IDs are arbitrary.
    #
    # We reorder them so:
    #
    # cluster_0 = lowest coverage
    # cluster_1 = next lowest
    # ...
    # cluster_N = highest coverage
    # ---------------------------------------------------------

    cluster_centers = kmeans.cluster_centers_.flatten()

    sorted_cluster_ids = np.argsort(
        cluster_centers
    )

    cluster_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(
            sorted_cluster_ids
        )
    }

    df["cluster"] = df["cluster"].map(
        cluster_mapping
    )

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Copy images
    # ---------------------------------------------------------

    successful = 0
    missing = 0

    for cluster_id in range(n_clusters):

        cluster_dir = (
            output_dir /
            f"cluster_{cluster_id}"
        )

        cluster_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cluster_df = df[
            df["cluster"] == cluster_id
        ]

        for _, row in cluster_df.iterrows():

            image_name = str(
                row["image"]
            )

            source_image = find_image(
                image_dir,
                image_name,
            )

            if source_image is None:

                print(
                    f"WARNING: Image not found: "
                    f"{image_name}"
                )

                missing += 1
                continue

            destination = (
                cluster_dir /
                source_image.name
            )

            if not source_image.exists():
                print(
                    f"WARNING: Source image does not exist:\n"
                    f"  {source_image}"
                )
                missing += 1
                continue

            if not source_image.is_file():
                print(
                    f"WARNING: Source is not a file:\n"
                    f"  {source_image}"
                )
                missing += 1
                continue

            try:
                shutil.copy2(
                    str(source_image),
                    str(destination),
                )

                successful += 1

            except Exception as e:
                print(
                    f"ERROR copying:\n"
                    f"  Source: {source_image}\n"
                    f"  Destination: {destination}\n"
                    f"  Error: {e}"
                )

                missing += 1
                continue

            successful += 1

    # ---------------------------------------------------------
    # Save clustering CSV
    # ---------------------------------------------------------

    clustering_csv = (
        output_dir /
        "coverage_clusters.csv"
    )

    df = df.sort_values(
        by=[
            "cluster",
            "leaf_coverage_percent",
        ]
    )

    df.to_csv(
        clustering_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # Print cluster summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    for cluster_id in range(n_clusters):

        cluster_df = df[
            df["cluster"] == cluster_id
        ]

        if len(cluster_df) == 0:
            continue

        min_coverage = (
            cluster_df[
                "leaf_coverage_percent"
            ].min()
        )

        max_coverage = (
            cluster_df[
                "leaf_coverage_percent"
            ].max()
        )

        mean_coverage = (
            cluster_df[
                "leaf_coverage_percent"
            ].mean()
        )

        center = (
            cluster_df[
                "leaf_coverage_percent"
            ].mean()
        )

        print(
            f"\nCluster {cluster_id}"
        )

        print(
            f"  Images       : {len(cluster_df)}"
        )

        print(
            f"  Min coverage : "
            f"{min_coverage:.4f}%"
        )

        print(
            f"  Max coverage : "
            f"{max_coverage:.4f}%"
        )

        print(
            f"  Mean coverage: "
            f"{mean_coverage:.4f}%"
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Images copied : {successful}"
    )

    print(
        f"Images missing: {missing}"
    )

    print(
        f"Output folder : {output_dir}"
    )

    print(
        f"CSV saved     : {clustering_csv}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()