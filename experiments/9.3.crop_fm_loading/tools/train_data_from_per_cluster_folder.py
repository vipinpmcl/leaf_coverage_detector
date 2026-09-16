import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create train/test dataset by selecting "
            "one image from each cluster."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing image_path and cluster columns"
    )

    parser.add_argument(
        "--output",
        default="data/cluster_split",
        help="Output directory"
    )

    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help=(
            "Do not use HDBSCAN noise cluster (-1) "
            "for training. Noise images go to test."
        )
    )

    args = parser.parse_args()

    # ==================================================
    # Paths
    # ==================================================

    csv_path = Path(args.csv)
    output_root = Path(args.output)

    train_dir = output_root / "1.train"
    test_dir = output_root / "2.test"

    train_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # Read CSV
    # ==================================================

    df = pd.read_csv(csv_path)

    required_columns = {
        "image_path",
        "cluster"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"CSV must contain: image_path, cluster"
        )

    print("=" * 70)
    print("Cluster-based Train/Test Split")
    print("=" * 70)

    print(
        f"Total rows in CSV : {len(df)}"
    )

    # ==================================================
    # Normalize cluster
    # ==================================================

    df["cluster"] = pd.to_numeric(
        df["cluster"],
        errors="coerce"
    )

    if df["cluster"].isna().any():

        bad = df[
            df["cluster"].isna()
        ]

        print(
            f"WARNING: {len(bad)} rows have invalid cluster values."
        )

        df = df[
            df["cluster"].notna()
        ].copy()

    df["cluster"] = df["cluster"].astype(int)

    # ==================================================
    # Prepare train candidates
    # ==================================================

    if args.exclude_noise:

        print(
            "Noise cluster (-1): TEST ONLY"
        )

        train_candidates = df[
            df["cluster"] != -1
        ].copy()

    else:

        print(
            "Noise cluster (-1): treated as a cluster"
        )

        train_candidates = df.copy()

    # ==================================================
    # Select ONE image per cluster
    # ==================================================

    selected_train = (
        train_candidates
        .sort_values(
            ["cluster", "image_path"]
        )
        .groupby(
            "cluster",
            sort=True
        )
        .first()
        .reset_index()
    )

    # ==================================================
    # Create lookup of selected training images
    # ==================================================

    selected_paths = set(
        selected_train["image_path"]
        .astype(str)
    )

    # ==================================================
    # Split dataset
    # ==================================================

    train_df = df[
        df["image_path"]
        .astype(str)
        .isin(selected_paths)
    ].copy()

    test_df = df[
        ~df["image_path"]
        .astype(str)
        .isin(selected_paths)
    ].copy()

    print()
    print(
        f"Clusters          : {df['cluster'].nunique()}"
    )

    print(
        f"Train images      : {len(train_df)}"
    )

    print(
        f"Test images       : {len(test_df)}"
    )

    # ==================================================
    # Copy TRAIN images
    # ==================================================

    train_manifest = []

    print()
    print("-" * 70)
    print("Copying TRAIN images")
    print("-" * 70)

    for _, row in train_df.iterrows():

        source = Path(
            str(row["image_path"])
        )

        cluster = int(
            row["cluster"]
        )

        if not source.exists():

            print(
                f"[MISSING] {source}"
            )

            continue

        # Keep original filename but prefix cluster.
        destination_name = (
            f"cluster_{cluster:03d}_"
            f"{source.name}"
        )

        destination = (
            train_dir /
            destination_name
        )

        shutil.copy2(
            source,
            destination
        )

        train_manifest.append({
            "image_path": str(source),
            "cluster": cluster,
            "output_path": str(destination)
        })

        print(
            f"[TRAIN] "
            f"cluster={cluster:3d} "
            f"{source.name}"
        )

    # ==================================================
    # Copy TEST images
    # ==================================================

    test_manifest = []

    print()
    print("-" * 70)
    print("Copying TEST images")
    print("-" * 70)

    for _, row in test_df.iterrows():

        source = Path(
            str(row["image_path"])
        )

        cluster = int(
            row["cluster"]
        )

        if not source.exists():

            print(
                f"[MISSING] {source}"
            )

            continue

        destination = (
            test_dir /
            source.name
        )

        # Handle duplicate filenames.
        if destination.exists():

            stem = source.stem
            suffix = source.suffix

            counter = 1

            while destination.exists():

                destination = (
                    test_dir /
                    f"{stem}_{counter}{suffix}"
                )

                counter += 1

        shutil.copy2(
            source,
            destination
        )

        test_manifest.append({
            "image_path": str(source),
            "cluster": cluster,
            "output_path": str(destination)
        })

        print(
            f"[TEST ] "
            f"cluster={cluster:3d} "
            f"{source.name}"
        )

    # ==================================================
    # Save manifests
    # ==================================================

    train_manifest_path = (
        output_root /
        "train_manifest.csv"
    )

    test_manifest_path = (
        output_root /
        "test_manifest.csv"
    )

    pd.DataFrame(
        train_manifest
    ).to_csv(
        train_manifest_path,
        index=False
    )

    pd.DataFrame(
        test_manifest
    ).to_csv(
        test_manifest_path,
        index=False
    )

    # ==================================================
    # Save split CSVs
    # ==================================================

    train_df.to_csv(
        output_root / "train.csv",
        index=False
    )

    test_df.to_csv(
        output_root / "test.csv",
        index=False
    )

    # ==================================================
    # Print cluster statistics
    # ==================================================

    print()
    print("=" * 70)
    print("Cluster distribution")
    print("=" * 70)

    stats = (
        df.groupby("cluster")
        .size()
        .reset_index(name="total")
    )

    stats["train"] = (
        stats["cluster"]
        .isin(
            train_df["cluster"]
        )
        .astype(int)
    )

    stats["test"] = (
        stats["total"]
        - stats["train"]
    )

    print(
        stats.to_string(
            index=False
        )
    )

    # ==================================================
    # Final summary
    # ==================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Original images : {len(df)}"
    )

    print(
        f"Train images    : {len(train_manifest)}"
    )

    print(
        f"Test images     : {len(test_manifest)}"
    )

    print(
        f"Output          : {output_root}"
    )

    print()
    print(
        "Train directory:"
    )

    print(
        f"  {train_dir}"
    )

    print(
        "Test directory:"
    )

    print(
        f"  {test_dir}"
    )

    print()
    print(
        "Original dataset has NOT been modified."
    )


if __name__ == "__main__":
    main()