from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LeafSegmentationDataset(Dataset):
    """Image/mask dataset for binary leaf segmentation.

    ``image_paths`` can contain image paths or explicit ``(image, mask)`` pairs.
    When paths are provided, ``mask_dir`` is used to find ``<image_stem>.png``.
    """

    def __init__(
        self,
        image_paths,
        mask_dir=None,
        image_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        require_masks=True,
        augment=False,
    ):
        self.samples = []
        for sample in image_paths:
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                image_path, mask_path = map(Path, sample)
            else:
                image_path = Path(sample)
                if mask_dir is None:
                    raise ValueError(
                        "mask_dir is required when image_paths contains image paths "
                        "instead of (image_path, mask_path) pairs."
                    )
                mask_path = Path(mask_dir) / f"{image_path.stem}.png"
            self.samples.append((image_path, mask_path))

        self.image_size = int(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.require_masks = bool(require_masks)
        self.augment = bool(augment)

        if self.require_masks:
            self.samples = [
                sample for sample in self.samples
                if sample[0].is_file() and sample[1].is_file()
            ]

        if not self.samples:
            raise RuntimeError("No usable image/mask samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        if mask_path.is_file():
            with Image.open(mask_path) as im:
                mask = im.convert("L")
        elif self.require_masks:
            raise FileNotFoundError(f"Mask not found for {image_path}: {mask_path}")
        else:
            mask = Image.new("L", image.size, 0)

        if image.size != mask.size:
            # Keep image/mask geometry aligned before applying transforms.
            mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)

        if self.augment:
            # Apply identical geometry to the image and its segmentation mask.
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            angle = random.choice((0, 90, 180, 270))
            if angle:
                image = TF.rotate(
                    image,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                )

            top, left, height, width = RandomResizedCrop.get_params(
                image,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )
        else:
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                antialias=True,
            )
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "path": str(image_path),
        }


def discover_images(image_dir):
    """Return supported image files directly inside ``image_dir``."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        (
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda p: (p.name.casefold(), p.name),
    )


def discover_samples(sources):
    """Discover image/mask pairs from ``(image_dir, mask_dir)`` source roots.

    Raises an error if any image lacks its expected PNG mask.
    """
    samples = []
    missing_masks = []

    for image_dir, mask_dir in sources:
        image_dir, mask_dir = Path(image_dir), Path(mask_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        for image_path in discover_images(image_dir):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.is_file():
                samples.append((image_path, mask_path))
            else:
                missing_masks.append((image_path, mask_path))

    if missing_masks:
        examples = "\n".join(
            f"  Image: {image}\n  Expected mask: {mask}"
            for image, mask in missing_masks[:10]
        )
        raise RuntimeError(
            f"{len(missing_masks)} image(s) have no matching PNG mask. Examples:\n{examples}"
        )

    return sorted(samples, key=lambda pair: (str(pair[0]).casefold(), str(pair[0])))


def split_paths(paths, val_ratio=0.2, seed=42):
    """Legacy random train/validation split for non-clustered datasets."""
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("At least 2 samples are required for a train/validation split.")
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    n_val = min(n_val, len(paths) - 1)
    return paths[n_val:], paths[:n_val]


def split_clustered_samples(cluster_root, mask_dir, seed=42):
    """Create train/validation/test image-mask pairs by cluster.

    For every cluster directory, choose one eligible sample for validation and
    one *different* eligible sample for test. All other eligible samples from
    that cluster go to training. An eligible sample has a matching PNG mask
    named ``<image_stem>.png`` in the shared mask directory.

    If a cluster has exactly one eligible sample, it is kept in training and
    reported because it cannot supply distinct validation and test images.
    If it has two eligible samples, both are assigned to validation/test and
    that cluster contributes no training sample.

    Returns:
        (train_samples, val_samples, test_samples), each a list of
        ``(image_path, mask_path)`` tuples.
    """
    cluster_root = Path(cluster_root)
    mask_dir = Path(mask_dir)

    if not cluster_root.is_dir():
        raise FileNotFoundError(f"Cluster root does not exist: {cluster_root}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    cluster_dirs = sorted(
        (p for p in cluster_root.iterdir() if p.is_dir()),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if not cluster_dirs:
        raise RuntimeError(f"No cluster subdirectories found in: {cluster_root}")

    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []
    missing_masks = []
    empty_clusters = []
    one_sample_clusters = []
    cluster_counts = []
    total_images = 0

    for cluster_dir in cluster_dirs:
        image_paths = discover_images(cluster_dir)
        total_images += len(image_paths)
        eligible = []

        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.is_file():
                eligible.append((image_path, mask_path))
            else:
                missing_masks.append((image_path, mask_path))

        rng.shuffle(eligible)
        usable_count = len(eligible)
        cluster_counts.append((cluster_dir.name, len(image_paths), usable_count))

        if usable_count == 0:
            empty_clusters.append(cluster_dir.name)
            continue

        if usable_count == 1:
            # Cannot select two distinct samples for validation and test.
            # Keep the lone usable sample in training rather than discarding it.
            train_samples.extend(eligible)
            one_sample_clusters.append(cluster_dir.name)
            continue

        # Pop distinct samples so validation/test can never overlap.
        val_samples.append(eligible.pop())
        test_samples.append(eligible.pop())
        train_samples.extend(eligible)

    # Sort final lists for stable downstream ordering; random selections are
    # already controlled by the seed.
    sort_key = lambda pair: (str(pair[0]).casefold(), str(pair[0]))
    train_samples.sort(key=sort_key)
    val_samples.sort(key=sort_key)
    test_samples.sort(key=sort_key)

    print("\n========== Cluster Split Summary ==========")
    print(f"Cluster directories : {len(cluster_dirs)}")
    print(f"Images discovered    : {total_images}")
    print(f"Usable image/mask pairs: {len(train_samples) + len(val_samples) + len(test_samples)}")
    print(f"Missing masks        : {len(missing_masks)}")
    print(f"Train images         : {len(train_samples)}")
    print(f"Validation images    : {len(val_samples)}")
    print(f"Test images          : {len(test_samples)}")
    print(f"Clusters with 0 usable images: {len(empty_clusters)}")
    print(f"Clusters with 1 usable image : {len(one_sample_clusters)}")

    if missing_masks:
        print("\nFirst missing-mask examples:")
        for image_path, expected_mask in missing_masks[:10]:
            print(f"  {image_path} (expected mask: {expected_mask})")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")

    if empty_clusters:
        print("\nClusters with no usable image/mask pairs:")
        for name in empty_clusters[:20]:
            print(f"  {name}")
        if len(empty_clusters) > 20:
            print(f"  ... and {len(empty_clusters) - 20} more")

    if one_sample_clusters:
        print("\nClusters with only one usable image (kept in train):")
        for name in one_sample_clusters[:20]:
            print(f"  {name}")
        if len(one_sample_clusters) > 20:
            print(f"  ... and {len(one_sample_clusters) - 20} more")

    if not train_samples:
        raise RuntimeError(
            "Training set is empty. More clusters need at least 3 usable images, "
            "or clusters with a single usable image must be reviewed."
        )
    if not val_samples or not test_samples:
        raise RuntimeError(
            "Could not create non-empty validation and test sets. At least one "
            "cluster must contain 2 or more images with matching masks."
        )

    # Defensive assertion: every image path appears in at most one split.
    train_ids = {str(image.resolve()) for image, _ in train_samples}
    val_ids = {str(image.resolve()) for image, _ in val_samples}
    test_ids = {str(image.resolve()) for image, _ in test_samples}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Data leakage detected: image paths overlap between splits.")

    return train_samples, val_samples, test_samples
