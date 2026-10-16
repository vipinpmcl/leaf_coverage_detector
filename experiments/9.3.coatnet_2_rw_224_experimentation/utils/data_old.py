from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LeafSegmentationDataset(Dataset):
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
        # Accept explicit (image, mask) pairs so datasets from separate roots
        # can be combined without flattening/copying them into one directory.
        self.samples = []
        for sample in image_paths:
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                image_path, mask_path = map(Path, sample)
            else:
                image_path = Path(sample)
                mask_path = Path(mask_dir) / f"{image_path.stem}.png"
            self.samples.append((image_path, mask_path))
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.require_masks = require_masks
        self.augment = augment

        if require_masks:
            self.samples = [sample for sample in self.samples if sample[1].exists()]

        if not self.samples:
            raise RuntimeError("No usable images found.")

    def __len__(self):
        return len(self.samples)

    def _mask_path(self, image_path):
        for sample_image, mask_path in self.samples:
            if sample_image == image_path:
                return mask_path
        raise KeyError(image_path)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        elif self.require_masks:
            raise FileNotFoundError(mask_path)
        else:
            mask = Image.new("L", image.size, 0)

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
    image_dir = Path(image_dir)
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def discover_samples(sources):
    """Discover image/mask pairs from (image_dir, mask_dir) source roots."""
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
        examples = ", ".join(str(image) for image, _ in missing_masks[:5])
        raise RuntimeError(
            f"{len(missing_masks)} image(s) have no matching PNG mask; examples: {examples}"
        )
    # Keep deterministic ordering while preserving the image-to-mask pairing.
    return sorted(samples, key=lambda pair: (str(pair[0]).casefold(), str(pair[0])))


def split_paths(paths, val_ratio=0.2, seed=42):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n_val = max(1, int(len(paths) * val_ratio))
    return paths[n_val:], paths[:n_val]
