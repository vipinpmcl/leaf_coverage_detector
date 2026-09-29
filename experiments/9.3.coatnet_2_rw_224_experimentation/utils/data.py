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
        mask_dir,
        image_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        require_masks=True,
        augment=False,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.require_masks = require_masks
        self.augment = augment

        if require_masks:
            valid = []
            for p in self.image_paths:
                mask = self.mask_dir / f"{p.stem}.png"
                if mask.exists():
                    valid.append(p)
            self.image_paths = valid

        if not self.image_paths:
            raise RuntimeError("No usable images found.")

    def __len__(self):
        return len(self.image_paths)

    def _mask_path(self, image_path):
        return self.mask_dir / f"{image_path.stem}.png"

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        mask_path = self._mask_path(image_path)

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


def split_paths(paths, val_ratio=0.2, seed=42):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n_val = max(1, int(len(paths) * val_ratio))
    return paths[n_val:], paths[:n_val]
