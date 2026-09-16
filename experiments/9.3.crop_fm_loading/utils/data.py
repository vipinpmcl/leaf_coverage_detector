from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
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
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.require_masks = require_masks

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

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            antialias=True,
        )
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()
        else:
            if self.require_masks:
                raise FileNotFoundError(mask_path)
            mask = torch.zeros(
                1, self.image_size, self.image_size
            )

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
