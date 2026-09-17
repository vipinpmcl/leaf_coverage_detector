from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LeafSegmentationDataset(Dataset):

    def __init__(
        self,
        image_paths,
        mask_dir,
        image_size=224
    ):

        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

        # --------------------------------------------------
        # Keep only images for which a mask exists
        # --------------------------------------------------

        self.image_paths = []
        self.missing_masks = []

        for image_path in image_paths:

            image_path = Path(image_path)

            mask_path = (
                self.mask_dir /
                f"{image_path.stem}.png"
            )

            if mask_path.exists():

                self.image_paths.append(
                    image_path
                )

            else:

                self.missing_masks.append(
                    image_path
                )

        # --------------------------------------------------
        # Report dataset statistics
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Leaf Segmentation Dataset")
        print("=" * 60)

        print(
            f"Images found       : {len(image_paths)}"
        )

        print(
            f"Masks found        : {len(self.image_paths)}"
        )

        print(
            f"Masks missing      : "
            f"{len(self.missing_masks)}"
        )

        if self.missing_masks:

            print()
            print(
                "Missing-mask examples:"
            )

            for p in self.missing_masks[:10]:

                print(
                    f"  {p}"
                )

            if len(self.missing_masks) > 10:

                print(
                    f"  ... and "
                    f"{len(self.missing_masks) - 10} more"
                )

        print("=" * 60)
        print()

        if len(self.image_paths) == 0:

            raise RuntimeError(
                "No images with corresponding masks were found."
            )

    def __len__(self):

        return len(
            self.image_paths
        )

    def __getitem__(self, idx):

        image_path = (
            self.image_paths[idx]
        )

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # --------------------------------------------------
        # Load mask
        # --------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        # --------------------------------------------------
        # Resize image
        # --------------------------------------------------

        image = self.transform(
            image
        )

        # --------------------------------------------------
        # Resize mask
        #
        # NEVER use bilinear interpolation for masks.
        # --------------------------------------------------

        mask = mask.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.NEAREST
        )

        # --------------------------------------------------
        # Convert mask:
        #
        # 0   -> background
        # 255 -> leaf
        # --------------------------------------------------

        mask = (
            np.asarray(mask) >= 128
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        )

        return (
            image,
            mask,
            str(image_path)
        )