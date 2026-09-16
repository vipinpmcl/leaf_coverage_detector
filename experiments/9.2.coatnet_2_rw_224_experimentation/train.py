
"""
train.py

Two-stage training for CoAtNet2 leaf segmentation.

Stage A:
    Freeze CoAtNet backbone.
    Train FeatureAdapters + MultiScaleDecoder.

Stage B:
    Unfreeze CoAtNet backbone.
    Fine-tune the complete model using a smaller learning rate.

Example:

python train.py \
    --config configs/default.yaml \
    --images data/images \
    --masks data/masks \
    --output runs/coatnet_leaf


Resume:

python train.py \
    --config configs/default.yaml \
    --images data/images \
    --masks data/masks \
    --output runs/coatnet_leaf \
    --resume runs/coatnet_leaf/last.pth


Stage B:

python train.py \
    --config configs/default.yaml \
    --images data/images \
    --masks data/masks \
    --output runs/coatnet_leaf \
    --stage B \
    --resume runs/coatnet_leaf/best_stage_A.pth
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from models import CoAtNetLeafDetector


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is useful for experiments.
    # Benchmarking can be slightly slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Configuration
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Dataset
# ============================================================

class LeafSegmentationDataset(Dataset):
    """
    Image/mask dataset.

    Images:
        .jpg
        .jpeg
        .png
        .bmp
        .webp

    Masks:
        same filename as image inside mask directory.

    Example:

        images/
            leaf_001.jpg
            leaf_002.jpg

        masks/
            leaf_001.png
            leaf_002.png

    Mask values can be:
        0   -> background
        >0  -> leaf
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        image_paths,
        mask_dir,
        image_size=224,
        training=False,
    ):
        self.image_paths = [
            Path(p) for p in image_paths
        ]

        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.training = training

        if len(self.image_paths) == 0:
            raise RuntimeError("Dataset contains no images.")

    def __len__(self):
        return len(self.image_paths)

    def find_mask(self, image_path):
        """
        Find corresponding mask.

        First try exactly the same filename.

        Then try common mask extensions.
        """

        candidates = [
            self.mask_dir / image_path.name,
            self.mask_dir / f"{image_path.stem}.png",
            self.mask_dir / f"{image_path.stem}.jpg",
            self.mask_dir / f"{image_path.stem}.jpeg",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def load_mask(self, path, size):
        if path is None:
            # Missing mask.
            # Return an empty mask.
            return Image.new("L", size, 0)

        mask = Image.open(path).convert("L")
        return mask

    def random_augmentation(self, image, mask):
        """
        Apply identical spatial transformations to image and mask.
        """

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Small random rotation.
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)

            image = TF.rotate(
                image,
                angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        # Mild color augmentation.
        if random.random() < 0.5:
            image = TF.adjust_brightness(
                image,
                random.uniform(0.8, 1.2),
            )

        if random.random() < 0.5:
            image = TF.adjust_contrast(
                image,
                random.uniform(0.8, 1.2),
            )

        if random.random() < 0.5:
            image = TF.adjust_saturation(
                image,
                random.uniform(0.8, 1.2),
            )

        return image, mask

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = self.load_image(image_path)

        mask_path = self.find_mask(image_path)
        mask = self.load_mask(mask_path, image.size)

        if self.training:
            image, mask = self.random_augmentation(
                image,
                mask,
            )

        # Resize image.
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        # Resize mask using nearest-neighbor.
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )

        # Convert image to tensor.
        image = TF.to_tensor(image)

        # image = TF.normalize(
        #     image,
        #     mean=(0.5, 0.5, 0.5),
        #     std=(0.5, 0.5, 0.5),
        # )
        image = TF.normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # Binary mask.
        mask = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        )

        mask = (mask > 127).float()

        # [H,W] -> [1,H,W]
        mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "has_mask": mask_path is not None,
        }


# ============================================================
# Dataset discovery
# ============================================================

def collect_images(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    paths = [
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in LeafSegmentationDataset.IMAGE_EXTENSIONS
    ]

    paths.sort()

    return paths


def split_dataset(
    image_paths,
    val_ratio=0.2,
    seed=42,
):
    paths = list(image_paths)

    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)

    n_val = max(
        1,
        int(n * val_ratio),
    )

    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    return train_paths, val_paths


# ============================================================
# Loss
# ============================================================

def dice_loss(
    logits,
    targets,
    smooth=1.0,
):
    """
    Binary Dice loss.

    logits:
        [B,1,H,W]

    targets:
        [B,1,H,W]
    """

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.contiguous().view(
        probabilities.shape[0],
        -1,
    )

    targets = targets.contiguous().view(
        targets.shape[0],
        -1,
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


def bce_dice_loss(
    logits,
    targets,
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    return bce + dice


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    predictions = predictions.view(
        predictions.shape[0],
        -1,
    )

    targets = targets.view(
        targets.shape[0],
        -1,
    )

    tp = (
        predictions * targets
    ).sum(dim=1)

    fp = (
        predictions * (1.0 - targets)
    ).sum(dim=1)

    fn = (
        (1.0 - predictions) * targets
    ).sum(dim=1)

    tn = (
        (1.0 - predictions)
        * (1.0 - targets)
    ).sum(dim=1)

    iou = (
        tp + eps
    ) / (
        tp + fp + fn + eps
    )

    dice = (
        2.0 * tp + eps
    ) / (
        2.0 * tp + fp + fn + eps
    )

    precision = (
        tp + eps
    ) / (
        tp + fp + eps
    )

    recall = (
        tp + eps
    ) / (
        tp + fn + eps
    )

    accuracy = (
        tp + tn + eps
    ) / (
        tp + tn + fp + fn + eps
    )

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "accuracy": accuracy.mean().item(),
    }


# ============================================================
# Model
# ============================================================

def build_model(config):
    model_cfg = config["model"]

    model = CoAtNetLeafDetector(
        pretrained=model_cfg.get(
            "pretrained",
            True,
        ),
        out_indices=tuple(
            model_cfg.get(
                "out_indices",
                [0, 1, 2, 3],
            )
        ),
        decoder_channels=model_cfg.get(
            "decoder_channels",
            128,
        ),
        adapter_channels=tuple(
            model_cfg.get(
                "adapter_channels",
                [64, 96, 192, 384],
            )
        ),
    )

    return model


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(
    model,
    config,
    stage,
):
    train_cfg = config["training"]

    head_lr = float(
        train_cfg.get(
            "head_lr",
            1e-4,
        )
    )

    backbone_lr = float(
        train_cfg.get(
            "backbone_lr",
            1e-5,
        )
    )

    weight_decay = float(
        train_cfg.get(
            "weight_decay",
            1e-4,
        )
    )

    if stage == "A":

        # Backbone frozen.
        model.freeze_backbone()

        head_parameters = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            head_parameters,
            lr=head_lr,
            weight_decay=weight_decay,
        )

    else:

        # Complete model trainable.
        model.unfreeze_backbone()

        backbone_parameters = []
        head_parameters = []

        for name, parameter in model.named_parameters():

            if not parameter.requires_grad:
                continue

            if name.startswith("backbone."):
                backbone_parameters.append(parameter)
            else:
                head_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_lr,
                },
                {
                    "params": head_parameters,
                    "lr": head_lr,
                },
            ],
            weight_decay=weight_decay,
        )

    return optimizer


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    threshold=0.5,
):
    model.train()

    running_loss = 0.0

    metric_sum = {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    num_batches = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # Mixed precision.
        use_amp = (
            device.type == "cuda"
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

            loss = bce_dice_loss(
                logits,
                masks,
            )

        if use_amp:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        running_loss += loss.item()

        metrics = segmentation_metrics(
            logits.detach(),
            masks,
            threshold=threshold,
        )

        for key in metric_sum:
            metric_sum[key] += metrics[key]

        num_batches += 1

    results = {
        "loss": running_loss / max(num_batches, 1)
    }

    for key in metric_sum:
        results[key] = (
            metric_sum[key]
            / max(num_batches, 1)
        )

    return results


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    threshold=0.5,
):
    model.eval()

    running_loss = 0.0

    metric_sum = {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    num_batches = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = bce_dice_loss(
            logits,
            masks,
        )

        running_loss += loss.item()

        metrics = segmentation_metrics(
            logits,
            masks,
            threshold=threshold,
        )

        for key in metric_sum:
            metric_sum[key] += metrics[key]

        num_batches += 1

    results = {
        "loss": running_loss / max(num_batches, 1)
    }

    for key in metric_sum:
        results[key] = (
            metric_sum[key]
            / max(num_batches, 1)
        )

    return results


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    stage,
    config,
):
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_metric": best_metric,
        "config": config,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    print(f"\nLoading checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict")
        is not None
    ):
        try:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        except Exception as e:
            print(
                "Warning: optimizer state could "
                f"not be restored: {e}"
            )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict")
        is not None
    ):
        try:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
        except Exception as e:
            print(
                "Warning: scheduler state could "
                f"not be restored: {e}"
            )

    epoch = checkpoint.get(
        "epoch",
        -1,
    )

    best_metric = checkpoint.get(
        "best_metric",
        -float("inf"),
    )

    print(
        f"Checkpoint loaded. "
        f"Epoch={epoch + 1}, "
        f"best Dice={best_metric:.4f}"
    )

    return epoch, best_metric


# ============================================================
# Parameter summary
# ============================================================

def print_model_summary(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(
        f"Total parameters     : {total:,}"
    )

    print(
        f"Trainable parameters : {trainable:,}"
    )

    print(
        f"Frozen parameters    : {frozen:,}"
    )

    print(
        f"Trainable percentage  : "
        f"{100.0 * trainable / max(total, 1):.2f}%"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train CoAtNet leaf segmentation model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Image directory.",
    )

    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Mask directory.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/coatnet_leaf",
        help="Output directory.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["A", "B"],
        default="A",
        help="Training stage.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume/load.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu. Automatically detected if omitted.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_everything(args.seed)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    train_cfg = config["training"]

    image_size = int(
        train_cfg.get(
            "image_size",
            224,
        )
    )

    batch_size = int(
        train_cfg.get(
            "batch_size",
            4,
        )
    )

    epochs = int(
        train_cfg.get(
            "epochs",
            20,
        )
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save configuration.
    with open(
        output_dir / "config_used.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device is not None:
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("\nDevice:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    image_paths = collect_images(
        args.images
    )

    print(
        f"\nTotal images found: "
        f"{len(image_paths)}"
    )

    train_paths, val_paths = split_dataset(
        image_paths,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        f"Training images   : "
        f"{len(train_paths)}"
    )

    print(
        f"Validation images : "
        f"{len(val_paths)}"
    )
    
    train_dataset = LeafSegmentationDataset(
        train_paths,
        args.masks,
        image_size=image_size,
        training=True,
    )

    val_dataset = LeafSegmentationDataset(
        val_paths,
        args.masks,
        image_size=image_size,
        training=False,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = build_model(
        config
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Stage configuration
    # --------------------------------------------------------

    if args.stage == "A":

        print(
            "\nStage A: "
            "FREEZE CoAtNet BACKBONE"
        )

        model.freeze_backbone()

    else:

        print(
            "\nStage B: "
            "UNFREEZE CoAtNet BACKBONE"
        )

        model.unfreeze_backbone()

    print_model_summary(
        model
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model,
        config,
        args.stage,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 0

    best_dice = -float("inf")

    if args.resume is not None:

        checkpoint_epoch, best_dice = (
            load_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler,
                device=device,
            )
        )

        # When explicitly starting Stage B from
        # Stage A checkpoint, don't necessarily want
        # to continue the epoch numbering.
        start_epoch = max(
            checkpoint_epoch + 1,
            0,
        )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_path = (
        output_dir
        / f"history_stage_{args.stage}.json"
    )

    if history_path.exists():

        with open(
            history_path,
            "r",
            encoding="utf-8",
        ) as f:
            history = json.load(f)

    else:

        history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    epochs_without_improvement = 0

    print("\n" + "=" * 70)

    print(
        f"Starting Stage {args.stage}"
    )

    print(
        f"Epochs: {epochs}"
    )

    print(
        f"Image size: "
        f"{image_size}x{image_size}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print("=" * 70)

    for epoch in range(
        start_epoch,
        epochs,
    ):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{epochs}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            threshold=args.threshold,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = validate(
            model,
            val_loader,
            device,
            threshold=args.threshold,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["dice"]
        )

        # ----------------------------------------------------
        # Learning rates
        # ----------------------------------------------------

        learning_rates = [
            group["lr"]
            for group in optimizer.param_groups
        ]

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            "\nTrain:"
        )

        print(
            f"  Loss      : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Dice      : "
            f"{train_metrics['dice']:.4f}"
        )

        print(
            f"  IoU       : "
            f"{train_metrics['iou']:.4f}"
        )

        print(
            f"  Precision : "
            f"{train_metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{train_metrics['recall']:.4f}"
        )

        print(
            "\nValidation:"
        )

        print(
            f"  Loss      : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Dice      : "
            f"{val_metrics['dice']:.4f}"
        )

        print(
            f"  IoU       : "
            f"{val_metrics['iou']:.4f}"
        )

        print(
            f"  Precision : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            "\nLearning rates:",
            [
                f"{lr:.2e}"
                for lr in learning_rates
            ],
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        record = {
            "epoch": epoch + 1,
            "stage": args.stage,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rates": learning_rates,
        }

        history.append(
            record
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            args.stage,
            config,
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        current_dice = (
            val_metrics["dice"]
        )

        if current_dice > best_dice:

            best_dice = current_dice

            epochs_without_improvement = 0

            best_path = (
                output_dir
                / f"best_stage_{args.stage}.pth"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_dice,
                args.stage,
                config,
            )

            print(
                f"\n✓ New best model!"
                f" Dice={best_dice:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/"
                f"{args.patience})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # Finished
    # ========================================================

    print("\n" + "=" * 70)

    print(
        f"Stage {args.stage} training complete."
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()