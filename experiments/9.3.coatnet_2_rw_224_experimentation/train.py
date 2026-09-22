from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CoAtNetLeafDetector
from utils.data import LeafSegmentationDataset, discover_images, split_paths
from utils.losses import bce_dice_loss
from utils.metrics import segmentation_metrics
from utils.checkpoint import save_checkpoint


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_run_config(config_path, output_dir):
    """
    Save a copy of the exact config file used for this run.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / "config.yaml"

    shutil.copy2(config_path, destination)

    print(f"Config saved to: {destination}")


def evaluate(model, loader, device, threshold):
    model.eval()

    total_loss = 0.0

    metric_sum = {
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            loss = bce_dice_loss(logits, masks)

            total_loss += loss.item()

            metrics = segmentation_metrics(
                logits,
                masks,
                threshold=threshold,
            )

            for k in metric_sum:
                metric_sum[k] += metrics[k]

    n = max(1, len(loader))

    return (
        total_loss / n,
        {k: v / n for k, v in metric_sum.items()},
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
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

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    config_path = Path(args.config)
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    out_dir = Path(args.output)

    # Create output directory first
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    cfg = load_config(config_path)

    # Save exact config used for this run
    save_run_config(config_path, out_dir)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    image_paths = discover_images(images_dir)

    train_paths, val_paths = split_paths(
        image_paths,
        cfg["data"]["val_ratio"],
        cfg["data"]["seed"],
    )

    print(f"Total images : {len(image_paths)}")
    print(f"Train images : {len(train_paths)}")
    print(f"Val images   : {len(val_paths)}")

    mean = tuple(cfg["normalization"]["mean"])
    std = tuple(cfg["normalization"]["std"])
    size = cfg["data"]["image_size"]

    train_ds = LeafSegmentationDataset(
        train_paths,
        masks_dir,
        image_size=size,
        mean=mean,
        std=std,
        require_masks=True,
    )

    val_ds = LeafSegmentationDataset(
        val_paths,
        masks_dir,
        image_size=size,
        mean=mean,
        std=std,
        require_masks=True,
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model_cfg = cfg["model"]

    model = CoAtNetLeafDetector(
        foundation_checkpoint=model_cfg["foundation_checkpoint"],
        model_name=model_cfg["name"],
        out_indices=tuple(model_cfg["out_indices"]),
        backbone_channels=tuple(model_cfg["backbone_channels"]),
        adapter_channels=tuple(model_cfg["adapter_channels"]),
        decoder_channels=model_cfg["decoder_channels"],
    ).to(device)

    if cfg["training"]["freeze_backbone"]:
        model.freeze_backbone()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(
            head_lr=cfg["training"]["head_lr"],
            backbone_lr=cfg["training"]["backbone_lr"],
        ),
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------

    amp_enabled = (
        cfg["training"]["amp"]
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_iou = -1.0

    for epoch in range(
        1,
        cfg["training"]["epochs"] + 1,
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{cfg['training']['epochs']}"
            ),
        )

        for batch in progress:

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

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):

                logits = model(images)

                loss = bce_dice_loss(
                    logits,
                    masks,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # -----------------------------------------------------
        # Epoch metrics
        # -----------------------------------------------------

        train_loss = (
            running_loss
            / max(1, len(train_loader))
        )

        val_loss, metrics = evaluate(
            model,
            val_loader,
            device,
            cfg["training"]["threshold"],
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"IoU={metrics['iou']:.4f} "
            f"Dice={metrics['dice']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f}"
        )

        # -----------------------------------------------------
        # Save last checkpoint
        # -----------------------------------------------------

        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            epoch,
            metrics,
        )

        # -----------------------------------------------------
        # Save best checkpoint
        # -----------------------------------------------------

        if metrics["iou"] > best_iou:

            best_iou = metrics["iou"]

            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch,
                metrics,
            )

            print("  -> saved best.pt")


if __name__ == "__main__":
    main()