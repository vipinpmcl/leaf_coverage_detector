from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CoAtNetLeafDetector
from utils.data import (
    LeafSegmentationDataset,
    discover_samples,
    split_paths,
    split_clustered_samples,
)
from utils.losses import bce_dice_loss
from utils.metrics import metrics_from_counts
from utils.checkpoint import save_checkpoint


class Tee:
    """Write console output to both the terminal and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return config


def save_run_config(config_path, output_dir):
    """Save a copy of the exact configuration used for this run."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "config.yaml"
    shutil.copy2(config_path, destination)
    print(f"Config saved to: {destination}")


def evaluate(model, loader, device, threshold):
    """Evaluate loss and pixel-level binary segmentation metrics."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).float()

            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= threshold).float()
            tp += (preds * masks).sum().item()
            fp += (preds * (1.0 - masks)).sum().item()
            fn += ((1.0 - preds) * masks).sum().item()
            tn += ((1.0 - preds) * (1.0 - masks)).sum().item()

    if len(loader) == 0:
        raise RuntimeError("Cannot evaluate an empty data loader.")

    metrics = metrics_from_counts(tp, fp, fn, tn)
    return total_loss / len(loader), metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the CoAtNet leaf segmentation model. For clustered data, "
            "one image per cluster is selected for validation and another "
            "for test, when the cluster has at least two usable images."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--cluster-root",
        type=str,
        help="Root directory whose immediate subdirectories are clusters.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        help="Directory containing PNG masks named <image_stem>.png.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Dataset root containing images/ and masks/. Repeat to combine datasets.",
    )
    parser.add_argument(
        "--images",
        type=str,
        help="Image directory for one dataset (legacy; use with --masks).",
    )
    parser.add_argument(
        "--masks",
        type=str,
        help="Mask directory for one dataset (legacy; use with --images).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/coatnet_leaf",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    # Load config before using seed or any other config values.
    cfg = load_config(config_path)

    cluster_mode = bool(args.cluster_root or args.mask_dir)
    if cluster_mode:
        if not args.cluster_root or not args.mask_dir:
            raise SystemExit("ERROR: --cluster-root and --mask-dir must be provided together.")
        if args.source or args.images or args.masks:
            raise SystemExit(
                "ERROR: Use --cluster-root/--mask-dir OR --source OR --images/--masks, not a combination."
            )
        train_paths, val_paths, test_paths = split_clustered_samples(
            cluster_root=args.cluster_root,
            mask_dir=args.mask_dir,
            seed=cfg["data"].get("seed", 42),
        )
    else:
        if args.source and (args.images or args.masks):
            raise SystemExit("ERROR: Use repeated --source options or --images/--masks, not both.")
        if bool(args.images) != bool(args.masks):
            raise SystemExit("ERROR: --images and --masks must be provided together.")
        if not args.source and not args.images:
            raise SystemExit(
                "ERROR: Provide --cluster-root and --mask-dir, --source, "
                "or both --images and --masks."
            )

        sources = [(Path(root) / "images", Path(root) / "masks") for root in args.source]
        if args.images:
            sources.append((Path(args.images), Path(args.masks)))
        samples = discover_samples(sources)
        train_paths, val_paths = split_paths(
            samples,
            cfg["data"].get("val_ratio", 0.2),
            cfg["data"].get("seed", 42),
        )
        # Legacy mode has no dedicated test set.
        test_paths = []

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "training.log").open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        save_run_config(config_path, out_dir)

        print("\n========== Dataset Summary ==========")
        print(f"Train images : {len(train_paths)}")
        print(f"Val images   : {len(val_paths)}")
        print(f"Test images  : {len(test_paths)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        mean = tuple(cfg["normalization"]["mean"])
        std = tuple(cfg["normalization"]["std"])
        size = cfg["data"]["image_size"]
        batch_size = cfg["training"]["batch_size"]
        num_workers = cfg["training"]["num_workers"]
        pin_memory = device.type == "cuda"

        train_ds = LeafSegmentationDataset(
            train_paths,
            None,
            image_size=size,
            mean=mean,
            std=std,
            require_masks=True,
            augment=cfg["training"].get("augmentation", False),
        )
        val_ds = LeafSegmentationDataset(
            val_paths,
            None,
            image_size=size,
            mean=mean,
            std=std,
            require_masks=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = None
        if test_paths:
            test_ds = LeafSegmentationDataset(
                test_paths,
                None,
                image_size=size,
                mean=mean,
                std=std,
                require_masks=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------
        model_cfg = cfg["model"]
        model = CoAtNetLeafDetector(
            foundation_checkpoint=model_cfg["foundation_checkpoint"],
            model_name=model_cfg["name"],
            out_indices=tuple(model_cfg["out_indices"]),
            backbone_channels=tuple(model_cfg["backbone_channels"]),
            adapter_channels=tuple(model_cfg["adapter_channels"]),
            decoder_channels=model_cfg["decoder_channels"],
        ).to(device)

        if cfg["training"].get("freeze_backbone", False):
            model.freeze_backbone()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,}")
        print(f"Total parameters:     {total:,}")

        optimizer = torch.optim.AdamW(
            model.trainable_parameter_groups(
                head_lr=cfg["training"]["head_lr"],
                backbone_lr=cfg["training"]["backbone_lr"],
            ),
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=0.0,
        )

        amp_enabled = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        threshold = cfg["training"]["threshold"]
        best_iou = -1.0
        best_state = None

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            model.train()
            running_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg['training']['epochs']}",
            )

            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = bce_dice_loss(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(train_loader))
            val_loss, metrics = evaluate(model, val_loader, device, threshold)
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"IoU={metrics['iou']:.4f} "
                f"Dice={metrics['dice']:.4f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"lr={[f'{lr:.2e}' for lr in current_lrs]}"
            )
            scheduler.step()

            save_checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                epoch,
                metrics,
                config=cfg,
            )

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    config=cfg,
                )
                print("  -> saved best.pt")

        # -----------------------------------------------------
        # Final test evaluation using best validation-IoU model
        # -----------------------------------------------------
        if test_loader is not None:
            if best_state is not None:
                model.load_state_dict(best_state)
            test_loss, test_metrics = evaluate(model, test_loader, device, threshold)
            print("\n========== Final Test Results (best validation checkpoint) ==========")
            print(f"Test loss : {test_loss:.4f}")
            print(f"IoU       : {test_metrics['iou']:.4f}")
            print(f"Dice      : {test_metrics['dice']:.4f}")
            print(f"Precision : {test_metrics['precision']:.4f}")
            print(f"Recall    : {test_metrics['recall']:.4f}")
        else:
            print("\nNo dedicated test set in legacy mode; skipping test evaluation.")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
