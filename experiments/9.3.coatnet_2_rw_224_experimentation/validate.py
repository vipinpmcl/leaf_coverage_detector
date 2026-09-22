from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from models import CoAtNetLeafDetector
from utils.data import LeafSegmentationDataset, discover_images
from utils.checkpoint import load_model_checkpoint
from utils.metrics import segmentation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    paths = discover_images(cfg["data"]["image_dir"])

    ds = LeafSegmentationDataset(
        paths,
        cfg["data"]["mask_dir"],
        image_size=cfg["data"]["image_size"],
        mean=tuple(cfg["normalization"]["mean"]),
        std=tuple(cfg["normalization"]["std"]),
        require_masks=True,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    mc = cfg["model"]

    model = CoAtNetLeafDetector(
        foundation_checkpoint=mc["foundation_checkpoint"],
        model_name=mc["name"],
        out_indices=tuple(mc["out_indices"]),
        backbone_channels=tuple(mc["backbone_channels"]),
        adapter_channels=tuple(mc["adapter_channels"]),
        decoder_channels=mc["decoder_channels"],
    ).to(device)

    load_model_checkpoint(args.checkpoint, model, device)

    model.eval()

    sums = {
        "precision": 0,
        "recall": 0,
        "dice": 0,
        "iou": 0,
        "accuracy": 0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            m = segmentation_metrics(
                logits,
                masks,
                cfg["training"]["threshold"],
            )

            for k in sums:
                sums[k] += m[k]

    n = max(1, len(loader))
    print("Validation results:")
    for k, v in sums.items():
        print(f"{k:10s}: {v / n:.4f}")


if __name__ == "__main__":
    main()
