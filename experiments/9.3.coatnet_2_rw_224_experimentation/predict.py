from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from models import CoAtNetLeafDetector
from utils.checkpoint import load_model_checkpoint
from utils.visualize import save_prediction_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="runs/predictions")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    x = TF.resize(
        image,
        [cfg["data"]["image_size"], cfg["data"]["image_size"]],
        antialias=True,
    )
    x = TF.to_tensor(x)
    x = TF.normalize(
        x,
        tuple(cfg["normalization"]["mean"]),
        tuple(cfg["normalization"]["std"]),
    )
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            x,
            output_size=(image.height, image.width),
        )
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    output_dir = (
        Path(args.output_dir) / image_path.stem
    )

    save_prediction_outputs(
        image=image,
        probability=probability,
        output_dir=output_dir,
        threshold=cfg["training"]["threshold"],
    )

    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
