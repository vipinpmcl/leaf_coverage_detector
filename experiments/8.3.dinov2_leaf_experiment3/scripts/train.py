import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from src.dataset import LeafSegmentationDataset
from src.dinov2 import DINOv2Backbone
from src.model import LeafSegmentationModel
from src.losses import segmentation_loss
from src.metrics import segmentation_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
    ])

    if len(images) < 2:
        raise RuntimeError("Need at least 2 labelled images.")

    dataset = LeafSegmentationDataset(
        images, mask_dir, cfg["model"]["image_size"]
    )

    val_size = max(1, int(len(dataset) * cfg["training"]["val_fraction"]))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"]
    )

    backbone = DINOv2Backbone(
        cfg["model"]["name"],
        cfg["model"]["image_size"],
        device
    )

    model = LeafSegmentationModel(
        backbone,
        cfg["model"]["feature_dim"],
        cfg["decoder"]["hidden_dim"],
        cfg["decoder"]["dropout"]
    ).to(device)

    # Warm-up: train decoder while DINOv2 is frozen.
    model.backbone.set_trainable(False)

    optimizer = torch.optim.AdamW([
        {
            "params": model.decoder.parameters(),
            "lr": cfg["training"]["decoder_learning_rate"]
        },
        {
            "params": model.backbone.model.parameters(),
            "lr": cfg["training"]["backbone_learning_rate"]
        }
    ], weight_decay=cfg["training"]["weight_decay"])

    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):

        if epoch > cfg["model"]["freeze_backbone_epochs"]:
            model.backbone.set_trainable(True)
            model.backbone.train()
        else:
            model.backbone.set_trainable(False)
            model.backbone.eval()

        model.decoder.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = segmentation_loss(
                logits,
                y,
                cfg["training"]["bce_weight"],
                cfg["training"]["dice_weight"]
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_logits = []
        all_masks = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                all_logits.append(model(x.to(device)).cpu())
                all_masks.append(y)

        logits = torch.cat(all_logits)
        masks = torch.cat(all_masks)

        m = segmentation_metrics(
            logits,
            masks,
            cfg["training"]["threshold"]
        )

        avg_loss = train_loss / max(1, len(train_loader))

        print(
            f"Epoch {epoch:03d} "
            f"loss={avg_loss:.4f} "
            f"IoU={m['iou']:.4f} "
            f"Dice={m['dice']:.4f} "
            f"Precision={m['precision']:.4f} "
            f"Recall={m['recall']:.4f}"
        )

        if m["iou"] > best_iou:
            best_iou = m["iou"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "val_metrics": m,
                    "epoch": epoch
                },
                output_dir / "best.pt"
            )

    print("Training complete.")
    print("Best IoU:", best_iou)
    print("Checkpoint:", output_dir / "best.pt")

if __name__ == "__main__":
    main()
