
"""
validate.py

Validate the CoAtNet-2 leaf segmentation model.

Architecture:
    CoAtNet-2
        |
        +-- FeatureAdapters
        |
        +-- MultiScaleDecoder
        |
        +-- 1-channel segmentation logits

Example:

python validate.py \
    --config configs/default.yaml \
    --images data/images \
    --masks data/masks \
    --checkpoint runs/coatnet_leaf/best_stage_A.pth \
    --output runs/coatnet_leaf/validation

"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
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


# ============================================================
# Dataset
# ============================================================

class LeafValidationDataset(Dataset):

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
    ):
        self.image_paths = [
            Path(p) for p in image_paths
        ]

        self.mask_dir = Path(mask_dir)

        self.image_size = image_size

        if len(self.image_paths) == 0:
            raise RuntimeError(
                "Validation dataset contains no images."
            )

    def __len__(self):
        return len(self.image_paths)

    def find_mask(self, image_path):

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

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        original_image = Image.open(
            image_path
        ).convert("RGB")

        original_size = original_image.size

        mask_path = self.find_mask(
            image_path
        )

        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for: {image_path}"
            )

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = original_image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BICUBIC,
        )

        mask = mask.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.NEAREST,
        )

        # ----------------------------------------------------
        # Image tensor
        # ----------------------------------------------------

        image_tensor = TF.to_tensor(
            image
        )

        # IMPORTANT:
        # CoAtNet checkpoint specifies:
        #
        # mean = (0.5, 0.5, 0.5)
        # std  = (0.5, 0.5, 0.5)
        #
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        # ----------------------------------------------------
        # Mask tensor
        # ----------------------------------------------------

        mask_tensor = torch.from_numpy(
            np.array(
                mask,
                dtype=np.float32,
            )
        )

        mask_tensor = (
            mask_tensor > 127
        ).float()

        mask_tensor = mask_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": original_size,
        }


# ============================================================
# Dataset utilities
# ============================================================

def collect_images(image_dir):

    image_dir = Path(image_dir)

    paths = [
        p
        for p in image_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in LeafValidationDataset.IMAGE_EXTENSIONS
        )
    ]

    paths.sort()

    if not paths:
        raise RuntimeError(
            f"No images found in {image_dir}"
        )

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

    return val_paths


# ============================================================
# Loss
# ============================================================

def dice_loss(
    logits,
    targets,
    smooth=1.0,
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.reshape(
        probabilities.shape[0],
        -1,
    )

    targets = targets.reshape(
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
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
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

def calculate_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
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
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
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
# Checkpoint loading
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    print(
        f"\nLoading checkpoint:"
        f"\n  {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Normal training checkpoint
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        checkpoint_epoch = checkpoint.get(
            "epoch",
            None,
        )

        checkpoint_stage = checkpoint.get(
            "stage",
            None,
        )

        checkpoint_best = checkpoint.get(
            "best_metric",
            None,
        )

        print(
            f"Checkpoint epoch : "
            f"{checkpoint_epoch}"
        )

        print(
            f"Checkpoint stage : "
            f"{checkpoint_stage}"
        )

        if checkpoint_best is not None:
            print(
                f"Best metric      : "
                f"{checkpoint_best:.4f}"
            )

    # --------------------------------------------------------
    # Raw state_dict
    # --------------------------------------------------------

    elif isinstance(
        checkpoint,
        dict
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    print(
        f"\nMissing keys: "
        f"{len(missing)}"
    )

    if missing:
        for key in missing[:20]:
            print(
                f"  MISSING: {key}"
            )

        if len(missing) > 20:
            print(
                f"  ... and "
                f"{len(missing) - 20} more"
            )

    print(
        f"\nUnexpected keys: "
        f"{len(unexpected)}"
    )

    if unexpected:
        for key in unexpected[:20]:
            print(
                f"  UNEXPECTED: {key}"
            )

        if len(unexpected) > 20:
            print(
                f"  ... and "
                f"{len(unexpected) - 20} more"
            )

    print("\nCheckpoint loaded.")

    return checkpoint


# ============================================================
# Visualization
# ============================================================

def create_visualization(
    image_path,
    ground_truth,
    prediction,
    output_path,
):

    original = Image.open(
        image_path
    ).convert("RGB")

    # Ground truth.
    gt = Image.fromarray(
        (
            ground_truth
            * 255
        ).astype(np.uint8)
    ).convert("L")

    # Prediction.
    pred = Image.fromarray(
        (
            prediction
            * 255
        ).astype(np.uint8)
    ).convert("L")

    # Resize masks to original image size.
    gt = gt.resize(
        original.size,
        Image.Resampling.NEAREST,
    )

    pred = pred.resize(
        original.size,
        Image.Resampling.NEAREST,
    )

    # Convert masks to RGB.
    gt_rgb = Image.merge(
        "RGB",
        (gt, gt, gt),
    )

    pred_rgb = Image.merge(
        "RGB",
        (pred, pred, pred),
    )

    width, height = original.size

    canvas = Image.new(
        "RGB",
        (
            width * 3,
            height,
        ),
    )

    canvas.paste(
        original,
        (0, 0),
    )

    canvas.paste(
        gt_rgb,
        (width, 0),
    )

    canvas.paste(
        pred_rgb,
        (width * 2, 0),
    )

    # Labels.
    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (10, 10),
        "IMAGE",
        fill="white",
    )

    draw.text(
        (width + 10, 10),
        "GROUND TRUTH",
        fill="white",
    )

    draw.text(
        (width * 2 + 10, 10),
        "PREDICTION",
        fill="white",
    )

    canvas.save(
        output_path
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    threshold,
    output_dir,
):

    model.eval()

    total_loss = 0.0

    metric_totals = {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    total_samples = 0

    predictions_dir = (
        output_dir / "predictions"
    )

    visualizations_dir = (
        output_dir / "visualizations"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualizations_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_results = []

    for batch_idx, batch in enumerate(
        loader
    ):

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            images
        )

        loss = bce_dice_loss(
            logits,
            masks,
        )

        total_loss += (
            loss.item()
            * images.shape[0]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            logits,
            masks,
            threshold=threshold,
        )

        batch_size = images.shape[0]

        for key in metric_totals:

            metric_totals[key] += (
                metrics[key]
                .sum()
                .item()
            )

        total_samples += batch_size

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= threshold
        ).float()

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        masks_np = (
            masks
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Process individual images
        # ----------------------------------------------------

        for i in range(batch_size):

            image_path = Path(
                batch["image_path"][i]
            )

            stem = image_path.stem

            pred_mask = (
                predictions_np[i, 0]
            )

            gt_mask = (
                masks_np[i, 0]
            )

            # Save binary mask.
            pred_image = Image.fromarray(
                (
                    pred_mask * 255
                ).astype(np.uint8)
            )

            pred_image.save(
                predictions_dir
                / f"{stem}_pred.png"
            )

            # Visualization.
            create_visualization(
                image_path,
                gt_mask,
                pred_mask,
                visualizations_dir
                / f"{stem}_comparison.jpg",
            )

            # Per-image metrics.
            per_image_results.append(
                {
                    "image": str(
                        image_path
                    ),
                    "dice": float(
                        metrics["dice"][i]
                        .item()
                    ),
                    "iou": float(
                        metrics["iou"][i]
                        .item()
                    ),
                    "precision": float(
                        metrics[
                            "precision"
                        ][i].item()
                    ),
                    "recall": float(
                        metrics[
                            "recall"
                        ][i].item()
                    ),
                    "accuracy": float(
                        metrics[
                            "accuracy"
                        ][i].item()
                    ),
                }
            )

        print(
            f"Validated batch "
            f"{batch_idx + 1}/"
            f"{len(loader)}",
            end="\r",
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    results = {
        "loss": (
            total_loss
            / max(total_samples, 1)
        )
    }

    for key in metric_totals:

        results[key] = (
            metric_totals[key]
            / max(total_samples, 1)
        )

    print("\n")

    return results, per_image_results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate CoAtNet leaf "
            "segmentation model."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--masks",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/validation",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_everything(
        args.seed
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )

    training_cfg = config[
        "training"
    ]

    image_size = int(
        training_cfg.get(
            "image_size",
            224,
        )
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(
            training_cfg.get(
                "batch_size",
                4,
            )
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device:

        device = torch.device(
            args.device
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"\nDevice: {device}"
    )

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
    # Images
    # --------------------------------------------------------

    all_images = collect_images(
        args.images
    )

    print(
        f"\nTotal images: "
        f"{len(all_images)}"
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    val_images = split_dataset(
        all_images,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        f"Validation images: "
        f"{len(val_images)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LeafValidationDataset(
        val_images,
        args.masks,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nCreating model..."
    )

    model = build_model(
        config
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "\nStarting validation..."
    )

    results, per_image_results = validate(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Loss      : "
        f"{results['loss']:.4f}"
    )

    print(
        f"Dice      : "
        f"{results['dice']:.4f}"
    )

    print(
        f"IoU       : "
        f"{results['iou']:.4f}"
    )

    print(
        f"Precision : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{results['recall']:.4f}"
    )

    print(
        f"Accuracy  : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Threshold : "
        f"{args.threshold:.2f}"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results["checkpoint"] = str(
        args.checkpoint
    )

    results["num_validation_images"] = (
        len(val_images)
    )

    results["image_size"] = image_size

    results["threshold"] = (
        args.threshold
    )

    with open(
        output_dir / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    with open(
        output_dir / "per_image_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            per_image_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save validation image list
    # --------------------------------------------------------

    with open(
        output_dir / "validation_images.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for path in val_images:

            f.write(
                str(path)
                + "\n"
            )

    print(
        "\nSaved:"
    )

    print(
        f"  Metrics:"
        f" {output_dir / 'validation_metrics.json'}"
    )

    print(
        f"  Per-image:"
        f" {output_dir / 'per_image_metrics.json'}"
    )

    print(
        f"  Predictions:"
        f" {output_dir / 'predictions'}"
    )

    print(
        f"  Visualizations:"
        f" {output_dir / 'visualizations'}"
    )


if __name__ == "__main__":
    main()