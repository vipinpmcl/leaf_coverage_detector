from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from models import CoAtNetLeafDetector
from utils.checkpoint import load_model_checkpoint
from utils.visualize import save_prediction_outputs


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def get_image_paths(image_dir: Path):
    """Return all supported image files from a directory."""

    image_paths = [
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def predict_image(
    model,
    image_path: Path,
    cfg,
    device,
    output_dir: Path,
):
    """Run prediction for one image."""

    image = Image.open(image_path).convert("RGB")

    # ---------------------------------------------------------
    # Resize image for model input
    # ---------------------------------------------------------
    x = TF.resize(
        image,
        [
            cfg["data"]["image_size"],
            cfg["data"]["image_size"],
        ],
        antialias=True,
    )

    # PIL -> Tensor
    x = TF.to_tensor(x)

    # Normalization
    x = TF.normalize(
        x,
        tuple(cfg["normalization"]["mean"]),
        tuple(cfg["normalization"]["std"]),
    )

    # Add batch dimension
    x = x.unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    with torch.no_grad():

        logits = model(
            x,
            output_size=(
                image.height,
                image.width,
            ),
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    # ---------------------------------------------------------
    # Create separate folder for this image
    # ---------------------------------------------------------
    image_output_dir = output_dir / image_path.stem

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    save_prediction_outputs(
        image=image,
        probability=probability,
        output_dir=image_output_dir,
        threshold=cfg["training"]["threshold"],
    )

    print(
        f"[OK] {image_path.name} "
        f"-> {image_output_dir}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Batch leaf segmentation prediction"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint",
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing input images",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/batch_predictions",
        help="Directory where predictions will be saved",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load config
    # ---------------------------------------------------------
    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------
    mc = cfg["model"]

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    print("\nCreating model...")

    model = CoAtNetLeafDetector(
        foundation_checkpoint=mc[
            "foundation_checkpoint"
        ],
        model_name=mc["name"],
        out_indices=tuple(
            mc["out_indices"]
        ),
        backbone_channels=tuple(
            mc["backbone_channels"]
        ),
        adapter_channels=tuple(
            mc["adapter_channels"]
        ),
        decoder_channels=mc[
            "decoder_channels"
        ],
    ).to(device)

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------
    print("\nLoading trained checkpoint...")

    load_model_checkpoint(
        args.checkpoint,
        model,
        device,
    )

    model.eval()

    print("Model ready.")

    # ---------------------------------------------------------
    # Input directory
    # ---------------------------------------------------------
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{image_dir}"
        )

    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory:\n"
            f"{image_dir}"
        )

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------
    image_paths = get_image_paths(
        image_dir
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in:\n"
            f"{image_dir}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nFound {len(image_paths)} images."
    )

    print(
        f"Output directory: {output_dir}"
    )

    # ---------------------------------------------------------
    # Batch prediction
    # ---------------------------------------------------------
    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            predict_image(
                model=model,
                image_path=image_path,
                cfg=cfg,
                device=device,
                output_dir=output_dir,
            )

        except Exception as e:

            print(
                f"[ERROR] {image_path.name}: "
                f"{type(e).__name__}: {e}"
            )

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch prediction completed.")
    print(f"Results: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()