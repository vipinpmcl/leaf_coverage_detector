"""
predict.py

CoAtNet-2 Leaf Segmentation Prediction Script.

Supports:
    1. Single image
    2. Directory of images

Outputs:
    - Binary predicted mask
    - Probability mask
    - Overlay
    - Side-by-side visualization

Example:

Single image:

python predict.py \
    --config configs/default.yaml \
    --checkpoint runs/coatnet_leaf/best_stage_A.pth \
    --image test.jpg \
    --output runs/predictions


Directory:

python predict.py \
    --config configs/default.yaml \
    --checkpoint runs/coatnet_leaf/best_stage_A.pth \
    --input-dir data/test \
    --output runs/predictions


Change threshold:

python predict.py \
    --config configs/default.yaml \
    --checkpoint runs/coatnet_leaf/best_stage_A.pth \
    --image test.jpg \
    --output runs/predictions \
    --threshold 0.4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
import yaml

from models import CoAtNetLeafDetector


# ============================================================
# Supported image extensions
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# Load configuration
# ============================================================

def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    return config


# ============================================================
# Build model
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
# Load checkpoint
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
    # Standard training checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch'] + 1}"
            )

        if "stage" in checkpoint:

            print(
                f"Checkpoint stage: "
                f"{checkpoint['stage']}"
            )

        if "best_metric" in checkpoint:

            print(
                f"Best metric: "
                f"{checkpoint['best_metric']:.4f}"
            )

    # --------------------------------------------------------
    # Raw state dict
    # --------------------------------------------------------

    elif isinstance(
        checkpoint,
        dict,
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing:

        print(
            f"\nWarning: "
            f"{len(missing)} missing keys."
        )

        for key in missing[:10]:

            print(
                f"  MISSING: {key}"
            )

    if unexpected:

        print(
            f"\nWarning: "
            f"{len(unexpected)} unexpected keys."
        )

        for key in unexpected[:10]:

            print(
                f"  UNEXPECTED: {key}"
            )

    print(
        "\nCheckpoint loaded successfully."
    )

    return checkpoint


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(
    image,
    image_size,
):

    # --------------------------------------------------------
    # Resize to CoAtNet input size.
    #
    # The pretrained checkpoint uses:
    #   3 x 224 x 224
    # --------------------------------------------------------

    resized = image.resize(
        (
            image_size,
            image_size,
        ),
        Image.Resampling.BICUBIC,
    )

    # PIL -> Tensor
    tensor = TF.to_tensor(
        resized
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Your CoAtNet pretrained configuration specifies:
    #
    # mean = (0.5, 0.5, 0.5)
    # std  = (0.5, 0.5, 0.5)
    # --------------------------------------------------------

    tensor = TF.normalize(
        tensor,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # [C,H,W] -> [1,C,H,W]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# Overlay
# ============================================================

def create_overlay(
    image,
    mask,
    alpha=0.45,
):

    image_array = np.array(
        image
    ).astype(np.float32)

    # --------------------------------------------------------
    # Create green-ish mask.
    #
    # We intentionally use RGB construction rather than
    # modifying the original image directly.
    # --------------------------------------------------------

    mask_rgb = np.zeros_like(
        image_array
    )

    mask_rgb[:, :, 0] = 0
    mask_rgb[:, :, 1] = 255
    mask_rgb[:, :, 2] = 0

    mask_bool = (
        mask > 0
    )

    overlay = image_array.copy()

    overlay[mask_bool] = (
        (1 - alpha)
        * overlay[mask_bool]
        +
        alpha
        * mask_rgb[mask_bool]
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        overlay
    )


# ============================================================
# Side-by-side visualization
# ============================================================

def create_comparison(
    image,
    mask,
    overlay,
):

    width, height = image.size

    mask_image = Image.fromarray(
        mask.astype(np.uint8)
    )

    mask_rgb = Image.merge(
        "RGB",
        (
            mask_image,
            mask_image,
            mask_image,
        ),
    )

    canvas = Image.new(
        "RGB",
        (
            width * 3,
            height,
        ),
    )

    canvas.paste(
        image,
        (0, 0),
    )

    canvas.paste(
        mask_rgb,
        (width, 0),
    )

    canvas.paste(
        overlay,
        (width * 2, 0),
    )

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
        "PREDICTED MASK",
        fill="white",
    )

    draw.text(
        (width * 2 + 10, 10),
        "OVERLAY",
        fill="white",
    )

    return canvas


# ============================================================
# Single image prediction
# ============================================================

@torch.no_grad()
def predict_image(
    model,
    image_path,
    output_dir,
    image_size,
    threshold,
    device,
):
    image_path = Path(image_path)

    print(f"\nPredicting: {image_path}")

    # --------------------------------------------------------
    # Image-specific output directory
    # --------------------------------------------------------

    image_output_dir = (
        output_dir / image_path.stem
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    # Save a copy of original image
    original_output = (
        image_output_dir
        / "original.jpg"
    )

    image.save(
        original_output,
        quality=95,
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor = preprocess_image(
        image,
        image_size,
    )

    tensor = tensor.to(device)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    logits = model(
        tensor,
        output_size=(
            original_size[1],
            original_size[0],
        ),
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = torch.sigmoid(
        logits
    )

    probability = (
        probability[0, 0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    binary_mask = (
        probability >= threshold
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Save binary mask
    # --------------------------------------------------------

    mask_path = (
        image_output_dir
        / "mask.png"
    )

    mask_image = Image.fromarray(
        binary_mask * 255
    )

    mask_image.save(
        mask_path
    )

    # --------------------------------------------------------
    # Save probability mask
    # --------------------------------------------------------

    probability_path = (
        image_output_dir
        / "probability.png"
    )

    probability_image = Image.fromarray(
        (
            probability * 255
        ).clip(
            0,
            255,
        ).astype(np.uint8)
    )

    probability_image.save(
        probability_path
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image,
        binary_mask,
    )

    overlay_path = (
        image_output_dir
        / "overlay.jpg"
    )

    overlay.save(
        overlay_path,
        quality=95,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = create_comparison(
        image,
        binary_mask * 255,
        overlay,
    )

    comparison_path = (
        image_output_dir
        / "comparison.jpg"
    )

    comparison.save(
        comparison_path,
        quality=95,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    leaf_pixels = int(
        binary_mask.sum()
    )

    total_pixels = (
        binary_mask.shape[0]
        * binary_mask.shape[1]
    )

    leaf_percentage = (
        100.0
        * leaf_pixels
        / max(total_pixels, 1)
    )

    max_probability = float(
        probability.max()
    )

    mean_probability = float(
        probability.mean()
    )

    if leaf_pixels > 0:
        leaf_probability = float(
            probability[
                binary_mask == 1
            ].mean()
        )
    else:
        leaf_probability = 0.0

    print(
        f"  Leaf coverage: "
        f"{leaf_percentage:.2f}%"
    )

    print(
        f"  Max probability: "
        f"{max_probability:.4f}"
    )

    print(
        f"  Output: "
        f"{image_output_dir}"
    )

    return {
        "image": str(image_path),
        "width": original_size[0],
        "height": original_size[1],
        "leaf_pixels": leaf_pixels,
        "leaf_percentage": leaf_percentage,
        "max_probability": max_probability,
        "mean_probability": mean_probability,
        "leaf_probability": leaf_probability,
        "threshold": threshold,

        "output_dir": str(
            image_output_dir
        ),

        "original": str(
            original_output
        ),

        "mask": str(
            mask_path
        ),

        "probability": str(
            probability_path
        ),

        "overlay": str(
            overlay_path
        ),

        "comparison": str(
            comparison_path
        ),
    }

# ============================================================
# Find images
# ============================================================

def collect_images(
    input_dir,
):

    input_dir = Path(
        input_dir
    )

    images = [
        p
        for p in input_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    images.sort()

    return images


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CoAtNet-2 leaf "
            "segmentation prediction."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained segmentation checkpoint.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image to predict.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/predictions",
        help="Output directory.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if (
        args.image is None
        and args.input_dir is None
    ):

        parser.error(
            "Provide either "
            "--image or --input-dir."
        )

    if (
        args.image is not None
        and args.input_dir is not None
    ):

        parser.error(
            "Use only one of "
            "--image or --input-dir."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    image_size = int(
        config["training"].get(
            "image_size",
            224,
        )
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
    # Threshold validation
    # --------------------------------------------------------

    if not (
        0.0
        < args.threshold
        < 1.0
    ):

        raise ValueError(
            "--threshold must be between "
            "0 and 1."
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        "\nBuilding model..."
    )

    model = build_model(
        config
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Determine images
    # --------------------------------------------------------

    if args.image:

        image_path = Path(
            args.image
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        image_paths = [
            image_path
        ]

    else:

        image_paths = collect_images(
            args.input_dir
        )

        if not image_paths:

            raise RuntimeError(
                f"No images found in "
                f"{args.input_dir}"
            )

    print(
        f"\nImages to process: "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    results = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\n[{index}/{len(image_paths)}]"
        )

        result = predict_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            image_size=image_size,
            threshold=args.threshold,
            device=device,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results_path = (
        output_dir
        / "predictions.json"
    )

    import json

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PREDICTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Images processed : "
        f"{len(results)}"
    )

    print(
        f"Threshold        : "
        f"{args.threshold:.2f}"
    )

    print(
        f"Output directory : "
        f"{output_dir}"
    )

    print(
        f"Results JSON     : "
        f"{results_path}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()