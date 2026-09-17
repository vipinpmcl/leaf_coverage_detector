import argparse, sys, json
from pathlib import Path

from matplotlib.pyplot import gray

from blur_texture_score import laplacian_variance
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import numpy as np
from PIL import Image

from src.dinov2 import DINOv2Backbone
from src.model import LeafSegmentationModel
from src.visualize import save_prediction_outputs
from src.components import connected_components
from src.sharpness import (
    laplacian_variance,
    tenengrad,
    mean_gradient_magnitude,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

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

    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    input_dir = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
    ])

    summary = []

    for path in images:
        print("\nProcessing:", path.name)

        image = Image.open(path).convert("RGB")
        x = backbone.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probability = torch.sigmoid(model(x)[0])

        output_dir = output_root / path.stem

        mask = save_prediction_outputs(
            image,
            probability,
            args.threshold,
            output_dir,
            cfg["output"]["alpha"]
        )

        mask_np = np.asarray(mask, dtype=np.uint8)
        total = mask_np.size
        leaf_pixels = int(mask_np.sum())

        leaf_coverage = (
            100.0 * leaf_pixels / total
        )

        components = connected_components(mask_np)

        if components:
            largest = components[0]
            crop_gray = gray[y:y+h, x:x+w]
            crop_mask = component_mask[y:y+h, x:x+w]
            largest_result = {
                "area_patches":
                    largest["area_patches"],

                "coverage_percent":
                    100.0
                    * largest["area_patches"]
                    / total,

                "bbox": {
                    "min_row":
                        largest["min_row"],
                    "min_col":
                        largest["min_col"],
                    "max_row":
                        largest["max_row"],
                    "max_col":
                        largest["max_col"]
                },

                "bbox_width_patches":
                    largest["width_patches"],

                "bbox_height_patches":
                    largest["height_patches"],

                "laplacian_variance": laplacian_variance(
                    crop_gray, crop_mask
                ),
                "tenengrad": tenengrad(
                    crop_gray, crop_mask
                ),
                "mean_gradient_magnitude": mean_gradient_magnitude(
                    crop_gray, crop_mask
                ),
            }

        else:
            largest_result = None

        result = {
            "image": str(path),
            "threshold": args.threshold,
            "image_width": image.width,
            "image_height": image.height,
            "leaf_pixels": leaf_pixels,
            "total_pixels": total,
            "leaf_coverage_percent": leaf_coverage,
            "number_of_components": len(components),
            "largest_component": largest_result
        }

        summary.append(result)

        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8"
        )

        print(
            f"Leaf coverage: {leaf_coverage:.2f}%"
        )

        if largest_result:
            print(
                "Largest component: "
                f"{largest_result['area_patches']} patches"
            )
            print(
                "Largest bbox: "
                f"{largest_result['bbox_width_patches']} x "
                f"{largest_result['bbox_height_patches']} patches"
            )

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nBatch processing complete.")
    print("Summary:", output_root / "summary.json")

if __name__ == "__main__":
    main()
