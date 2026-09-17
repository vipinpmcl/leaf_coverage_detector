import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import numpy as np
from PIL import Image

from src.dinov2 import DINOv2Backbone
from src.model import LeafSegmentationModel
from src.visualize import save_prediction_outputs
from src.components import connected_components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    image = Image.open(args.image).convert("RGB")
    x = backbone.transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = torch.sigmoid(model(x)[0])

    output_dir = Path(args.output)

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
    leaf_coverage = 100.0 * leaf_pixels / total

    components = connected_components(mask_np)

    if components:
        largest = components[0]
        largest_result = {
            "area_patches": largest["area_patches"],
            "coverage_percent": (
                100.0 * largest["area_patches"] / total
            ),
            "bbox": {
                "min_row": largest["min_row"],
                "min_col": largest["min_col"],
                "max_row": largest["max_row"],
                "max_col": largest["max_col"]
            },
            "bbox_width_patches": largest["width_patches"],
            "bbox_height_patches": largest["height_patches"]
        }
    else:
        largest_result = None

    result = {
        "image": args.image,
        "threshold": args.threshold,
        "image_width": image.width,
        "image_height": image.height,
        "leaf_pixels": leaf_pixels,
        "total_pixels": total,
        "leaf_coverage_percent": leaf_coverage,
        "number_of_components": len(components),
        "largest_component": largest_result
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
