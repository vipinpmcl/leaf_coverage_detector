from pathlib import Path

import numpy as np
from PIL import Image


def save_prediction_outputs(
    image,
    probability,
    output_dir,
    threshold=0.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = image.convert("RGB")
    original.save(output_dir / "original.jpg", quality=95)

    prob = np.clip(probability * 255, 0, 255).astype(np.uint8)
    Image.fromarray(prob).save(output_dir / "probability.png")

    mask = (probability >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_dir / "mask.png")

    rgb = np.asarray(original).astype(np.float32)
    overlay = rgb.copy()

    # Red tint on predicted leaf pixels.
    leaf = mask > 0
    overlay[leaf, 0] = 0.55 * overlay[leaf, 0] + 0.45 * 255
    overlay[leaf, 1] *= 0.55
    overlay[leaf, 2] *= 0.55

    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(
        output_dir / "overlay.jpg",
        quality=95,
    )

    comparison = Image.new(
        "RGB",
        (original.width * 2, original.height),
    )
    comparison.paste(original, (0, 0))
    comparison.paste(
        Image.fromarray(np.uint8(np.clip(overlay, 0, 255))),
        (original.width, 0),
    )
    comparison.save(output_dir / "comparison.jpg", quality=95)
