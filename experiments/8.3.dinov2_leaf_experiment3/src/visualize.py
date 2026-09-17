from pathlib import Path
import numpy as np
from PIL import Image

def save_prediction_outputs(image, probability, threshold, output_dir, alpha=0.45):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probability = probability.detach().cpu().numpy()
    mask = (probability >= threshold).astype(np.uint8)

    Image.fromarray(
        np.clip(probability * 255, 0, 255).astype(np.uint8)
    ).save(output_dir / "probability.png")

    Image.fromarray(mask * 255).save(output_dir / "mask.png")

    w, h = image.size
    mask_image = Image.fromarray(mask * 255).resize(
        (w, h), Image.Resampling.NEAREST
    )
    prob_image = Image.fromarray(
        np.clip(probability * 255, 0, 255).astype(np.uint8)
    ).resize((w, h), Image.Resampling.BILINEAR)

    mask_image.save(output_dir / "mask_resized.png")
    prob_image.save(output_dir / "probability_resized.png")

    base = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    m = np.asarray(mask_image).astype(np.float32) / 255.0

    overlay = base.copy()
    selected = m > 0.5
    overlay[selected] = (1-alpha) * overlay[selected] + alpha
    overlay[~selected] *= 0.8

    Image.fromarray(
        np.clip(overlay * 255, 0, 255).astype(np.uint8)
    ).save(output_dir / "overlay.png")

    return mask
