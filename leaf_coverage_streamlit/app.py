from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from models import CoAtNetLeafDetector


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# ---------------------------------------------------------------------
# Configuration / model loading
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_state_dict(checkpoint):
    """Accept common PyTorch checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        # A plain state_dict is also a dict of tensors.
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError(
        "Could not find a model state_dict in the trained checkpoint."
    )


def _clean_state_dict(state_dict):
    cleaned = {}
    prefixes = ("module.", "model.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


@st.cache_resource(show_spinner="Loading leaf coverage model...")
def load_model(config_path: str, checkpoint_path: str, foundation_path: str, device_name: str):
    cfg = load_config(config_path)

    device = torch.device(device_name)
    mc = cfg["model"]

    model = CoAtNetLeafDetector(
        foundation_checkpoint=foundation_path,
        model_name=mc["name"],
        out_indices=tuple(mc["out_indices"]),
        backbone_channels=tuple(mc["backbone_channels"]),
        adapter_channels=tuple(mc["adapter_channels"]),
        decoder_channels=mc["decoder_channels"],
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = _clean_state_dict(_extract_state_dict(checkpoint))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Fail loudly when the trained checkpoint is clearly incompatible.
    if len(missing) > max(10, int(0.15 * len(model.state_dict()))):
        raise RuntimeError(
            f"Too many missing model parameters ({len(missing)}). "
            "The trained checkpoint does not appear to match this architecture."
        )

    model.eval()

    return model, cfg, device, missing, unexpected


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_url_image(url: str) -> tuple[Image.Image, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "LeafCoverageDetector/1.0"},
    )
    response.raise_for_status()

    image = load_pil_from_bytes(response.content)
    name = Path(parsed.path).name or "internet_image.jpg"
    return image, name


def find_images(directory: str) -> list[Path]:
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    # Recursive search so nested crop/sample folders are supported.
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def predict_image(
    model,
    image: Image.Image,
    cfg: dict,
    device: torch.device,
    threshold: float,
):
    original_w, original_h = image.size

    resized = TF.resize(
        image,
        [cfg["data"]["image_size"], cfg["data"]["image_size"]],
        antialias=True,
    )

    x = TF.to_tensor(resized)
    x = TF.normalize(
        x,
        tuple(cfg["normalization"]["mean"]),
        tuple(cfg["normalization"]["std"]),
    )
    x = x.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(
            x,
            output_size=(original_h, original_w),
        )
        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    mask = probability >= threshold

    leaf_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    coverage = 100.0 * leaf_pixels / max(total_pixels, 1)

    mean_probability = float(probability.mean())

    if leaf_pixels:
        leaf_probability = float(probability[mask].mean())
        non_leaf_probability = float(probability[~mask].mean())
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        bbox_w = bbox[2] - bbox[0] + 1
        bbox_h = bbox[3] - bbox[1] + 1
    else:
        leaf_probability = 0.0
        non_leaf_probability = float(probability.mean())
        bbox = None
        bbox_w = bbox_h = 0

    return {
        "probability": probability,
        "mask": mask,
        "coverage": coverage,
        "leaf_pixels": leaf_pixels,
        "total_pixels": total_pixels,
        "mean_probability": mean_probability,
        "leaf_probability": leaf_probability,
        "non_leaf_probability": non_leaf_probability,
        "bbox": bbox,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "width": original_w,
        "height": original_h,
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    overlay = rgb.copy()

    # Green leaf mask without requiring OpenCV.
    leaf_color = np.zeros_like(rgb)
    leaf_color[..., 1] = 255.0

    overlay[mask] = (
        (1.0 - alpha) * rgb[mask] + alpha * leaf_color[mask]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_mask_image(mask: np.ndarray):
    return (mask.astype(np.uint8) * 255)


def make_probability_image(probability: np.ndarray):
    # Streamlit/PIL can display this directly as a grayscale probability map.
    return np.clip(probability * 255.0, 0, 255).astype(np.uint8)


def make_comparison(image: Image.Image, overlay: np.ndarray):
    left = np.asarray(image.convert("RGB"))
    right = overlay

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    return canvas


def png_bytes(array_or_image) -> bytes:
    if isinstance(array_or_image, Image.Image):
        image = array_or_image
    else:
        arr = np.asarray(array_or_image)
        image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------
# Single result UI
# ---------------------------------------------------------------------

def show_metrics(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leaf coverage", f'{result["coverage"]:.2f}%')
    c2.metric("Leaf probability", f'{result["leaf_probability"]:.3f}')
    c3.metric("Leaf pixels", f'{result["leaf_pixels"]:,}')
    c4.metric("Resolution", f'{result["width"]} × {result["height"]}')

    c5, c6, c7 = st.columns(3)
    c5.metric("Mean probability", f'{result["mean_probability"]:.3f}')
    c6.metric("BBox width", f'{result["bbox_width"]:,} px')
    c7.metric("BBox height", f'{result["bbox_height"]:,} px')


def show_single_result(image: Image.Image, result, threshold: float):
    overlay = make_overlay(image, result["mask"])
    mask_img = make_mask_image(result["mask"])
    prob_img = make_probability_image(result["probability"])

    show_metrics(result)

    st.markdown("### Visualization")

    # c1, c2 = st.columns(2)
    # with c1:
    #     st.image(image, caption="Original", use_container_width=True)
    # with c2:
    #     st.image(overlay, caption="Leaf overlay", use_container_width=True)

    st.image(
        make_comparison(image, overlay),
        caption="Original  |  Leaf overlay",
        use_container_width=True,
    )

    with st.expander("Probability map and binary mask"):
        p1, p2 = st.columns(2)
        with p1:
            st.image(prob_img, caption="Pixel probability", use_container_width=True)
        with p2:
            st.image(mask_img, caption=f"Binary mask @ {threshold:.2f}", use_container_width=True)

    st.download_button(
        "Download overlay (PNG)",
        data=png_bytes(overlay),
        file_name="leaf_overlay.png",
        mime="image/png",
    )


# ---------------------------------------------------------------------
# Batch UI
# ---------------------------------------------------------------------

def process_batch(
    model,
    items,
    cfg,
    device,
    threshold,
):
    rows = []
    visual_results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(items)

    for i, (name, image) in enumerate(items, start=1):
        try:
            result = predict_image(
                model=model,
                image=image,
                cfg=cfg,
                device=device,
                threshold=threshold,
            )

            rows.append({
                "image": name,
                "coverage_%": round(result["coverage"], 4),
                "leaf_probability": round(result["leaf_probability"], 4),
                "mean_probability": round(result["mean_probability"], 4),
                "leaf_pixels": result["leaf_pixels"],
                "width": result["width"],
                "height": result["height"],
                "bbox_width": result["bbox_width"],
                "bbox_height": result["bbox_height"],
            })

            visual_results.append((name, image, result))

        except Exception as exc:
            rows.append({
                "image": name,
                "coverage_%": np.nan,
                "leaf_probability": np.nan,
                "mean_probability": np.nan,
                "leaf_pixels": np.nan,
                "width": np.nan,
                "height": np.nan,
                "bbox_width": np.nan,
                "bbox_height": np.nan,
                "error": str(exc),
            })

        progress.progress(i / total)
        status.write(f"Processed {i}/{total}: {name}")

    progress.empty()
    status.empty()

    return pd.DataFrame(rows), visual_results


# ---------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Leaf Coverage Detector",
    page_icon="🍃",
    layout="wide",
)

st.title("🍃 Leaf Coverage Detector")
st.caption(
    "CoAtNet2 + multi-scale decoder leaf segmentation — "
    "single image, local directory, uploaded images, or internet URL."
)

# Sidebar configuration
with st.sidebar:
    st.header("Model")

    config_path = st.text_input(
        "Config YAML",
        value="config.yaml",
        help="Path to the YAML configuration used for training.",
    )

    checkpoint_path = st.text_input(
        "Trained model checkpoint",
        value="../experiments/9.3.coatnet_2_rw_224_experimentation/runs/exp11_7/best.pt",
        help="Path to the trained leaf segmentation checkpoint.",
    )

    default_foundation = "../experiments/9.3.coatnet_2_rw_224_experimentation/checkpoints/crop_fm1_coatNet2rw224.pth"
    foundation_path = st.text_input(
        "Foundation checkpoint",
        value=default_foundation,
        help="CoAtNet foundation checkpoint used by the model.",
    )

    device_options = ["auto", "cuda", "cpu"]
    device_choice = st.selectbox("Device", device_options, index=0)

    if device_choice == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = device_choice

    st.caption(f"Using device: `{device_name}`")

    st.divider()

    threshold = st.slider(
        "Leaf mask threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.60,
        step=0.01,
        help="Pixels with probability >= threshold are counted as leaf.",
    )

    overlay_alpha = st.slider(
        "Overlay opacity",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )

# Validate model files before loading.
if not Path(config_path).exists():
    st.info(f"Config file not found: `{config_path}`")
    st.stop()

if not Path(checkpoint_path).exists():
    st.info(
        f"Trained checkpoint not found: `{checkpoint_path}`. "
        "Set its path in the sidebar."
    )
    st.stop()

if not Path(foundation_path).exists():
    st.info(
        f"Foundation checkpoint not found: `{foundation_path}`. "
        "Set its path in the sidebar."
    )
    st.stop()

try:
    model, cfg, device, missing_keys, unexpected_keys = load_model(
        config_path,
        checkpoint_path,
        foundation_path,
        device_name,
    )
except Exception as exc:
    st.error("Model loading failed.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.success("Model loaded")

    if missing_keys:
        st.warning(f"{len(missing_keys)} checkpoint keys were missing.")
    if unexpected_keys:
        st.warning(f"{len(unexpected_keys)} unexpected checkpoint keys found.")

# Tabs
single_tab, batch_tab = st.tabs(["🖼️ Single Image", "📁 Directory / Batch"])

with single_tab:
    source = st.radio(
        "Input source",
        ["Upload image", "Local image path", "Internet URL"],
        horizontal=True,
    )

    image = None
    image_name = "image"

    if source == "Upload image":
        uploaded = st.file_uploader(
            "Choose an image",
            type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)],
        )
        if uploaded:
            image = load_pil_from_bytes(uploaded.getvalue())
            image_name = uploaded.name

    elif source == "Local image path":
        local_path = st.text_input(
            "Image path",
            placeholder=r"D:\dataset\sample.jpg or /home/user/sample.jpg",
        )
        if local_path:
            p = Path(local_path).expanduser()
            if p.exists() and p.is_file():
                image = Image.open(p).convert("RGB")
                image_name = p.name
            else:
                st.warning("Image path does not exist.")

    else:
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/leaf.jpg",
        )
        if url:
            try:
                with st.spinner("Downloading image..."):
                    image, image_name = load_url_image(url)
            except Exception as exc:
                st.error(f"Could not load URL: {exc}")

    if image is not None:
        st.markdown(f"### {image_name}")

        result = predict_image(
            model=model,
            image=image,
            cfg=cfg,
            device=device,
            threshold=threshold,
        )

        # Re-render overlay with the UI-selected opacity.
        show_metrics(result)

        overlay = make_overlay(image, result["mask"], alpha=overlay_alpha)

        # c1, c2 = st.columns(2)
        # with c1:
        #     st.image(image, caption="Original", use_container_width=True)
        # with c2:
        #     st.image(overlay, caption="Leaf overlay", use_container_width=True)

        st.image(
            make_comparison(image, overlay),
            caption="Original  |  Leaf overlay",
            use_container_width=True,
        )

        p1, p2 = st.columns(2)
        with p1:
            st.image(
                make_probability_image(result["probability"]),
                caption="Pixel probability",
                use_container_width=True,
            )
        with p2:
            st.image(
                make_mask_image(result["mask"]),
                caption=f"Binary leaf mask @ {threshold:.2f}",
                use_container_width=True,
            )

        details = pd.DataFrame([{
            "Image": image_name,
            "Width": result["width"],
            "Height": result["height"],
            "Leaf coverage (%)": result["coverage"],
            "Leaf pixels": result["leaf_pixels"],
            "Mean probability": result["mean_probability"],
            "Leaf probability": result["leaf_probability"],
            "Non-leaf probability": result["non_leaf_probability"],
            "BBox width": result["bbox_width"],
            "BBox height": result["bbox_height"],
            "Threshold": threshold,
        }])

        with st.expander("Detailed metrics"):
            st.dataframe(details, use_container_width=True)

        st.download_button(
            "⬇️ Download overlay",
            data=png_bytes(overlay),
            file_name=f"{Path(image_name).stem}_leaf_overlay.png",
            mime="image/png",
        )

        st.download_button(
            "⬇️ Download binary mask",
            data=png_bytes(make_mask_image(result["mask"])),
            file_name=f"{Path(image_name).stem}_leaf_mask.png",
            mime="image/png",
        )

with batch_tab:
    st.markdown(
        "Process a local directory recursively. "
        "Nested crop/sample folders are supported."
    )

    batch_source = st.radio(
        "Batch input",
        ["Local directory", "Upload multiple images"],
        horizontal=True,
    )

    items = []

    if batch_source == "Local directory":
        directory = st.text_input(
            "Directory path",
            placeholder=r"D:\datasets\leaf_images",
        )

        if directory:
            try:
                paths = find_images(directory)
                st.write(f"Found **{len(paths)}** supported images.")

                max_images = st.number_input(
                    "Maximum images to process",
                    min_value=1,
                    max_value=max(1, len(paths)),
                    value=min(100, max(1, len(paths))),
                    step=1,
                )

                paths = paths[:int(max_images)]

                if st.button("🚀 Run batch prediction", type="primary"):
                    for p in paths:
                        try:
                            items.append((str(p), Image.open(p).convert("RGB")))
                        except Exception as exc:
                            st.warning(f"Could not open {p}: {exc}")

                    if items:
                        df, visual_results = process_batch(
                            model,
                            items,
                            cfg,
                            device,
                            threshold,
                        )

                        st.session_state["batch_df"] = df
                        st.session_state["batch_visual_results"] = visual_results

            except Exception as exc:
                st.error(str(exc))

    else:
        uploads = st.file_uploader(
            "Choose multiple images",
            type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )

        if uploads:
            st.write(f"Selected **{len(uploads)}** images.")

            if st.button("🚀 Run batch prediction", type="primary"):
                items = []
                for uploaded in uploads:
                    try:
                        items.append(
                            (uploaded.name, load_pil_from_bytes(uploaded.getvalue()))
                        )
                    except Exception as exc:
                        st.warning(f"Could not open {uploaded.name}: {exc}")

                if items:
                    df, visual_results = process_batch(
                        model,
                        items,
                        cfg,
                        device,
                        threshold,
                    )

                    st.session_state["batch_df"] = df
                    st.session_state["batch_visual_results"] = visual_results

    if "batch_df" in st.session_state:
        df = st.session_state["batch_df"]
        visual_results = st.session_state["batch_visual_results"]

        st.markdown("### Batch results")
        st.dataframe(df, use_container_width=True)

        valid = df["coverage_%"].dropna()

        if len(valid):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Images", len(df))
            m2.metric("Mean coverage", f"{valid.mean():.2f}%")
            m3.metric("Min coverage", f"{valid.min():.2f}%")
            m4.metric("Max coverage", f"{valid.max():.2f}%")

            st.markdown("### Coverage distribution")
            st.bar_chart(
                df.set_index("image")["coverage_%"],
                use_container_width=True,
            )

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="leaf_coverage_results.csv",
            mime="text/csv",
        )

        st.markdown("### Batch visualizations")

        # Avoid rendering hundreds of large images at once.
        preview_count = st.slider(
            "Number of result previews",
            min_value=1,
            max_value=min(20, len(visual_results)),
            value=min(6, len(visual_results)),
        )

        for name, image, result in visual_results[:preview_count]:
            with st.expander(
                f"{name} — coverage {result['coverage']:.2f}%"
            ):
                overlay = make_overlay(
                    image,
                    result["mask"],
                    alpha=overlay_alpha,
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Original", use_container_width=True)
                with c2:
                    st.image(overlay, caption="Overlay", use_container_width=True)
                show_metrics(result)
