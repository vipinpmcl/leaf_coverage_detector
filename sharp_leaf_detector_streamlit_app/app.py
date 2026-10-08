from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import yaml
import cv2
import threading
import time
from collections import deque
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import functional as TF
from streamlit_webrtc import webrtc_streamer

from models import CoAtNetLeafDetector
from utils.quality_metrics import calculate_quality_metrics

SUPPORTED_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}


@st.cache_data(show_spinner=False)
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading leaf coverage model...")
def load_model(config_path, device_name):
    cfg = load_config(config_path)
    model_config = cfg["model"]
    checkpoint_path = resolve_model_path(
        model_config["trained_checkpoint"], config_path
    )
    foundation_path = resolve_model_path(
        model_config["foundation_checkpoint"], config_path
    )
    device = torch.device(device_name)
    mc = model_config
    model = CoAtNetLeafDetector(
        foundation_checkpoint=foundation_path,
        model_name=mc["name"],
        out_indices=tuple(mc["out_indices"]),
        backbone_channels=tuple(mc["backbone_channels"]),
        adapter_channels=tuple(mc["adapter_channels"]),
        decoder_channels=mc["decoder_channels"],
    ).to(device)

    try:
        from utils.checkpoint import load_model_checkpoint
        load_model_checkpoint(checkpoint_path, model, device)
    except ImportError:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break
        cleaned = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            for prefix in ("module.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        missing, _ = model.load_state_dict(cleaned, strict=False)
        if len(missing) > max(10, int(.15*len(model.state_dict()))):
            raise RuntimeError(f"Too many missing checkpoint parameters: {len(missing)}")
    model.eval()
    return model, cfg, device


def resolve_model_path(path, config_path):
    """Resolve a model path from config, supporting config-relative paths."""
    value = str(path).strip()
    # Accept /d/path notation in configs created in a Linux/Git Bash environment.
    if len(value) > 3 and value[0] == "/" and value[1].isalpha() and value[2] == "/":
        value = f"{value[1].upper()}:{value[2:]}"
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = Path(config_path).expanduser().resolve().parent / resolved
    return resolved


def find_images(folder):
    root = Path(folder).expanduser()
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def segregate_images_by_coverage(model, cfg, device, source_folder, output_folder,
                                 mask_threshold, coverage_threshold, operation,
                                 progress_callback=None):
    """Predict coverage and copy/move images into below/at-or-above folders."""
    source = Path(source_folder).expanduser().resolve()
    output = Path(output_folder).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Select an existing source folder.")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Choose a destination folder that does not overlap the source folder.")

    paths = find_images(source)
    if not paths:
        raise ValueError("No supported images were found in the source folder.")

    low_dir = output / f"below_{coverage_threshold:g}_percent"
    high_dir = output / f"at_or_above_{coverage_threshold:g}_percent"
    results = []
    for index, path in enumerate(paths, 1):
        try:
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            prediction = predict_image(model, image, cfg, device, mask_threshold)
            coverage = prediction["metrics"]["leaf_coverage_percent"]
            group = low_dir if coverage < coverage_threshold else high_dir
            destination = group / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if operation == "Move":
                shutil.move(str(path), str(destination))
            else:
                shutil.copy2(path, destination)
            results.append({"image": str(path), "leaf_coverage_percent": coverage,
                            "group": group.name, "result": "success"})
        except Exception as exc:
            results.append({"image": str(path), "leaf_coverage_percent": np.nan,
                            "group": "", "result": f"error: {exc}"})
        if progress_callback:
            progress_callback(index / len(paths))
    return pd.DataFrame(results)


def browse_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select image folder")
        root.destroy()
        return selected
    except Exception as exc:
        st.warning(f"Native folder picker unavailable; enter the path manually. ({exc})")
        return ""


def load_url_image(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")
    response = requests.get(url, timeout=30, headers={"User-Agent":"LeafCoverageDetector/1.0"})
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB"), (Path(parsed.path).name or "internet_image.jpg")


def predict_image(model, image, cfg, device, threshold):
    w, h = image.size
    x = TF.resize(image, [cfg["data"]["image_size"], cfg["data"]["image_size"]], antialias=True)
    x = TF.normalize(TF.to_tensor(x), tuple(cfg["normalization"]["mean"]), tuple(cfg["normalization"]["std"]))
    x = x.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x, output_size=(h, w))
        probability = torch.sigmoid(logits)[0,0].float().cpu().numpy()

    mask = probability >= threshold
    metrics = calculate_quality_metrics(image, mask, probability)
    metrics["mean_probability"] = float(probability.mean())
    metrics["threshold"] = threshold
    return {"probability": probability, "mask": mask, "metrics": metrics}


def make_live_camera_callback(model, cfg, device, threshold, overlay_alpha, interval):
    """Create a WebRTC callback that overlays smoothed leaf coverage on live frames."""
    state = {"last_time": 0.0, "last_mask": None, "coverage": None, "coverages": deque(maxlen=8)}
    lock = threading.Lock()

    def render_overlay(bgr, mask, coverage):
        overlay = bgr.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        overlay[mask] = ((1 - overlay_alpha) * overlay[mask] + overlay_alpha * green[mask]).astype(np.uint8)
        cv2.rectangle(overlay, (10, 10), (300, 62), (0, 0, 0), -1)
        cv2.putText(overlay, f"Leaf coverage: {coverage:.1f}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return overlay

    def process_frame(frame):
        bgr = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        with lock:
            if state["last_mask"] is not None and now - state["last_time"] < interval:
                current_mask = cv2.resize(state["last_mask"].astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                return av.VideoFrame.from_ndarray(render_overlay(bgr, current_mask, state["coverage"]), format="bgr24")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            height, width = rgb.shape[:2]
            x = TF.resize(image, [cfg["data"]["image_size"], cfg["data"]["image_size"]], antialias=True)
            x = TF.normalize(TF.to_tensor(x), tuple(cfg["normalization"]["mean"]), tuple(cfg["normalization"]["std"]))
            x = x.unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(x, output_size=(height, width))
                probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

            mask = probability >= threshold
            coverage = float(mask.mean() * 100)
            state["coverages"].append(coverage)
            smoothed = sum(state["coverages"]) / len(state["coverages"])

            state["last_time"] = now
            state["last_mask"] = mask
            state["coverage"] = smoothed
            overlay = render_overlay(bgr, mask, smoothed)
            return av.VideoFrame.from_ndarray(overlay, format="bgr24")

    import av
    return process_frame


def make_overlay(image, mask, alpha=.45):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    overlay = rgb.copy()
    green = np.zeros_like(rgb)
    green[...,1] = 255
    overlay[mask] = (1-alpha)*rgb[mask] + alpha*green[mask]
    return np.clip(overlay,0,255).astype(np.uint8)


def detect_leaf_veins(image, leaf_mask):
    """Estimate thin vein-like ridges with a multiscale, multi-angle Gabor bank."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    response = np.zeros(gray.shape, dtype=np.float32)
    for wavelength in (6.0, 10.0, 16.0):
        for angle in np.arange(0, np.pi, np.pi / 12):
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=3.0, theta=float(angle), lambd=wavelength,
                gamma=0.5, psi=0, ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            response = np.maximum(response, np.abs(filtered))

    valid = np.asarray(leaf_mask, dtype=bool)
    if not valid.any():
        return np.zeros(valid.shape, dtype=bool)
    local_responses = response[valid]
    cutoff = float(np.percentile(local_responses, 96))
    veins = (response >= cutoff) & valid
    # Remove isolated specks while preserving narrow, connected line responses.
    return cv2.morphologyEx(
        veins.astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)


def make_vein_overlay(image, veins, alpha=0.9):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    color = np.zeros_like(rgb)
    color[..., 2] = 255  # cyan in RGB
    rgb[veins] = (1 - alpha) * rgb[veins] + alpha * color[veins]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def png_bytes(arr):
    image = arr if isinstance(arr, Image.Image) else Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def process_video(video_bytes, model, cfg, device, threshold, alpha, progress_callback=None, suffix=".mp4"):
    """Run leaf segmentation on each frame and return an overlaid MP4."""
    # Use a temporary file so uploaded videos work consistently on Windows.
    import tempfile
    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(video_bytes)
            temp_path = temp.name
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("Could not read this video. Try MP4, AVI, or MOV.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("Could not read the video dimensions.")

        output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output.name
        output.close()
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the output video on this system.")
        try:
            processed = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                result = predict_image(model, image, cfg, device, threshold)
                overlay = make_overlay(image, result["mask"], alpha)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                processed += 1
                if progress_callback and frame_count > 0:
                    progress_callback(min(processed / frame_count, 1.0))
            if processed == 0:
                raise ValueError("The uploaded video contains no readable frames.")
        finally:
            writer.release()
        return Path(output_path).read_bytes(), processed, fps
    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        if 'output_path' in locals():
            Path(output_path).unlink(missing_ok=True)


def show_metrics(m):
    a,b,c,d = st.columns(4)
    a.metric("Leaf coverage", f'{m["leaf_coverage_percent"]:.2f}%')
    b.metric("Leaf probability", f'{m["leaf_mean_probability"]:.3f}')
    c.metric("Masked Laplacian", f'{m["masked_laplacian_variance"]:.6f}')
    d.metric("Resolution", f'{m["image_width"]} × {m["image_height"]}')
    a,b,c,d = st.columns(4)
    a.metric("Masked Tenengrad", f'{m["masked_tenengrad"]:.5f}')
    b.metric("Masked Brenner", f'{m["masked_brenner"]:.5f}')
    c.metric("Masked edge density", f'{m["masked_edge_density"]:.3f}')
    d.metric("BBox fill", f'{m["leaf_bbox_fill_percent"]:.1f}%')
    st.metric(
        "Coverage-weighted edge-saturation score",
        f'{m["coverage_weighted_edge_saturation_score"]:.4f}',
        help="(Leaf coverage percent / 100) × masked edge density × masked saturation mean. All factors are on a 0–1 scale; an empty foreground mask gives 0.",
    )


def run_batch(model, items, cfg, device, threshold):
    rows, visuals = [], []
    progress = st.progress(0)
    for i,(name,image) in enumerate(items,1):
        try:
            r = predict_image(model,image,cfg,device,threshold)
            rows.append({
                "image": name,
                **{feature: r["metrics"][feature] for feature in CLUSTER_FEATURES},
            })
            visuals.append((name,image,r))
        except Exception as exc:
            rows.append({"image":name,"error":str(exc)})
        progress.progress(i/len(items))
    progress.empty()
    return pd.DataFrame(rows), visuals


CLUSTER_FEATURES = [
    "masked_tenengrad",
    "masked_fft_high_frequency_ratio",
    "masked_edge_density",
    "masked_brightness_mean",
    "masked_brightness_std",
    "masked_saturation_mean",
]
EMBEDDING_VERSION = 4


def assign_parameter_clusters(df, n_clusters):
    """Embed the selected quality metrics and cluster each metric independently."""
    feature_columns = [column for column in CLUSTER_FEATURES if column in df.columns]
    if not feature_columns:
        raise ValueError("No image-processing metrics are available for clustering.")
    valid = df[["image", *feature_columns]].dropna(
        subset=feature_columns,
        how="all",
    ).copy()
    if len(valid) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} valid images; found {len(valid)}.")
    features = valid[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0.0)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    full_embedding = scaler.fit_transform(features)
    embedding_frame = pd.DataFrame(
        full_embedding,
        columns=[f"embedding_{column}" for column in feature_columns],
        index=valid.index,
    )
    valid[embedding_frame.columns] = embedding_frame
    summary_rows = []
    for feature_index, feature in enumerate(feature_columns):
        raw_values = features[feature].to_numpy()
        values = full_embedding[:, feature_index].reshape(-1, 1)
        unique_count = len(np.unique(raw_values))
        cluster_count = min(n_clusters, unique_count)
        if cluster_count < 2:
            labels = np.zeros(len(values), dtype=int)
            centers = np.array([float(raw_values[0])])
        else:
            km = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
            raw_labels = km.fit_predict(values)
            normalized_centers = km.cluster_centers_.ravel()
            order = np.argsort(normalized_centers)
            mapping = {old: new for new, old in enumerate(order)}
            labels = np.array([mapping[label] for label in raw_labels], dtype=int)
            ordered_normalized_centers = np.sort(normalized_centers)
            centers = (
                scaler.data_min_[feature_index]
                + ordered_normalized_centers * scaler.data_range_[feature_index]
            )
        cluster_column = f"cluster_{feature}"
        valid[cluster_column] = labels
        for cluster_id in range(cluster_count):
            cluster_values = features.loc[valid.index[labels == cluster_id], feature]
            summary_rows.append({
                "parameter": feature,
                "cluster_id": cluster_id,
                "images": int(len(cluster_values)),
                "value_min": float(cluster_values.min()),
                "value_max": float(cluster_values.max()),
                "value_mean": float(cluster_values.mean()),
                "center": float(centers[cluster_id]),
                "normalization_min": float(scaler.data_min_[feature_index]),
                "normalization_max": float(scaler.data_max_[feature_index]),
            })
    return valid, pd.DataFrame(summary_rows), feature_columns


st.set_page_config(page_title="Leaf Coverage Detector", page_icon="🍃", layout="wide")
st.title("🍃 Leaf Coverage Detector")
st.caption("Segmentation + leaf coverage + masked-area image quality analysis")

with st.sidebar:
    st.header("Model")
    config_path = st.text_input("Inference config", "config.yaml")
    device_choice = st.selectbox("Device", ["auto","cuda","cpu"])
    device_name = ("cuda" if device_choice=="auto" and torch.cuda.is_available() else "cpu" if device_choice=="auto" else device_choice)
    threshold = st.slider("Leaf mask threshold", .05, .95, .60, .01)
    overlay_alpha = st.slider("Overlay opacity", .10, .90, .45, .05)

if not Path(config_path).exists():
    st.info(f"Config not found: `{config_path}`"); st.stop()

try:
    cfg = load_config(config_path)
    checkpoint_path = resolve_model_path(
        cfg["model"]["trained_checkpoint"], config_path
    )
    foundation_path = resolve_model_path(
        cfg["model"]["foundation_checkpoint"], config_path
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    if not foundation_path.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {foundation_path}")
    model,cfg,device = load_model(config_path,device_name)
except Exception as exc:
    st.error("Model loading failed."); st.exception(exc); st.stop()

live,single,batch,segregate,cluster,video = st.tabs(["📱 Live Camera","🖼️ Single Image","📁 Batch Analysis","🗂️ Segregate Images","🧩 Coverage Clustering","🎞️ Video Overlay"])

with live:
    st.subheader("Live leaf coverage")
    st.caption("Point the rear camera at the plant and move slowly. The overlay and percentage update from recent camera frames.")
    inference_interval = st.slider("Inference interval (seconds)", 0.25, 2.0, 0.75, 0.25, help="Increasing this reduces server load; the preview continues between model updates.")
    callback = make_live_camera_callback(model, cfg, device, threshold, overlay_alpha, inference_interval)
    webrtc_streamer(
        key="leaf-live-camera",
        video_frame_callback=callback,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        media_toggle_controls=False,
    )
    st.caption("Coverage is the fraction of visible frame pixels classified as leaf. It is an image estimate, not physical leaf area.")

with video:
    st.subheader("Create a leaf overlay from a local video")
    st.caption("Upload a video from this PC. Each frame is segmented and the overlay is saved as an MP4.")
    video_file = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video_upload")
    if video_file:
        st.video(video_file.getvalue())
        if st.button("Generate video overlay", type="primary", key="process_video"):
            progress = st.progress(0)
            status = st.empty()
            try:
                result_bytes, frame_count, fps = process_video(
                    video_file.getvalue(), model, cfg, device, threshold,
                    overlay_alpha, progress_callback=progress.progress,
                    suffix=Path(video_file.name).suffix or ".mp4",
                )
                status.success(f"Processed {frame_count} frames at {fps:.2f} FPS.")
                st.video(result_bytes)
                st.download_button(
                    "Download overlaid video",
                    data=result_bytes,
                    file_name=f"{Path(video_file.name).stem}_overlay.mp4",
                    mime="video/mp4",
                )
            except Exception as exc:
                status.error(f"Video processing failed: {exc}")
            finally:
                progress.empty()

with single:
    source = st.radio("Input source",["Use camera","Upload image","Local image path","Internet URL"],horizontal=True)
    image = None; image_name = "image"
    if source == "Use camera":
        camera_photo = st.camera_input("Take a photo of the plant", key="single_camera_photo")
        if camera_photo:
            image = Image.open(io.BytesIO(camera_photo.getvalue())).convert("RGB")
            image_name = "camera_capture.jpg"
    elif source=="Upload image":
        u=st.file_uploader("Choose image",type=[x[1:] for x in sorted(SUPPORTED_EXTENSIONS)])
        if u: image=Image.open(io.BytesIO(u.getvalue())).convert("RGB"); image_name=u.name
    elif source=="Local image path":
        p=st.text_input("Image path")
        if p and Path(p).is_file(): image=Image.open(p).convert("RGB"); image_name=Path(p).name
    else:
        url=st.text_input("Image URL")
        if url:
            try: image,image_name=load_url_image(url)
            except Exception as exc: st.error(str(exc))
    if image is not None:
        r=predict_image(model,image,cfg,device,threshold); m=r["metrics"]
        show_metrics(m)
        show_overlay = st.checkbox("Show leaf overlay", value=True, key="single_show_overlay")
        if show_overlay:
            overlay=make_overlay(image,r["mask"],overlay_alpha)
            a,b=st.columns(2); a.image(image,caption="Original",use_container_width=True); b.image(overlay,caption="Leaf overlay",use_container_width=True)
        else:
            st.image(image,caption="Captured image" if source == "Use camera" else "Input image",use_container_width=True)
        if source == "Use camera":
            show_veins = st.checkbox("Show estimated leaf veins (experimental)", value=False, key="single_show_veins")
            if show_veins:
                veins = detect_leaf_veins(image, r["mask"])
                vein_view = make_vein_overlay(image, veins)
                st.image(vein_view, caption="Estimated vein-like lines inside the leaf mask", use_container_width=True)
                st.caption("This filter-based estimate can mark leaf texture, shadows, and edges as veins. Image detail and lighting affect the result.")
        with st.expander("All quality metrics"):
            st.dataframe(pd.DataFrame([{"image":image_name,**m}]),use_container_width=True)
        if show_overlay:
            st.download_button("Download overlay",png_bytes(overlay),f"{Path(image_name).stem}_overlay.png","image/png")

with batch:
    mode=st.radio("Batch input",["Select folder","Enter folder path","Upload images"],horizontal=True)
    items=[]
    if mode=="Select folder":
        if st.button("📂 Pick folder from this PC"):
            selected=browse_folder()
            if selected: st.session_state["selected_folder"]=selected
        folder=st.session_state.get("selected_folder","")
    elif mode=="Enter folder path":
        folder=st.text_input("Folder path")
    else:
        folder=""
        uploads=st.file_uploader("Select multiple images",type=[x[1:] for x in sorted(SUPPORTED_EXTENSIONS)],accept_multiple_files=True)
        if uploads:
            if st.button("🚀 Analyze uploads",type="primary"):
                items=[(u.name,Image.open(io.BytesIO(u.getvalue())).convert("RGB")) for u in uploads]
                df,visuals=run_batch(model,items,cfg,device,threshold)
                st.session_state["batch_df"]=df
                st.session_state["batch_visuals"]=visuals
    if folder and Path(folder).is_dir():
        paths=find_images(folder); st.write(f"Found **{len(paths)}** images.")
        if st.button("🚀 Analyze folder",type="primary"):
            items=[(str(p),Image.open(p).convert("RGB")) for p in paths]
            df,visuals=run_batch(model,items,cfg,device,threshold)
            st.session_state["batch_df"]=df; st.session_state["batch_visuals"]=visuals
    if "batch_df" in st.session_state:
        df=st.session_state["batch_df"]; st.dataframe(df,use_container_width=True)
        st.download_button("Download metrics CSV",df.to_csv(index=False).encode(),"leaf_quality_metrics.csv","text/csv")
        visuals=st.session_state.get("batch_visuals",[])
        if visuals:
            n=st.slider("Preview images",1,min(20,len(visuals)),min(6,len(visuals)))
            for name,img,r in visuals[:n]:
                with st.expander(f"{name} — {r['metrics']['leaf_coverage_percent']:.2f}%"):
                    ov=make_overlay(img,r["mask"],overlay_alpha)
                    a,b=st.columns(2); a.image(img,use_container_width=True); b.image(ov,use_container_width=True)
                    show_metrics(r["metrics"])

with segregate:
    st.subheader("Sort a PC folder by leaf coverage")
    st.caption("Each image is segmented and placed in one of two folders based on its predicted leaf coverage percentage.")
    if st.button("📂 Choose source folder", key="pick_segregate_source"):
        selected = browse_folder()
        if selected:
            st.session_state["segregate_source"] = selected
            st.session_state["segregate_source_path"] = selected
    source_folder = st.text_input(
        "Source folder path", value=st.session_state.get("segregate_source", ""),
        key="segregate_source_path",
    )
    output_folder = ""
    if source_folder:
        source_path = Path(source_folder).expanduser()
        output_folder = str(source_path.parent / f"{source_path.name}_segregated")
        st.caption(f"Output folder will be created automatically: `{output_folder}`")

    coverage_threshold = st.number_input(
        "Leaf coverage threshold (%)", min_value=0.0, max_value=100.0,
        value=50.0, step=1.0,
        help="Images below this predicted coverage go in the lower folder; images equal to or above it go in the higher folder.",
    )
    operation = st.radio(
        "What should happen to the original images?", ["Copy", "Move"],
        horizontal=True, key="segregate_operation",
    )
    if source_folder and Path(source_folder).is_dir():
        st.write(f"Found **{len(find_images(source_folder))}** supported images.")
    if st.button(f"🚀 {operation} images into two coverage folders", type="primary", key="run_segregation"):
        if not source_folder or not output_folder:
            st.error("Choose both a source folder and a destination folder.")
        else:
            progress = st.progress(0)
            status = st.empty()
            try:
                segregation_df = segregate_images_by_coverage(
                    model, cfg, device, source_folder, output_folder,
                    threshold, coverage_threshold, operation,
                    progress_callback=progress.progress,
                )
                st.session_state["segregation_df"] = segregation_df
                succeeded = int((segregation_df["result"] == "success").sum())
                failed = len(segregation_df) - succeeded
                action = "Copied" if operation == "Copy" else "Moved"
                status.success(f"{action} {succeeded} of {len(segregation_df)} images into `{Path(output_folder)}`.")
                if failed:
                    st.warning(f"{failed} images could not be processed; see the results table.")
            except Exception as exc:
                status.error(f"Segregation failed: {exc}")
            finally:
                progress.empty()
    if "segregation_df" in st.session_state:
        segregation_df = st.session_state["segregation_df"]
        st.dataframe(segregation_df, use_container_width=True)
        st.download_button(
            "Download segregation report",
            segregation_df.to_csv(index=False).encode("utf-8"),
            "leaf_coverage_segregation.csv", "text/csv",
        )

with cluster:
    st.subheader("Cluster folder by image-processing parameters")

    st.info(
        "Select a folder to calculate six foreground-mask quality metrics: "
        "Tenengrad, FFT high-frequency ratio, edge density, brightness mean and "
        "variation, and saturation. The CSV and parameter clustering use only these metrics."
    )

    # ---------------------------------------------------------
    # Folder selection
    # ---------------------------------------------------------

    if st.button("📂 Pick folder from this PC", key="pick_cluster_folder"):
        selected = browse_folder()

        if selected:
            st.session_state["cluster_folder"] = selected

    cluster_folder = st.session_state.get("cluster_folder", "")

    # Manual path option
    manual_folder = st.text_input(
        "Or enter folder path manually",
        value=cluster_folder,
        key="cluster_folder_manual",
    )

    if manual_folder:
        cluster_folder = manual_folder
        st.session_state["cluster_folder"] = manual_folder

    # ---------------------------------------------------------
    # Display selected folder
    # ---------------------------------------------------------

    if cluster_folder:

        cluster_path = Path(cluster_folder)

        if not cluster_path.exists():
            st.error(
                f"Folder does not exist:\n\n{cluster_folder}"
            )
            st.stop()

        if not cluster_path.is_dir():
            st.error(
                f"Selected path is not a directory:\n\n{cluster_folder}"
            )
            st.stop()

        st.success(
            f"Selected folder:\n\n`{cluster_folder}`"
        )

        # -----------------------------------------------------
        # Find images
        # -----------------------------------------------------

        paths = find_images(cluster_folder)

        st.metric(
            "Images found",
            len(paths),
        )

        if not paths:
            st.warning(
                "No supported images found in this folder."
            )
            st.stop()

        # -----------------------------------------------------
        # Clustering configuration
        # -----------------------------------------------------

        st.markdown("### Clustering settings")

        k = st.slider(
            "Maximum clusters per parameter",
            min_value=2,
            max_value=min(20, len(paths)),
            value=min(5, len(paths)),
            step=1,
            key="coverage_cluster_count",
        )

        default_output = str(
            cluster_path / "parameter_clusters"
        )

        output = st.text_input(
            "Cluster output folder",
            value=default_output,
            key="coverage_cluster_output",
        )

        st.markdown("---")

        # -----------------------------------------------------
        # START BUTTON
        # -----------------------------------------------------

        start_processing = st.button(
            "🚀 Start Processing & Clustering",
            type="primary",
            use_container_width=True,
            key="start_coverage_clustering",
        )

        if start_processing:

            st.session_state.pop(
                "cluster_df",
                None,
            )

            st.session_state.pop(
                "cluster_summary",
                None,
            )

            st.session_state.pop(
                "cluster_source",
                None,
            )

            st.session_state.pop(
                "cluster_visuals",
                None,
            )

            items = []

            load_progress = st.progress(0)
            load_status = st.empty()

            for i, p in enumerate(paths, start=1):

                try:

                    image = Image.open(p).convert("RGB")

                    items.append(
                        (
                            str(p),
                            image,
                        )
                    )

                except Exception as exc:

                    st.warning(
                        f"Could not open `{p}`: {exc}"
                    )

                load_progress.progress(
                    i / len(paths)
                )

                load_status.write(
                    f"Loading images: {i}/{len(paths)}"
                )

            load_progress.empty()
            load_status.empty()

            if not items:
                st.error(
                    "No images could be loaded."
                )
                st.stop()

            # -------------------------------------------------
            # Run model
            # -------------------------------------------------

            st.markdown("### Step 1 — Leaf segmentation")

            df, visuals = run_batch(
                model=model,
                items=items,
                cfg=cfg,
                device=device,
                threshold=threshold,
            )

            if df.empty:
                st.error(
                    "No prediction results were generated."
                )
                st.stop()

            # -------------------------------------------------
            # Cluster
            # -------------------------------------------------

            st.markdown(
                "### Step 2 — Parameter embeddings and individual clusters"
            )

            try:

                clustered, summary, embedding_features = assign_parameter_clusters(df, k)

            except Exception as exc:

                st.error(
                    f"Clustering failed: {exc}"
                )
                st.stop()

            # -------------------------------------------------
            # Save in session state
            # -------------------------------------------------

            st.session_state["cluster_df"] = clustered

            st.session_state["cluster_source"] = cluster_folder

            st.session_state["cluster_visuals"] = visuals

            st.session_state["cluster_output"] = output
            st.session_state["cluster_embedding_features"] = embedding_features
            st.session_state["cluster_embedding_version"] = EMBEDDING_VERSION

            st.success(
                f"Processing completed for {len(clustered)} images."
            )

    # =========================================================
    # RESULTS
    # =========================================================

    if "cluster_df" in st.session_state:

        df = st.session_state["cluster_df"].copy()
        summary = st.session_state.get("cluster_summary")
        embedding_features = st.session_state.get("cluster_embedding_features")
        expected_features = [feature for feature in CLUSTER_FEATURES if feature in df.columns]
        assignments_exist = (
            embedding_features == expected_features
            and all(f"cluster_{feature}" in df.columns for feature in embedding_features)
        )
        if (
            summary is None
            or not assignments_exist
            or st.session_state.get("cluster_embedding_version") != EMBEDDING_VERSION
        ):
            # Streamlit can preserve a dataframe from an older app run while
            # its newer session-state keys are absent. Rebuild from raw metrics.
            cluster_count = min(
                int(st.session_state.get("coverage_cluster_count", 5)),
                len(df),
            )
            df, summary, embedding_features = assign_parameter_clusters(
                df,
                cluster_count,
            )
            st.session_state["cluster_df"] = df
            st.session_state["cluster_summary"] = summary
            st.session_state["cluster_embedding_features"] = embedding_features
            st.session_state["cluster_embedding_version"] = EMBEDDING_VERSION

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        st.markdown(
            "## 📊 Parameter Correlation"
        )

        st.caption(
            "Correlation is calculated across the original metric values for all "
            "successfully processed images. The CSV also contains per-feature "
            "min-max normalized values in the embedding_* columns (0–1)."
        )

        correlation = df[st.session_state["cluster_embedding_features"]].corr()
        st.dataframe(correlation.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

        feature_labels = {
            "masked_tenengrad": "Masked Tenengrad",
            "masked_fft_high_frequency_ratio": "Masked FFT high-frequency ratio",
            "masked_edge_density": "Masked edge density",
            "masked_brightness_mean": "Masked mean brightness",
            "masked_brightness_std": "Masked brightness variation",
            "masked_saturation_mean": "Masked mean saturation",
        }
        export_parameter = st.selectbox(
            "Parameter to use for cluster folders",
            options=st.session_state["cluster_embedding_features"],
            format_func=lambda feature: feature_labels.get(feature, feature),
            key="cluster_export_parameter",
        )
        cluster_col = f"cluster_{export_parameter}"
        parameter_summary = summary[summary["parameter"] == export_parameter]

        st.markdown(f"### Cluster ranges for {feature_labels.get(export_parameter, export_parameter)}")
        st.dataframe(parameter_summary, use_container_width=True, hide_index=True)

        # -----------------------------------------------------
        # Coverage distribution
        # -----------------------------------------------------

        st.markdown(
            f"## 📈 {feature_labels.get(export_parameter, export_parameter)} by image"
        )

        st.bar_chart(
            df.set_index("image")[export_parameter],
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Detailed metrics
        # -----------------------------------------------------

        st.markdown(
            "## 📋 Image Metrics"
        )

        display_df = df.sort_values(
            [
                cluster_col,
                export_parameter,
            ]
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Download CSV
        # -----------------------------------------------------

        st.download_button(
            "⬇️ Download six metrics and parameter clusters CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="parameter_clusters.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Copy images
        # -----------------------------------------------------

        st.markdown(
            "## 📦 Export Clustered Images"
        )

        output = st.session_state.get(
            "cluster_output",
            "",
        )

        if st.button(
            f"📦 Create {export_parameter} folders and copy images",
            use_container_width=True,
        ):

            source = Path(
                st.session_state[
                    "cluster_source"
                ]
            )

            out = Path(output)
            parameter_out = out / export_parameter

            copied = 0
            missing = 0

            progress = st.progress(0)

            rows = list(df.iterrows())

            for index, (_, row) in enumerate(
                rows,
                start=1,
            ):

                src = Path(
                    str(row["image"])
                )

                # If stored path is not valid,
                # search by filename.
                if not src.is_file():

                    matches = list(
                        source.rglob(
                            src.name
                        )
                    )

                    src = (
                        matches[0]
                        if matches
                        else None
                    )

                if src is None:

                    missing += 1
                    continue

                destination_dir = (
                    parameter_out
                    / f"cluster_{int(row[cluster_col])}"
                    / "images"
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                try:

                    shutil.copy2(
                        src,
                        destination_dir
                        / src.name,
                    )

                    copied += 1

                except Exception:

                    missing += 1

                progress.progress(
                    index / len(rows)
                )

            progress.empty()

            # Save a portable map from each cluster folder to its feature profile.
            out.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out / "cluster_summary.csv", index=False)
            parameter_out.mkdir(parents=True, exist_ok=True)
            for _, cluster_row in parameter_summary.iterrows():
                cluster_id = int(cluster_row["cluster_id"])
                cluster_dir = parameter_out / f"cluster_{cluster_id}"
                cluster_dir.mkdir(parents=True, exist_ok=True)
                cluster_info = {
                    "parameter": export_parameter,
                    "cluster_id": cluster_id,
                    "image_count": int(cluster_row["images"]),
                    "value_min": float(cluster_row["value_min"]),
                    "value_max": float(cluster_row["value_max"]),
                    "value_mean": float(cluster_row["value_mean"]),
                    "cluster_center": float(cluster_row["center"]),
                    "normalization_min": float(cluster_row["normalization_min"]),
                    "normalization_max": float(cluster_row["normalization_max"]),
                }
                with (cluster_dir / "cluster_info.json").open("w", encoding="utf-8") as info_file:
                    json.dump(cluster_info, info_file, indent=2)

            st.success(
                f"Copied {copied} images."
            )

            if missing:
                st.warning(
                    f"{missing} images could not be copied."
                )

        # -----------------------------------------------------
        # Copy predicted masks
        # -----------------------------------------------------

        if st.button(
            f"🌓 Copy predicted masks for {export_parameter}",
            use_container_width=True,
            key="copy_cluster_predicted_masks",
        ):

            out = Path(output)
            parameter_out = out / export_parameter
            visuals = st.session_state.get(
                "cluster_visuals",
                [],
            )

            # Match each clustered image to its in-memory segmentation result.
            visual_map = {
                Path(str(name)).name: result
                for name, _image, result in visuals
            }

            copied = 0
            missing = 0
            progress = st.progress(0)
            rows = list(df.iterrows())

            for index, (_, row) in enumerate(rows, start=1):
                image_name = Path(str(row["image"])).name
                result = visual_map.get(image_name)

                if result is None:
                    missing += 1
                    progress.progress(index / len(rows))
                    continue

                destination_dir = (
                    parameter_out
                    / f"cluster_{int(row[cluster_col])}"
                    / "masks"
                )
                destination_dir.mkdir(parents=True, exist_ok=True)

                try:
                    mask_png = np.asarray(result["mask"], dtype=np.uint8) * 255
                    destination = destination_dir / f"{Path(image_name).stem}.png"
                    Image.fromarray(mask_png, mode="L").save(
                        destination,
                        format="PNG",
                    )
                    copied += 1
                except Exception as exc:
                    st.warning(
                        f"Could not save predicted mask for {image_name}: {exc}"
                    )
                    missing += 1

                progress.progress(index / len(rows))

            progress.empty()
            st.success(f"Copied {copied} predicted masks.")

            if missing:
                st.warning(f"{missing} predicted masks could not be copied.")

        # -----------------------------------------------------
        # Copy overlay images
        # -----------------------------------------------------

        if st.button(
            f"🎨 Copy overlay images for {export_parameter}",
            use_container_width=True,
            key="copy_cluster_overlay_images",
        ):

            source = Path(
                st.session_state[
                    "cluster_source"
                ]
            )

            out = Path(output)
            parameter_out = out / export_parameter

            visuals = st.session_state.get(
                "cluster_visuals",
                [],
            )

            # Map the filename to the already-generated
            # in-memory image + segmentation result.
            visual_map = {}

            for name, image, result in visuals:
                visual_map[
                    Path(str(name)).name
                ] = (
                    image,
                    result,
                )

            copied = 0
            missing = 0

            progress = st.progress(0)

            rows = list(df.iterrows())

            for index, (_, row) in enumerate(
                rows,
                start=1,
            ):

                image_name = Path(
                    str(row["image"])
                ).name

                item = visual_map.get(
                    image_name
                )

                if item is None:
                    missing += 1

                    progress.progress(
                        index / len(rows)
                    )

                    continue

                image, result = item

                destination_dir = (
                    parameter_out
                    / f"cluster_{int(row[cluster_col])}"
                    / "overlay"
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                try:

                    overlay = make_overlay(
                        image,
                        result["mask"],
                        alpha=overlay_alpha,
                    )

                    destination = (
                        destination_dir
                        / (
                            f"{Path(image_name).stem}"
                            "_overlay.png"
                        )
                    )

                    Image.fromarray(
                        overlay
                    ).save(
                        destination,
                        format="PNG",
                    )

                    copied += 1

                except Exception as exc:

                    st.warning(
                        f"Could not save overlay "
                        f"for {image_name}: {exc}"
                    )

                    missing += 1

                progress.progress(
                    index / len(rows)
                )

            progress.empty()

            st.success(
                f"Copied {copied} overlay images."
            )

            if missing:
                st.warning(
                    f"{missing} overlay images could not be copied."
                )
