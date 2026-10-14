from __future__ import annotations



import io

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


import time


from PIL import Image

from sklearn.cluster import KMeans

from torchvision.transforms import functional as TF




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





def make_overlay(image, mask, alpha=.45):

          rgb = np.asarray(image.convert("RGB")).astype(np.float32)

          overlay = rgb.copy()

          green = np.zeros_like(rgb)

          green[...,1] = 255

          overlay[mask] = (1-alpha)*rgb[mask] + alpha*green[mask]

          return np.clip(overlay,0,255).astype(np.uint8)





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





def run_batch(model, items, cfg, device, threshold):

          rows, visuals = [], []

          progress = st.progress(0)

          for i,(name,image) in enumerate(items,1):

                    try:

                              r = predict_image(model,image,cfg,device,threshold)

                              rows.append({"image":name, **r["metrics"]})

                              visuals.append((name,image,r))

                    except Exception as exc:

                              rows.append({"image":name,"error":str(exc)})

                    progress.progress(i/len(items))

          progress.empty()

          return pd.DataFrame(rows), visuals





def cluster_by_coverage(df, n_clusters):

          valid = df.dropna(subset=["leaf_coverage_percent"]).copy()

          if len(valid) < n_clusters:

                    raise ValueError(f"Need at least {n_clusters} valid images; found {len(valid)}.")

          km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)

          valid["cluster_raw"] = km.fit_predict(valid[["leaf_coverage_percent"]])

          order = np.argsort(km.cluster_centers_.ravel())

          mapping = {old:new for new,old in enumerate(order)}

          valid["coverage_cluster"] = valid["cluster_raw"].map(mapping)

          summary = valid.groupby("coverage_cluster")["leaf_coverage_percent"].agg(

                    images="count", coverage_min="min", coverage_max="max", coverage_mean="mean"

          ).reset_index()

          return valid, summary





def cluster_by_coverage(df, n_clusters):

          if not isinstance(n_clusters, int) or n_clusters < 1 or n_clusters > 100:

                    raise ValueError("n_clusters must be an integer between 1 and 100.")



          valid = df.dropna(subset=["leaf_coverage_percent"]).copy()



          coverage = valid["leaf_coverage_percent"]



          if ((coverage < 0) | (coverage > 100)).any():

                    raise ValueError("leaf_coverage_percent must be between 0 and 100.")



          # Create fixed-width bins over the coverage range [0, 100].

          bin_width = 100.0 / n_clusters

          edges = np.linspace(0, 100, n_clusters + 1)



          # Cluster IDs are zero-based.

          # For 5 clusters: [0,20), [20,40), [40,60), [60,80), [80,100].

          valid["coverage_cluster"] = pd.cut(

                    coverage,

                    bins=edges,

                    labels=False,

                    right=False,

                    include_lowest=True,

          )



          # Include coverage=100 in the final cluster.

          valid.loc[coverage == 100, "coverage_cluster"] = n_clusters - 1

          valid["coverage_cluster"] = valid["coverage_cluster"].astype(int)



          # Include empty clusters in the summary.

          summary = (

                    valid.groupby("coverage_cluster")["leaf_coverage_percent"]

                    .agg(

                              images="count",

                              coverage_min="min",

                              coverage_max="max",

                              coverage_mean="mean",

                    )

                    .reindex(range(n_clusters))

                    .rename_axis("coverage_cluster")

                    .reset_index()

          )



          summary["images"] = summary["images"].fillna(0).astype(int)



          # Add the nominal coverage interval for each cluster.

          summary["bin_min"] = edges[:-1]

          summary["bin_max"] = edges[1:]



          return valid, summary



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



live, single, batch, segregate, cluster, video = st.tabs([
    "📱 Live Camera", "🖼️ Single Image", "📁 Batch Analysis",
    "🗂️ Segregate Images", "🧩 Coverage Clustering", "🎞️ Video Overlay",
])

with live:
    st.subheader("📷 Capture and classify one photo at a time")
    st.caption(
        "The live video stream is disabled. Take a still photo, and the model will "
        "estimate leaf coverage and classify it as Fair or Bad."
    )

    coverage_threshold = st.slider(
        "Minimum leaf coverage for a Fair photo (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="photo_coverage_threshold",
        help="Photos with coverage equal to or above this value are saved as Fair; lower coverage photos are saved as Bad.",
    )
    capture_dir = st.text_input(
        "Save photos on this PC", "captures", key="classified_capture_dir",
        help="Relative paths are resolved from the directory where Streamlit was started.",
    )

    if "photo_capture_index" not in st.session_state:
        st.session_state["photo_capture_index"] = 0
    if "classified_photo_result" not in st.session_state:
        st.session_state["classified_photo_result"] = None

    capture_index = st.session_state["photo_capture_index"]

    # Clear the prior camera widget value BEFORE instantiating the widget.
    # Streamlit forbids assigning to a widget-owned key after it is created;
    # deleting it before creation is the supported session-state reset pattern.
    if st.session_state.pop("reset_classified_camera", False):
        st.session_state.pop("classified_photo_camera", None)

    camera_photo = st.camera_input(
        "Take a photo of the plant or leaf",
        key="classified_photo_camera",
    )

    if camera_photo is not None:
        import hashlib

        photo_bytes = camera_photo.getvalue()
        photo_digest = hashlib.sha256(photo_bytes).hexdigest()
        photo_key = photo_digest
        previous_result = st.session_state.get("classified_photo_result")

        # Process/save each captured photo only once, despite Streamlit reruns.
        if not previous_result or previous_result.get("photo_key") != photo_key:
            try:
                image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
                prediction = predict_image(model, image, cfg, device, threshold)
                metrics = prediction["metrics"]
                coverage = float(metrics["leaf_coverage_percent"])
                classification = "fair" if coverage >= float(coverage_threshold) else "bad"

                output_root = Path(capture_dir).expanduser()
                if not output_root.is_absolute():
                    output_root = Path.cwd() / output_root
                class_dir = output_root / classification
                overlay_dir = class_dir / "overlays"
                class_dir.mkdir(parents=True, exist_ok=True)
                overlay_dir.mkdir(parents=True, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = class_dir / f"leaf_{timestamp}.jpg"
                suffix = 1
                while image_path.exists():
                    image_path = class_dir / f"leaf_{timestamp}_{suffix:02d}.jpg"
                    suffix += 1
                image.save(image_path, format="JPEG", quality=95, subsampling=0)

                overlay_rgb = make_overlay(image, prediction["mask"], overlay_alpha)
                # Add a clearly readable coverage label directly onto the saved overlay.
                label = f"Leaf coverage: {coverage:.2f}%"
                h, w = overlay_rgb.shape[:2]
                font_scale = max(0.65, min(w, h) / 900.0)
                thickness = max(2, int(round(font_scale * 2)))
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                pad = max(10, int(round(font_scale * 12)))
                cv2.rectangle(
                    overlay_rgb, (0, 0),
                    (min(w - 1, text_w + pad * 2), min(h - 1, text_h + baseline + pad * 2)),
                    (0, 0, 0), thickness=-1,
                )
                cv2.putText(
                    overlay_rgb, label, (pad, min(h - 1, text_h + pad)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA,
                )
                overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
                Image.fromarray(overlay_rgb).save(overlay_path, format="PNG")

                metrics_path = output_root / "capture_classification_metrics.csv"
                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                    "leaf_coverage_percent": round(coverage, 4),
                    "coverage_threshold_percent": float(coverage_threshold),
                    "classification": classification,
                    "image_width": image.width,
                    "image_height": image.height,
                    "mask_threshold": float(threshold),
                }
                pd.DataFrame([row]).to_csv(
                    metrics_path, mode="a", header=not metrics_path.exists(), index=False
                )

                st.session_state["classified_photo_result"] = {
                    "photo_key": photo_key,
                    "capture_index": capture_index,
                    "classification": classification,
                    "coverage": coverage,
                    "coverage_threshold": float(coverage_threshold),
                    "image_path": str(image_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                    "metrics_path": str(metrics_path.resolve()),
                    "width": image.width,
                    "height": image.height,
                    "overlay_rgb": overlay_rgb,
                }
            except Exception as exc:
                st.error(f"Could not process or save this photo: {exc}")

    result = st.session_state.get("classified_photo_result")
    if result and result.get("capture_index") == capture_index:
        coverage = result["coverage"]
        if result["classification"] == "fair":
            st.success(
                f"✅ Selected as FAIR — leaf coverage {coverage:.2f}% "
                f"(minimum {result['coverage_threshold']:.0f}%)."
            )
        else:
            st.warning(
                f"⚠️ Selected as BAD — leaf coverage {coverage:.2f}% "
                f"(minimum {result['coverage_threshold']:.0f}%)."
            )
        st.caption(f"Original photo saved to: `{result['image_path']}`")
        st.caption(f"Overlay saved to: `{result['overlay_path']}`")
        st.caption(
            f"Image: {result['width']} × {result['height']} · "
            f"Metrics CSV: `{result['metrics_path']}`"
        )

        show_overlay = st.checkbox(
            "Show segmentation overlay (optional)",
            value=False,
            key=f"show_classified_overlay_{capture_index}",
        )
        if show_overlay:
            st.image(
                result["overlay_rgb"],
                caption=f"Saved overlay · leaf coverage {coverage:.2f}%",
                use_container_width=True,
            )

        st.info("Photo classified and saved. Please collect the next photo quickly to continue the dataset collection.")
        if st.button("📸 Collect next photo", type="primary", key=f"next_classified_photo_{capture_index}"):
            # On the next rerun, clear the camera widget's previous captured
            # value before recreating the widget, so the user gets a fresh capture.
            # Keep the previous result internally for duplicate protection.
            st.session_state["photo_capture_index"] = capture_index + 1
            st.session_state["reset_classified_camera"] = True
            st.rerun()
    else:
        st.info(
            "Take a new photo to continue. The previous capture has been cleared; "
            "the original image and labeled overlay are saved automatically, while the overlay "
            "remains hidden unless you request it."
        )


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

                    with st.expander("All quality metrics"):

                              st.dataframe(pd.DataFrame([{"image":image_name,**m}]),use_container_width=True)

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

          st.subheader("Cluster folder by leaf coverage")



          st.info(

                    "Select a folder containing images. The app will first run leaf "

                    "segmentation, calculate leaf coverage, then cluster the images "

                    "based on leaf coverage."

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

                              "Number of coverage clusters",

                              min_value=2,

                              max_value=min(20, len(paths)),

                              value=min(5, len(paths)),

                              step=1,

                              key="coverage_cluster_count",

                    )



                    default_output = str(

                              cluster_path / "coverage_clusters"

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

                                        "### Step 2 — Coverage clustering"

                              )



                              try:



                                        clustered, summary = cluster_by_coverage(

                                                  df,

                                                  k,

                                        )



                              except Exception as exc:



                                        st.error(

                                                  f"Clustering failed: {exc}"

                                        )

                                        st.stop()



                              # -------------------------------------------------

                              # Save in session state

                              # -------------------------------------------------



                              st.session_state["cluster_df"] = clustered



                              st.session_state["cluster_summary"] = summary



                              st.session_state["cluster_source"] = cluster_folder



                              st.session_state["cluster_visuals"] = visuals



                              st.session_state["cluster_output"] = output



                              st.success(

                                        f"Processing completed for {len(clustered)} images."

                              )



          # =========================================================

          # RESULTS

          # =========================================================



          if "cluster_df" in st.session_state:



                    df = st.session_state["cluster_df"].copy()

                    summary = st.session_state.get("cluster_summary")

                    if (

                              summary is None

                              and "coverage_cluster" in df.columns

                              and "leaf_coverage_percent" in df.columns

                    ):

                              summary = df.groupby("coverage_cluster")[

                                        "leaf_coverage_percent"

                              ].agg(

                                        images="count",

                                        coverage_min="min",

                                        coverage_max="max",

                                        coverage_mean="mean",

                              ).reset_index()



                    # -----------------------------------------------------

                    # Summary

                    # -----------------------------------------------------



                    st.markdown(

                              "## 📊 Cluster Summary"

                    )



                    st.dataframe(

                              summary,

                              use_container_width=True,

                              hide_index=True,

                    )



                    # -----------------------------------------------------

                    # Coverage distribution

                    # -----------------------------------------------------



                    st.markdown(

                              "## 📈 Coverage Distribution"

                    )



                    st.bar_chart(

                              df.set_index("image")[

                                        "leaf_coverage_percent"

                              ],

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

                                        "coverage_cluster",

                                        "leaf_coverage_percent",

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

                              "⬇️ Download coverage clusters CSV",

                              data=df.to_csv(

                                        index=False

                              ).encode("utf-8"),

                              file_name="coverage_clusters.csv",

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

                              "📦 Copy images into cluster folders",

                              use_container_width=True,

                    ):



                              source = Path(

                                        st.session_state[

                                                  "cluster_source"

                                        ]

                              )



                              out = Path(output)



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

                                                  out

                                                  / f"cluster_{int(row['coverage_cluster'])}"

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

                              "🌓 Copy predicted masks into cluster folders",

                              use_container_width=True,

                              key="copy_cluster_predicted_masks",

                    ):



                              out = Path(output)

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

                                                  out

                                                  / f"cluster_{int(row['coverage_cluster'])}"

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

                              "🎨 Copy overlay images into cluster folders",

                              use_container_width=True,

                              key="copy_cluster_overlay_images",

                    ):



                              source = Path(

                                        st.session_state[

                                                  "cluster_source"

                                        ]

                              )



                              out = Path(output)



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

                                                  out

                                                  / f"cluster_{int(row['coverage_cluster'])}"

                                                  / "overlays"

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
