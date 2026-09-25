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
def load_model(config_path, checkpoint_path, foundation_path, device_name):
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


def find_images(folder):
    root = Path(folder).expanduser()
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


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


st.set_page_config(page_title="Leaf Coverage Detector", page_icon="🍃", layout="wide")
st.title("🍃 Leaf Coverage Detector")
st.caption("Segmentation + leaf coverage + masked-area image quality analysis")

with st.sidebar:
    st.header("Model")
    config_path = st.text_input("Inference config", "config.yaml")
    checkpoint_path = st.text_input("Trained model checkpoint", "checkpoints/best.pt")
    foundation_path = st.text_input("Foundation checkpoint", "checkpoints/crop_fm1_coatNet2rw224.pth")
    device_choice = st.selectbox("Device", ["auto","cuda","cpu"])
    device_name = ("cuda" if device_choice=="auto" and torch.cuda.is_available() else "cpu" if device_choice=="auto" else device_choice)
    threshold = st.slider("Leaf mask threshold", .05, .95, .60, .01)
    overlay_alpha = st.slider("Overlay opacity", .10, .90, .45, .05)

if not Path(config_path).exists():
    st.info(f"Config not found: `{config_path}`"); st.stop()
if not Path(checkpoint_path).exists():
    st.info(f"Trained checkpoint not found: `{checkpoint_path}`"); st.stop()
if not Path(foundation_path).exists():
    st.info(f"Foundation checkpoint not found: `{foundation_path}`"); st.stop()

try:
    model,cfg,device = load_model(config_path,checkpoint_path,foundation_path,device_name)
except Exception as exc:
    st.error("Model loading failed."); st.exception(exc); st.stop()

single,batch,cluster = st.tabs(["🖼️ Single Image","📁 Batch Analysis","🧩 Coverage Clustering"])

with single:
    source = st.radio("Input source",["Upload image","Local image path","Internet URL"],horizontal=True)
    image = None; image_name = "image"
    if source=="Upload image":
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
        overlay=make_overlay(image,r["mask"],overlay_alpha)
        a,b=st.columns(2); a.image(image,caption="Original",use_container_width=True); b.image(overlay,caption="Leaf overlay",use_container_width=True)
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

        df = st.session_state["cluster_df"]

        summary = st.session_state[
            "cluster_summary"
        ]

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
