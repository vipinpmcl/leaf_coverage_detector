# Leaf Coverage Detector v2

Adds:
- local video upload with frame-by-frame leaf overlay and MP4 export
- local PC folder picker (Tkinter when Streamlit runs locally)
- folder batch inference
- min-max normalized embeddings of six foreground-mask metrics, with an independent
  K-Means cluster assignment for each metric
- foreground-mask sharpness and image-quality metrics
- CSV export

Mask-derived metrics include leaf coverage/area, bounding-box dimensions/fill, and
mean leaf confidence. Sharpness, edge density, FFT high-frequency ratio,
brightness/contrast, clipping, and saturation are calculated over the predicted
foreground mask.

`coverage_weighted_edge_saturation_score` is calculated as `(leaf coverage percent / 100)`
times masked edge density times masked saturation mean. All factors are on a 0–1
scale. The score is zero when the foreground mask is empty and is included in the
parameter CSV, correlation matrix, and per-parameter clustering.

No single overall image-quality score is used; the coverage-weighted edge-saturation
feature is a separate parameter.

Folder processing uses only masked Tenengrad, FFT high-frequency ratio, edge density,
brightness mean, brightness standard deviation, and saturation mean for clustering.
It min-max normalizes these metrics to 0–1 and assigns each image to independent
one-dimensional K-Means clusters for every metric. The CSV includes these original
measurements, normalized `embedding_*` dimensions, and `cluster_<metric>` columns.
A correlation matrix is shown in the app. Choose a metric to group images into folders; exports are
written under `<output>/<metric>/cluster_N/`, with `cluster_info.json`,
`images/`, `overlay/`, and `masks/`. The output root receives
`cluster_summary.csv` with value ranges for every metric's clusters.

## Live camera on a phone

The **Live Camera** tab streams the phone's rear camera to the app over WebRTC,
runs leaf segmentation on the server, and displays a smoothed coverage estimate
on the preview. Connect the phone and computer to the same Wi-Fi network.

Start Streamlit so it accepts connections from other devices:

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Open `http://<computer-LAN-IP>:8501` on the phone to check the app. Mobile
browsers generally require HTTPS before they allow camera access to a LAN
address. For local testing, install `streamlit-remote` and bind it to the
network interface while enabling local HTTPS:

```powershell
python -m pip install streamlit-remote
st-remote app.py --host 0.0.0.0 --https self-signed --no-remote
```

Open `https://<computer-LAN-IP>:8501` on the phone. A self-signed development
certificate may trigger a browser warning; continue only for this trusted home
network test, or use a locally trusted certificate with `--https mkcert` after
installing/trusting its root certificate on both devices. Allow camera access
in the browser. If Windows Firewall prompts for Streamlit, allow it on the
private network.

Coverage is the percentage of pixels in the current camera view classified as
leaf. It can change with framing, distance, lighting, and movement; it is not a
measurement of the plant's physical leaf area.
