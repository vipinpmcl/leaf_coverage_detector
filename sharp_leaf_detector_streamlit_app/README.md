# Leaf Coverage Detector v2

Adds:
- local video upload with frame-by-frame leaf overlay and MP4 export
- local PC folder picker (Tkinter when Streamlit runs locally)
- folder batch inference
- K-Means clustering using standardized embeddings of leaf coverage, mask geometry,
  leaf confidence, masked sharpness/edge metrics, brightness, clipping, and saturation
- raw masked-leaf sharpness/quality metrics
- CSV export

Masked quality metrics include Laplacian variance, Tenengrad, Brenner,
FFT high-frequency ratio, edge density, brightness/contrast, clipping,
saturation, leaf confidence, and resolution/scale metrics.

No combined quality score is used.

Folder clustering standardizes all available metrics into a full embedding for
each image. The selected features choose which embedding dimensions K-Means uses.
The exported cluster CSV includes original measurements and every standardized
`embedding_*` dimension, so images can be compared across multiple processing
signals rather than coverage alone.
When clustered images are copied to the output folder, each `cluster_N` folder
also receives a `cluster_info.json` containing its image count and selected
feature averages; the output root receives a `cluster_summary.csv`. Each cluster
stores source images in `images/`, overlays in `overlay/`, and predicted masks in
`masks/`.

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
