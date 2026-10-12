# Leaf Coverage Detector v2

Adds:
- local video upload with frame-by-frame leaf overlay and MP4 export
- local PC folder picker (Tkinter when Streamlit runs locally)
- folder batch inference
- clustering by leaf coverage using K-Means
- raw masked-leaf sharpness/quality metrics
- CSV export

Masked quality metrics include Laplacian variance, Tenengrad, Brenner,
FFT high-frequency ratio, edge density, brightness/contrast, clipping,
saturation, leaf confidence, and resolution/scale metrics.

No combined quality score is used.

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
