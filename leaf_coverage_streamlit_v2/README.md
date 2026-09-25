# Leaf Coverage Detector v2

Adds:
- local PC folder picker (Tkinter when Streamlit runs locally)
- folder batch inference
- clustering by leaf coverage using K-Means
- raw masked-leaf sharpness/quality metrics
- CSV export

Masked quality metrics include Laplacian variance, Tenengrad, Brenner,
FFT high-frequency ratio, edge density, brightness/contrast, clipping,
saturation, leaf confidence, and resolution/scale metrics.

No combined quality score is used.
