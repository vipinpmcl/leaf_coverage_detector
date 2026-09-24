# Leaf Coverage Detector — Streamlit

Streamlit UI for the trained CoAtNet2 leaf segmentation model.

## Features

- Single image upload
- Local image path
- Internet image URL
- Recursive local directory inference
- Multiple uploaded images
- Original / leaf overlay comparison
- Pixel probability map
- Binary leaf mask
- Leaf coverage percentage
- Leaf probability
- Resolution and bounding-box metrics
- Batch CSV export
- Coverage distribution

## Project layout

```text
leaf_coverage_streamlit/
├── app.py
├── config.yaml
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── adapters.py
│   ├── decoder.py
│   └── leaf_detector.py
└── utils/
    └── __init__.py
```

Put your checkpoints outside or inside this directory, for example:

```text
checkpoints/crop_fm1_coatNet2rw224.pth
runs/coatnet2_leaf/best.pth
```

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sidebar lets you change the config, trained checkpoint and foundation checkpoint.

## Important

The app uses the same 224x224 input resizing and ImageNet normalization defined in
the supplied training configuration. The model output is resized back to the
original image dimensions before calculating leaf coverage.
