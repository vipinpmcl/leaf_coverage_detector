# CoAtNet-2 Foundation Leaf Segmentation

Separate project for **Case 1**:

- Foundation `.pth` contains a CoAtNet-2 backbone plus DINO/iBOT/other heads.
- Only the **student CoAtNet backbone** is loaded.
- `dino_head`, `ibot_head`, `slot_projector`, and `mask_embed` are ignored.
- The backbone is frozen initially.
- New lightweight adapters + multi-scale segmentation decoder are trained.
- Input normalization follows the supplied foundation metadata: mean/std = `(0.5, 0.5, 0.5)`.
- Backbone feature channels expected from the checkpoint architecture are `(128, 256, 512, 1024)`.
- Designed for an NVIDIA T400 4 GB: default batch size is 2 and input size is 224.

## Important checkpoint note

The conversation currently contains the checkpoint architecture dump, but not the binary `.pth` file itself. Put your actual foundation checkpoint here:

```text
checkpoints/coatnet2_foundation.pth
```

The loader expects the structure shown by the architecture dump:

```text
student.backbone.backbone.*
student.dino_head.*
student.ibot_head.*
student.slot_projector.*
teacher.backbone.backbone.*
```

It loads **student.backbone.backbone.\*** only.

## Dataset layout

```text
data/
├── images/
│   ├── image001.jpg
│   └── image002.jpg
└── masks/
    ├── image001.png
    └── image002.png
```

Mask pixels should be:
- 0 = background
- >0 = leaf

Missing masks are allowed for prediction, but training requires a matching mask.

## Install

```bash
conda create -n coatnet_leaf python=3.11 -y
conda activate coatnet_leaf

pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA setup, then install the remaining requirements if needed.

## 1. Verify the foundation checkpoint

```bash
python tools/check_foundation.py \
  --checkpoint checkpoints/coatnet2_foundation.pth
```

This reports:
- checkpoint structure
- number of student backbone tensors
- number of compatible tensors
- shape mismatches
- missing model tensors

The script does not silently accept a checkpoint that loads zero backbone parameters.

## 2. Train

```bash
python train.py --config configs/default.yaml
```

The first stage freezes the foundation backbone and trains only:

```text
FeatureAdapters
MultiScaleDecoder
```

The best checkpoint is saved to:

```text
runs/coatnet2_leaf/best.pt
```

## 3. Validate

```bash
python validate.py \
  --config configs/default.yaml \
  --checkpoint runs/coatnet2_leaf/best.pt
```

## 4. Predict one image

```bash
python predict.py \
  --config configs/default.yaml \
  --checkpoint runs/coatnet2_leaf/best.pt \
  --image data/images/image001.jpg
```

Output:

```text
runs/predictions/image001/
├── original.jpg
├── mask.png
├── probability.png
├── overlay.jpg
└── comparison.jpg
```

## Architecture

```text
                         foundation .pth
                               |
                               v
                  student.backbone.backbone
                               |
                    frozen CoAtNet-2
                               |
              +----------------+----------------+
              |                |                |
           f1 128           f2 256           f3 512        f4 1024
              |                |                |             |
          1x1 adapter       1x1 adapter      1x1 adapter   1x1 adapter
              +----------------+----------------+-------------+
                               |
                       FPN-style decoder
                               |
                         1-channel logits
                               |
                         sigmoid probability
                               |
                         leaf segmentation
```

## Why channels are 128/256/512/1024

The supplied checkpoint architecture explicitly shows:
- stem output: 128
- stage 1: 256
- stage 2: 512
- stage 3: 1024

For example, the architecture dump shows the stage-2 attention projection at 512 channels and the final backbone norm at 1024 channels.

## Case 1 training policy

This project intentionally does **not** use the DINO head for segmentation. The DINO head is a pretraining objective, not the leaf segmentation head.

The first experiment is:

1. Load foundation backbone.
2. Freeze it.
3. Train segmentation adapters/decoder.
4. Evaluate IoU/Dice/precision/recall.
5. Only after this baseline is established should backbone fine-tuning be considered.

