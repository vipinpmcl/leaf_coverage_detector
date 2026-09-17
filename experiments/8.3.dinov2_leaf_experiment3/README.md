# DINOv2 Leaf Segmentation - Experiment 3

Experiment 3 fine-tunes pretrained DINOv2 for leaf segmentation.

Pipeline:
RGB image -> DINOv2 ViT-S/14 -> patch tokens -> segmentation decoder -> 16x16 logits -> bilinear upsampling -> 224x224 mask.

Experiment 2 used frozen DINOv2 + an MLP patch classifier. Experiment 3 adapts DINOv2 to the actual leaf-mask task.

Dataset:
data/images/image001.jpg
data/masks/image001.png

Masks: 255=leaf, 0=background.

Install:
pip install -r requirements.txt

Train:
python scripts/train.py --config configs/config.yaml

Predict:
python scripts/predict.py --image data/images/image001.jpg --checkpoint outputs/best.pt --output outputs/image001

Batch:
python scripts/batch_predict.py --input data/images --checkpoint outputs/best.pt --output outputs/batch

Validation metrics:
IoU, Dice, precision, recall, accuracy.

Prediction additionally reports:
- total leaf coverage
- number of connected components
- largest component coverage
- largest component bounding-box size

This is a research baseline. Because ViT-S/14 has a coarse patch grid, the decoder output is still limited by the spatial information available from patch tokens. A stronger high-resolution decoder is a natural next experiment.
