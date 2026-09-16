# CoAtNet-2 Leaf Detector
PyTorch leaf semantic-segmentation project using `coatnet_2_rw_224` as a hierarchical encoder.

Architecture:
Input -> CoAtNet-2 -> hierarchical features -> 1x1 adapters -> multi-scale decoder -> 1-channel leaf mask.

The segmentation head is isolated so it can be replaced with another leaf decoder.

The backbone requests `out_indices=(0,1,2,3)` and reads channel counts dynamically from timm.
