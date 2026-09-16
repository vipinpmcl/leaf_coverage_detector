import torch.nn as nn
import torch.nn.functional as F
from .backbone import CoAtNet2Backbone
from .adapters import FeatureAdapters
from .decoder import MultiScaleDecoder

class CoAtNetLeafDetector(nn.Module):
    def __init__(
        self,
        pretrained=True,
        out_indices=(0, 1, 2, 3),
        decoder_channels=128,
        adapter_channels=(64, 96, 192, 384),
    ):
        super().__init__()
        self.backbone = CoAtNet2Backbone(pretrained, out_indices)
        self.adapters = FeatureAdapters(self.backbone.channels, adapter_channels)
        self.decoder = MultiScaleDecoder(adapter_channels, decoder_channels)

    def forward(self, x, output_size=None, return_features=False):
        input_size = x.shape[-2:]
        raw_features = self.backbone(x)
        adapted = self.adapters(raw_features)
        logits = self.decoder(adapted)
        output_size = output_size or input_size
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        if return_features:
            return logits, raw_features, adapted
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
