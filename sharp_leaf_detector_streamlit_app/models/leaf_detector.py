import torch.nn as nn
import torch.nn.functional as F

from .backbone import CoAtNet2Backbone
from .adapters import FeatureAdapters
from .decoder import MultiScaleDecoder


class CoAtNetLeafDetector(nn.Module):
    def __init__(
        self,
        foundation_checkpoint,
        model_name="coatnet_2_rw_224",
        out_indices=(0, 1, 2, 3),
        backbone_channels=(128, 256, 512, 1024),
        adapter_channels=(64, 96, 192, 384),
        decoder_channels=128,
    ):
        super().__init__()

        self.backbone = CoAtNet2Backbone(
            checkpoint_path=foundation_checkpoint,
            model_name=model_name,
            out_indices=out_indices,
            expected_channels=backbone_channels,
            pretrained=False,
        )

        self.adapters = FeatureAdapters(
            self.backbone.channels,
            adapter_channels,
        )

        self.decoder = MultiScaleDecoder(
            adapter_channels,
            decoder_channels,
        )

    def forward(self, x, output_size=None, return_features=False):
        input_size = x.shape[-2:]

        raw_features = self.backbone(x)
        adapted = self.adapters(raw_features)
        logits = self.decoder(adapted)

        output_size = output_size or input_size
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        if return_features:
            return logits, raw_features, adapted

        return logits

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()

    def trainable_parameter_groups(self, head_lr, backbone_lr):
        head_params = list(self.adapters.parameters()) + list(
            self.decoder.parameters()
        )

        backbone_params = [
            p for p in self.backbone.parameters()
            if p.requires_grad
        ]

        return [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
