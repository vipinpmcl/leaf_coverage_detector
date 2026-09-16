import torch.nn as nn
import timm

class CoAtNet2Backbone(nn.Module):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.encoder = timm.create_model(
            "coatnet_2_rw_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )
        self.feature_info = self.encoder.feature_info
        self.channels = list(self.feature_info.channels())
        self.reductions = list(self.feature_info.reduction())

    def forward(self, x):
        return {
            f"f{i + 1}": feature
            for i, feature in enumerate(self.encoder(x))
        }
