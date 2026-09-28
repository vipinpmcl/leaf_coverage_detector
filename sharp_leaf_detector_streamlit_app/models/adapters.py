import torch.nn as nn


class FeatureAdapters(nn.Module):
    def __init__(self, in_channels, out_channels=(64, 96, 192, 384)):
        super().__init__()

        if len(in_channels) != len(out_channels):
            raise ValueError(
                f"Feature count mismatch: backbone={len(in_channels)}, "
                f"adapters={len(out_channels)}"
            )

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False), #conv_1x1
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
            for cin, cout in zip(in_channels, out_channels)
        ])

    def forward(self, features):
        return [
            adapter(features[f"f{i + 1}"])
            for i, adapter in enumerate(self.adapters)
        ]
