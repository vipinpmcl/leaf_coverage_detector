import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class MultiScaleDecoder(nn.Module):
    def __init__(self, channels=(64, 96, 192, 384), decoder_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_channels, 1) for c in channels])
        self.refine = nn.ModuleList([ConvBlock(decoder_channels, decoder_channels) for _ in channels])
        self.head = nn.Sequential(
            ConvBlock(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

    def forward(self, features):
        x = self.lateral[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.lateral[i](features[i])
            x = self.refine[i](x)
        return self.head(x)
