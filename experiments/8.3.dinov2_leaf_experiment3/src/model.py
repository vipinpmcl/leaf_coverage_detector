import torch.nn as nn
import torch.nn.functional as F

class LeafSegmentationModel(nn.Module):
    def __init__(self, backbone, feature_dim=384, hidden_dim=192, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, x):
        tokens = self.backbone.features(x)
        b, n, c = tokens.shape
        grid = int(n ** 0.5)

        if grid * grid != n:
            raise RuntimeError(f"Expected square patch grid, got {n} patches")

        features = tokens.transpose(1, 2).reshape(b, c, grid, grid)
        logits = self.decoder(features)

        return F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )[:, 0]
