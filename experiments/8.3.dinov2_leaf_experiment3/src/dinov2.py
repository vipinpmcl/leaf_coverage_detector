import torch
from torchvision import transforms

class DINOv2Backbone:
    def __init__(self, name="dinov2_vits14", image_size=224, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.hub.load(
            "facebookresearch/dinov2", name, pretrained=True
        ).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))
        ])

    def set_trainable(self, trainable):
        for p in self.model.parameters():
            p.requires_grad = trainable

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def features(self, x):
        return self.model.forward_features(x)["x_norm_patchtokens"]
