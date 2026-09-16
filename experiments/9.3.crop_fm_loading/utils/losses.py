import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)

    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, targets):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice
