import torch
import torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    probability = torch.sigmoid(logits)
    probability = probability.flatten(1)
    target = target.flatten(1)

    intersection = (probability * target).sum(1)
    denominator = probability.sum(1) + target.sum(1)

    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()

def segmentation_loss(logits, target, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice
