import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = logits.sigmoid().flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(1)
    dice = (2 * intersection + eps) / (probs.sum(1) + targets.sum(1) + eps)
    return 1 - dice.mean()

def bce_dice_loss(logits, targets, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    return bce_weight * bce + (1 - bce_weight) * dice_loss(logits, targets)
