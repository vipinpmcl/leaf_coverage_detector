import torch

def segmentation_metrics(logits, target, threshold=0.5):
    probability = torch.sigmoid(logits)
    prediction = probability >= threshold
    target = target >= 0.5

    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice
    }
