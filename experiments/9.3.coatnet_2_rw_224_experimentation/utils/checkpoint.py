from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)


def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)

    return checkpoint
