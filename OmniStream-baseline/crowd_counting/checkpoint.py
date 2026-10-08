"""Save/restore the complete counting model, independently of pretrained files."""

from pathlib import Path
import os
import tempfile
import torch
from .config import CrowdConfig
from .model import OmniStreamCrowd


def save_checkpoint(path, model, optimizer=None, epoch=0, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "crowd_config": model.config.to_dict(),
        "backbone_config": model.backbone.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
    }
    # Write beside the destination so the final replacement stays on one volume.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device="cpu"):
    """Return (model, training_state); no network or original checkpoint required.

    Restore optimizer/scaler states after creating them from the returned model.
    Sampler/RNG states are not saved: this restores training state, not bitwise replay.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported counting checkpoint format.")
    model = OmniStreamCrowd.from_backbone_config(
        payload["backbone_config"], CrowdConfig(**payload["crowd_config"])
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    state = {name: payload.get(name) for name in ("optimizer", "scaler", "epoch")}
    return model, state
