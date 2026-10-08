"""Training/evaluation interfaces for the dataset adapter to be added in step 3."""

from contextlib import nullcontext
import math
import torch
from .losses import align_target


def build_optimizer(model):
    cfg = model.config
    groups = [{"params": list(model.head.parameters()), "lr": cfg.learning_rate}]
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if trainable_backbone:
        groups.append({"params": trainable_backbone, "lr": cfg.backbone_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def move_batch(batch, device):
    if "pixel_values" not in batch or "density" not in batch:
        raise KeyError("A batch needs pixel_values and density; frame_mask is optional.")
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_step(model, batch, optimizer, criterion, amp=False, scaler=None, max_grad_norm=1.0):
    """One optimizer step. Caller controls epochs, sampler and dataset split.

    AMP uses float16 on CUDA and requires an externally retained GradScaler.
    This function does not silently create a fresh scaler at each iteration.
    """
    model.train()
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    if amp and (device.type != "cuda" or scaler is None):
        raise ValueError("AMP requires CUDA and a persistent torch.cuda.amp.GradScaler.")
    optimizer.zero_grad(set_to_none=True)
    autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    with autocast:
        outputs = model(batch["pixel_values"])
    # Density/count reductions remain float32, including under AMP.
    losses = criterion(outputs, batch["density"], batch.get("frame_mask"))
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Non-finite loss; optimizer update was not performed.")
    if amp:
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
    else:
        losses["loss"].backward()
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=not amp)
    elif not amp and any(not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("Non-finite gradient; optimizer update was not performed.")
    if amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return {name: value.detach().item() for name, value in losses.items()}


@torch.no_grad()
def evaluate(model, batches):
    """Aggregate MAE/RMSE over labeled frames, not over batch averages."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    absolute_error = squared_error = 0.0
    num_frames = 0
    try:
        for batch in batches:
            batch = move_batch(batch, device)
            outputs = model(batch["pixel_values"])
            target, valid = align_target(outputs["density"], batch["density"], batch.get("frame_mask"))
            errors = outputs["counts"][valid] - target.sum(dim=(2, 3, 4))[valid]
            if not torch.isfinite(errors).all():
                raise FloatingPointError("Non-finite evaluation counts.")
            absolute_error += errors.abs().sum().item()
            squared_error += errors.square().sum().item()
            num_frames += errors.numel()
    finally:
        model.train(was_training)
    if num_frames == 0:
        raise ValueError("No labeled frames were evaluated.")
    return {"mae": absolute_error / num_frames,
            "rmse": math.sqrt(squared_error / num_frames), "frames": num_frames}
