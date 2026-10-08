"""Baseline density MSE, optional count L1, and labeled-frame masking."""

import torch
from torch import nn
import torch.nn.functional as F
from .density import sum_preserving_downsample


def align_target(prediction, target, frame_mask=None):
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError("Both densities must have shape (B,T,1,H,W).")
    if target.shape[:3] != prediction.shape[:3] or prediction.shape[2] != 1:
        raise ValueError("Prediction and target B,T,1 dimensions must match.")
    target = target.to(device=prediction.device, dtype=torch.float32)
    if frame_mask is None:
        frame_mask = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    elif frame_mask.dtype != torch.bool or tuple(frame_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("frame_mask must be bool with shape (B,T).")
    else:
        frame_mask = frame_mask.to(prediction.device)
    if not frame_mask.any():
        raise ValueError("A batch must contain at least one labeled frame.")
    selected = target[frame_mask]
    if not torch.isfinite(selected).all() or (selected < 0).any():
        raise ValueError("Labeled target densities must be finite and nonnegative.")
    # Unlabeled frames are ignored; callers should normally fill them with zeros.
    target = target.masked_fill(~frame_mask[..., None, None, None], 0)
    if target.shape[-2:] != prediction.shape[-2:]:
        target = sum_preserving_downsample(target, prediction.shape[-2:])
    return target, frame_mask


class CrowdCountingLoss(nn.Module):
    def __init__(self, density_weight=1.0, count_weight=0.0):
        super().__init__()
        if density_weight <= 0 or count_weight < 0:
            raise ValueError("Invalid loss weights.")
        self.density_weight = density_weight
        self.count_weight = count_weight

    def forward(self, outputs, target_density, frame_mask=None):
        prediction = outputs["density"].float()
        target, valid = align_target(prediction, target_density, frame_mask)
        pred, truth = prediction[valid], target[valid]
        density_mse = F.mse_loss(pred, truth)
        pred_counts = pred.sum(dim=(1, 2, 3))
        true_counts = truth.sum(dim=(1, 2, 3))
        count_l1 = F.l1_loss(pred_counts, true_counts)
        return {
            "loss": self.density_weight * density_mse + self.count_weight * count_l1,
            "density_mse": density_mse,
            "count_l1": count_l1,
        }
