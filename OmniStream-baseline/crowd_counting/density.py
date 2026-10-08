"""Density-map operations: each cell stores count mass, not display intensity."""

import torch
import torch.nn.functional as F


def sum_preserving_downsample(density, size):
    """Sum disjoint pixel blocks. Supports (..., H, W), including B,T,1,H,W.

    Only integer downsampling is accepted. Reconstruct labels from transformed
    point coordinates for other image transforms; ordinary interpolation is not
    a count-preserving operation.
    """
    height, width = density.shape[-2:]
    out_h, out_w = size
    if min(out_h, out_w) <= 0 or height % out_h or width % out_w:
        raise ValueError("Target density size must divide the source size exactly.")
    if out_h > height or out_w > width:
        raise ValueError("This function downsamples density labels only.")
    factor_h, factor_w = height // out_h, width // out_w
    flat = density.reshape(-1, 1, height, width).float()
    pooled = F.avg_pool2d(flat, (factor_h, factor_w), (factor_h, factor_w))
    return (pooled * factor_h * factor_w).reshape(*density.shape[:-2], out_h, out_w)


def resize_density_for_display(density, size):
    """Bilinear resizing with per-map sum correction. Not used to create targets."""
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("size must be a positive (height, width).")
    old_h, old_w = density.shape[-2:]
    flat = density.reshape(-1, 1, old_h, old_w).float()
    if not torch.isfinite(flat).all() or (flat < 0).any():
        raise ValueError("Density maps must be finite and nonnegative.")
    resized = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    old_mass = flat.sum(dim=(-2, -1), keepdim=True)
    new_mass = resized.sum(dim=(-2, -1), keepdim=True)
    # A small isolated peak can disappear when downsampling by interpolation.
    if ((old_mass > 0) & (new_mass == 0)).any():
        raise ValueError("Resizing lost a nonzero density peak; use block sums for downsampling.")
    resized = resized * (old_mass / new_mass.clamp_min(torch.finfo(torch.float32).tiny))
    return resized.reshape(*density.shape[:-2], *size)
