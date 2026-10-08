"""DSTI-format Bus clips. Dataset files are opened read-only.

Each sample supervises only its last frame, so evaluation counts each frame once.
At a partition boundary the earliest available frame is repeated as context.
"""
from pathlib import Path
import json
import random
import re

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def catalog(root, split):
    root = Path(root)
    images = list((root / split / "images").glob("bus_*.jpg"))
    if not images:
        raise FileNotFoundError(f"No bus_*.jpg images under {root / split / 'images'}")
    def number(path):
        match = re.fullmatch(r"bus_(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Unexpected frame name: {path}")
        return int(match.group(1))
    images.sort(key=number)
    ids = [number(p) for p in images]
    if any(b - a != 10 for a, b in zip(ids, ids[1:])):
        raise ValueError("This Bus adapter expects a single sequence sampled every 10 frame IDs.")
    for path in images:
        label = root / split / "ground_truth" / (path.stem + ".h5")
        if not label.is_file():
            raise FileNotFoundError(label)
    return [p.stem for p in images]


def make_manifest(root, val_fraction=0.2, gap=16):
    if not 0 < val_fraction < 1 or gap < 0:
        raise ValueError("Require 0 < val_fraction < 1 and gap >= 0.")
    train = catalog(root, "train")
    test = catalog(root, "test")
    if set(train) & set(test):
        raise ValueError("Train/test frame IDs overlap.")
    cut = int(len(train) * (1 - val_fraction))
    if cut <= gap or cut >= len(train):
        raise ValueError("Insufficient frames for this validation split and gap.")
    return {"format_version": 1, "root": str(Path(root).resolve()),
            "train": train[:cut-gap], "gap": train[cut-gap:cut],
            "val": train[cut:], "test": test,
            "val_fraction": val_fraction, "gap_frames": gap,
            "protocol": "Chronological internal validation; official test kept separate."}


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported Bus manifest.")
    seen = set()
    for split in ("train", "gap", "val", "test"):
        names = manifest[split]
        if split != "gap" and not names:
            raise ValueError(f"Empty partition: {split}")
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Repeated frames across/within partitions.")
        if any(re.fullmatch(r"bus_\d+", name) is None for name in names):
            raise ValueError("Invalid frame ID in manifest.")
        ids = [int(n.split("_")[-1]) for n in names]
        if any(b - a != 10 for a, b in zip(ids, ids[1:])):
            raise ValueError(f"Nonconsecutive frames in {split}.")
        seen.update(names)
    return manifest


def load_roi(root, shape, enabled=True):
    if not enabled:
        return np.ones(shape, dtype=np.float32)
    roi = np.load(Path(root) / "bus_roi.npy", allow_pickle=False)
    if roi.shape != shape or not np.isfinite(roi).all():
        raise ValueError(f"ROI must be finite with shape {shape}; got {roi.shape}.")
    if not np.isin(roi, [0, 1, 255]).all():
        raise ValueError("Expected binary ROI encoded as 0/1 or 0/255.")
    if not np.any(roi):
        raise ValueError("ROI is empty.")
    return (roi != 0).astype(np.float32)


def load_density(path, shape):
    with h5py.File(path, "r") as handle:
        if "density" not in handle:
            raise KeyError(f"{path}: missing 'density'; keys={list(handle.keys())}")
        density = np.asarray(handle["density"], dtype=np.float32)
    if density.shape != shape:
        raise ValueError(f"{path}: density shape {density.shape} != image shape {shape}")
    if not np.isfinite(density).all() or (density < 0).any():
        raise ValueError(f"{path}: density must be finite and nonnegative.")
    return density


class BusClips(Dataset):
    def __init__(self, manifest, split, frames=2, crop_size=256, use_roi=True, full_frame=False):
        if split not in ("train", "val", "test") or frames < 1:
            raise ValueError("Invalid split or frame count.")
        if not full_frame and (crop_size < 16 or crop_size % 16):
            raise ValueError("crop_size must be a positive multiple of 16.")
        self.root = Path(manifest["root"])
        self.names = manifest[split]
        self.source = "test" if split == "test" else "train"
        self.split, self.frames, self.crop_size = split, frames, crop_size
        self.full_frame = full_frame
        with Image.open(self.image_path(self.names[0])) as image:
            self.shape = (image.height, image.width)
        if any(size % 16 for size in self.shape):
            raise ValueError("Bus full-frame dimensions must be divisible by patch size 16.")
        if split == "train" and not full_frame and min(self.shape) < crop_size:
            raise ValueError("Training crop exceeds image dimensions.")
        self.roi = load_roi(self.root, self.shape, use_roi)

    def image_path(self, name):
        return self.root / self.source / "images" / (name + ".jpg")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        context = [self.names[max(0, index - self.frames + 1 + offset)]
                   for offset in range(self.frames)]
        images = []
        for name in context:
            with Image.open(self.image_path(name)) as image:
                array = np.array(image.convert("RGB"), dtype=np.uint8)
            if array.shape[:2] != self.shape:
                raise ValueError(f"Image dimensions changed at {name}.")
            images.append(array)
        name = self.names[index]
        target = load_density(self.root / self.source / "ground_truth" / (name + ".h5"), self.shape)
        roi = self.roi
        target = target * roi
        if self.split == "train":
            if not self.full_frame:
                top = random.randint(0, self.shape[0] - self.crop_size)
                left = random.randint(0, self.shape[1] - self.crop_size)
                ys, xs = slice(top, top + self.crop_size), slice(left, left + self.crop_size)
                images = [image[ys, xs] for image in images]
                target, roi = target[ys, xs], roi[ys, xs]
            # Keep clip-consistent augmentation in both training modes.
            # Full-frame mode never crops or resizes images, labels, or ROI.
            if random.random() < 0.5:
                images = [image[:, ::-1] for image in images]
                target, roi = target[:, ::-1], roi[:, ::-1]
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
        density = torch.zeros(self.frames, 1, *target.shape, dtype=torch.float32)
        density[-1, 0] = torch.from_numpy(target.copy())
        frame_mask = torch.zeros(self.frames, dtype=torch.bool)
        frame_mask[-1] = True
        return {"pixel_values": pixels, "density": density, "frame_mask": frame_mask,
                "roi": torch.from_numpy(roi.copy())[None], "frame_id": name}


def mask_prediction(outputs, roi):
    """ROI area fraction per output cell; target was masked before block summing.

    Boundary cells assume uniform predicted mass within their 4x4 source area.
    This is an explicit ROI counting convention, not a change to the model head.
    """
    density = outputs["density"].float()
    height, width = roi.shape[-2:]
    out_h, out_w = density.shape[-2:]
    if height % out_h or width % out_w:
        raise ValueError("ROI dimensions must be divisible by output dimensions.")
    weight = F.avg_pool2d(roi.float(), (height // out_h, width // out_w))
    density = density * weight[:, None]
    return {"density": density, "counts": density.sum(dim=(2, 3, 4))}
