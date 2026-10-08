"""Opt-in Bus diagnostics. No training, parameter updates, or file reads at import."""
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .bus import mask_prediction


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def panel(path, sample):
    """Native density grid is displayed enlarged; saved NPY contains count mass."""
    rgb, roi, truth, prediction = sample
    scale = max(float(truth.max()), float(prediction.max()), 1e-12)
    error = prediction - truth
    error_scale = max(float(np.abs(error).max()), 1e-12)
    def heat(a):
        x = np.clip(a / scale, 0, 1)
        return np.stack([255*x, 180*x, 40*x], -1).astype(np.uint8)
    x = error / error_scale
    err = np.stack([255*(1 + np.minimum(x, 0)), 255*(1-np.abs(x)),
                    255*(1-np.maximum(x, 0))], -1).astype(np.uint8)
    images = [rgb, np.repeat((roi*255).astype(np.uint8)[..., None], 3, -1),
              heat(truth), heat(prediction), err]
    titles = ["RGB", "ROI", f"GT sum={truth.sum():.3f}",
              f"Pred sum={prediction.sum():.3f}", "Error: red=over, blue=under"]
    w, h = 320, max(1, round(rgb.shape[0] * 320 / rgb.shape[1]))
    canvas = Image.new("RGB", (w*5, h+65), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (array, title) in enumerate(zip(images, titles)):
        tile = Image.fromarray(array).resize((w, h), Image.Resampling.NEAREST)
        canvas.paste(tile, (i*w, 25))
        draw.text((i*w+4, 5), title, fill="black")
    draw.text((5, h+32), f"GT/Pred shared scale: 0..{scale:.6g}; error: +/-{error_scale:.6g}. Display enlargement only.", fill="black")
    canvas.save(path)


def timeline(path, rows):
    """Dependency-free chronological count chart; CSV retains frame identifiers."""
    canvas = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(canvas)
    vals = [r[k] for r in rows for k in ("target", "prediction")]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1
    for i in range(6):
        y = 350-i*60
        draw.line((70, y, 1170, y), fill="#dddddd")
        draw.text((3, y-5), f"{lo+(hi-lo)*i/5:.2f}", fill="black")
    for key, color in (("target", "#238544"), ("prediction", "#2865bf")):
        pts = [(70+i*1100/max(1, len(rows)-1), 350-(r[key]-lo)*300/(hi-lo))
               for i, r in enumerate(rows)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        else:
            x, y = pts[0]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    draw.text((70, 12), "Count by chronological sample index | green: GT | blue: prediction", fill="black")
    draw.text((70, 375), f"0: {rows[0]['frame_id']}       last ({len(rows)-1}): {rows[-1]['frame_id']} | see frames.csv for IDs", fill="black")
    canvas.save(path)


class BusDiagnostics:
    def __init__(self, directory, dataset_size, samples=6, temporal=False, features=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.limit = samples
        self.fixed_indices = set(np.linspace(0, dataset_size-1, samples, dtype=int)) if samples else set()
        self.rows, self.worst, self.temporal_rows, self.feature_rows = [], [], [], []
        self.temporal, self.features = temporal, features
        self.previous = None
        self.handles = []

    def start_batch(self, model, index):
        self.end_batch()
        if not self.features or index not in self.fixed_indices:
            return
        # Hooks only inspect selected validation batches and never retain GPU tensors.
        def capture(name, value):
            a = value.detach().float()
            finite = torch.isfinite(a)
            good = a[finite]
            self.feature_rows.append({"batch_index": index, "module": name,
                "shape": list(a.shape), "nonfinite": int((~finite).sum()),
                "mean": float(good.mean()) if good.numel() else None,
                "std": float(good.std(unbiased=False)) if good.numel() else None})
            if a.ndim == 4:
                np.save(self.directory / f"features_{index:05d}_{name}.npy",
                        a[-1, :4].cpu().numpy(), allow_pickle=False)
        self.handles.append(model.head.register_forward_pre_hook(
            lambda module, args: capture("backbone_spatial", args[0])))
        for name, layer in (("head_stage1", model.head.count[6]),
                            ("head_stage2", model.head.count[10])):
            self.handles.append(layer.register_forward_hook(
                lambda module, args, out, name=name: capture(name, out)))

    def end_batch(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def add(self, model, batch, raw, outputs, target, valid, amp):
        self.end_batch()
        for b, t in valid.nonzero(as_tuple=False).tolist():
            name = batch["frame_id"][b]
            pred = outputs["density"][b, t, 0].detach().float().cpu().numpy()
            gt = target[b, t, 0].detach().float().cpu().numpy()
            raw_map = raw["density"][b, t, 0].detach().float()
            if not torch.isfinite(raw_map).all():
                raise FloatingPointError(f"Non-finite raw density: {name}")
            prediction = float(outputs["counts"][b, t])
            truth = float(target[b, t].sum())
            original_truth = float(batch["density"][b, t].double().sum())
            err = prediction-truth
            row = {"frame_id": name, "target": truth, "prediction": prediction,
                   "signed_error": err, "absolute_error": abs(err),
                   "density_mse": float(np.mean((pred-gt)**2)),
                   "raw_max": float(raw_map.max()), "raw_zero_fraction": float((raw_map == 0).float().mean()),
                   "raw_total": float(raw_map.sum()), "roi_total": prediction,
                   "excluded_roi_mass": float(raw_map.sum())-prediction,
                   "target_mass_difference": truth-original_truth}
            index = len(self.rows)
            self.rows.append(row)
            keep = self.limit and (len(self.worst) < self.limit or abs(err) > self.worst[-1][0])
            if index in self.fixed_indices or keep:
                rgb = batch["pixel_values"][b, t].detach().cpu().permute(1, 2, 0).numpy()
                roi = batch["roi"][b, 0].detach().cpu().numpy()
                sample = (rgb, roi, gt, pred)
                if index in self.fixed_indices:
                    self.save_sample("fixed", name, sample)
                if keep:
                    self.worst.append((abs(err), name, sample))
                    self.worst.sort(key=lambda item: item[0], reverse=True)
                    self.worst = self.worst[:self.limit]
        if self.temporal:
            self.temporal_check(model, batch, outputs, target, amp)

    @torch.no_grad()
    def temporal_check(self, model, batch, outputs, target, amp):
        pixels = batch["pixel_values"]
        if pixels.shape[0] != 1 or pixels.shape[1] < 2:
            raise ValueError("Temporal diagnostics require evaluation batch size 1 and at least 2 frames.")
        truth = float(target[0, -1].sum())
        normal = float(outputs["counts"][0, -1])
        name = batch["frame_id"][0]
        self.temporal_rows.append({"frame_id": name, "variant": "real_history", "donor": "",
            "target": truth, "prediction": normal, "absolute_error": abs(normal-truth),
            "count_delta": 0.0, "density_l1_delta": 0.0})
        variants = [("repeat_current", pixels[:, -1:].expand_as(pixels).clone(), name)]
        if self.previous is not None:
            changed = pixels.clone()
            changed[:, :-1] = self.previous[1].to(pixels.device).expand_as(changed[:, :-1])
            variants.append(("fixed_earlier_donor", changed, self.previous[0]))
        for label, changed, donor in variants:
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                alternate = model(changed)
            alternate = mask_prediction(alternate, batch["roi"])
            p = float(alternate["counts"][0, -1])
            if not np.isfinite(p):
                raise FloatingPointError("Non-finite temporal prediction")
            delta = float((alternate["density"][0, -1]-outputs["density"][0, -1]).abs().mean())
            self.temporal_rows.append({"frame_id": name, "variant": label, "donor": donor,
                "target": truth, "prediction": p, "absolute_error": abs(p-truth),
                "count_delta": p-normal, "density_l1_delta": delta})
        if self.previous is None:
            self.previous = (name, pixels[:, -1:].detach().cpu().clone())

    def save_sample(self, group, name, sample):
        stem = self.directory / f"{group}_{name}"
        panel(stem.with_suffix(".png"), sample)
        np.savez_compressed(stem.with_suffix(".npz"), target=sample[2], prediction=sample[3])

    def finish(self):
        self.end_batch()
        write_csv(self.directory / "frames.csv", self.rows)
        write_csv(self.directory / "temporal.csv", self.temporal_rows)
        if self.rows:
            timeline(self.directory / "counts.png", self.rows)
            errors = np.asarray([r["signed_error"] for r in self.rows])
            overview = {"frames": len(self.rows), "mae": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "mean_signed_error": float(errors.mean()),
                "density_mse": float(np.mean([r["density_mse"] for r in self.rows])),
                "max_target_mass_difference": max(abs(r["target_mass_difference"]) for r in self.rows),
                "fixed_indices": sorted(self.fixed_indices),
                "note": "Raw ROI-excluded mass is diagnostic only. Temporal replacement is a sensitivity probe, not a retrained ablation. Panels use a shared GT/pred scale within each panel; scales can differ between panels/epochs."}
            (self.directory / "summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
        for _, name, sample in self.worst:
            self.save_sample("worst", name, sample)
        (self.directory / "features.json").write_text(json.dumps(self.feature_rows, indent=2), encoding="utf-8")
        summary = {}
        for variant in sorted({r["variant"] for r in self.temporal_rows}):
            subset = [r for r in self.temporal_rows if r["variant"] == variant]
            summary[variant] = {"frames": len(subset),
                "mae": float(np.mean([r["absolute_error"] for r in subset])),
                "paired_real_mae": float(np.mean([abs(r["prediction"]-r["count_delta"]-r["target"]) for r in subset]))}
        (self.directory / "temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
