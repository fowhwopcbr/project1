"""Inspect a saved Bus checkpoint on validation only; never train or open test H5/images.

Constants are fitted on manifest['train'] full-frame ROI counts, excluding val/gap.
Uses the existing data/model/ROI/target-alignment implementation without changing it.
"""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from crowd_counting.bus import BusClips, load_density, load_roi, mask_prediction, read_manifest
from crowd_counting.checkpoint import load_checkpoint
from crowd_counting.engine import move_batch
from crowd_counting.losses import align_target


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite counts.")
    return {"frames": int(values.size), "min": float(values.min()),
            "max": float(values.max()), "mean": float(values.mean()),
            "median": float(np.median(values)), "std_population": float(values.std())}


def metrics(predictions, targets):
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Expected nonempty finite errors.")
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean())),
            "mean_signed_error": float(errors.mean()), "frames": int(errors.size)}


def training_counts(manifest, use_roi):
    # Do not iterate a train-mode BusClips instance: its crops/flips are augmentations.
    root = Path(manifest["root"])
    names = manifest["train"]
    with Image.open(root / "train" / "images" / (names[0] + ".jpg")) as image:
        shape = (image.height, image.width)
    roi = load_roi(root, shape, use_roi)
    rows = []
    for index, name in enumerate(names):
        density = load_density(root / "train" / "ground_truth" / (name + ".h5"), shape)
        count = float((density * roi).sum(dtype=np.float64))
        if not np.isfinite(count):
            raise ValueError(f"Non-finite training count: {name}")
        rows.append({"frame_id": name, "target_count": count})
        if (index + 1) % 250 == 0:
            print(f"Read training labels: {index + 1}/{len(names)}", flush=True)
    return rows


@torch.no_grad()
def validation_rows(model, dataset, device, workers, amp, mean_count, median_count):
    model.eval()
    batches = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                         pin_memory=device.type == "cuda")
    rows = []
    for index, batch in enumerate(batches):
        name = batch["frame_id"][0]
        if name != dataset.names[index]:
            raise ValueError("Unexpected validation order.")
        batch = move_batch(batch, device)
        frame_mask = batch["frame_mask"]
        if (tuple(frame_mask.shape) != (1, dataset.frames)
                or not bool(frame_mask[0, -1]) or int(frame_mask.sum()) != 1):
            raise ValueError("Expected supervision on the last frame only.")
        with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
            outputs = model(batch["pixel_values"])
        outputs = mask_prediction(outputs, batch["roi"])
        target, valid = align_target(outputs["density"], batch["density"], frame_mask)
        # Same count and ROI convention as train_bus.evaluate; sum before subtraction.
        prediction = float(outputs["counts"][valid].item())
        truth = float(target.sum(dim=(2, 3, 4))[valid].item())
        if not np.isfinite([prediction, truth]).all():
            raise FloatingPointError(f"Non-finite validation count: {name}")
        raw_truth = float(batch["density"][valid].double().sum().item())
        if not np.isclose(truth, raw_truth, rtol=1e-5, atol=1e-4):
            raise ValueError(f"Target downsampling changed count mass: {name}")
        context = [dataset.names[max(0, index - dataset.frames + 1 + offset)]
                   for offset in range(dataset.frames)]
        error = prediction - truth
        rows.append({"frame_id": name, "input_frames": ";".join(context),
                     "target_count": truth, "prediction": prediction,
                     "signed_error": error, "absolute_error": abs(error),
                     "squared_error": error * error,
                     "train_mean_prediction": mean_count,
                     "train_mean_absolute_error": abs(mean_count - truth),
                     "train_median_prediction": median_count,
                     "train_median_absolute_error": abs(median_count - truth),
                     "target_mass_difference": truth - raw_truth})
        if (index + 1) % 100 == 0:
            print(f"Evaluated validation frames: {index + 1}/{len(dataset)}", flush=True)
    if [row["frame_id"] for row in rows] != dataset.names:
        raise ValueError("Validation coverage differs from the manifest.")
    return rows


def compare_history(path, epoch, current):
    if not path.is_file():
        return {"status": "unavailable", "reason": "history.jsonl is missing"}
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    matches = [record for record in records if record.get("epoch") == epoch]
    if len(matches) != 1:
        return {"status": "unavailable", "reason": "checkpoint epoch is not uniquely recorded"}
    previous = matches[0].get("val", {})
    if not all(key in previous for key in ("mae", "rmse", "frames")):
        return {"status": "unavailable", "reason": "validation metrics are missing"}
    delta = {key: current[key] - previous[key] for key in ("mae", "rmse")}
    matched = (previous["frames"] == current["frames"]
               and all(abs(value) <= 1e-3 for value in delta.values()))
    return {"status": "matched" if matched else "MISMATCH", "recorded": previous,
            "difference": delta, "absolute_tolerance": 1e-3}


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", help="New directory; default: checkpoint directory/val_diagnostics")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers cannot be negative")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "val_diagnostics"
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output-dir")
    settings_path = checkpoint.parent / "settings.json"
    manifest_path = checkpoint.parent / "manifest.json"
    with settings_path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    manifest = read_manifest(manifest_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current Python environment.")
    seed = int(settings.get("seed", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    amp = bool(settings.get("amp", False)) and device.type == "cuda"
    use_roi = bool(settings["use_roi"])
    print("Validation diagnosis only. Constants use manifest.train; no training or test inference.", flush=True)
    train_rows = training_counts(manifest, use_roi)
    train_stats = describe([row["target_count"] for row in train_rows])
    mean_count, median_count = train_stats["mean"], train_stats["median"]
    dataset = BusClips(manifest, "val", frames=settings["frames"],
                       crop_size=settings["crop_size"], use_roi=use_roi,
                       full_frame=settings.get("full_frame", False))
    print("Loading saved checkpoint...", flush=True)
    model, state = load_checkpoint(checkpoint, device)
    restored_config = json.loads(json.dumps(model.config.to_dict()))
    if restored_config != settings["model_config"]:
        raise ValueError("Checkpoint model configuration does not match settings.json.")
    rows = validation_rows(model, dataset, device, args.workers, amp, mean_count, median_count)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target_count"] for row in rows], dtype=np.float64)
    model_metrics = metrics(predictions, targets)
    mean_metrics = metrics(np.full_like(targets, mean_count), targets)
    median_metrics = metrics(np.full_like(targets, median_count), targets)
    target_std, prediction_std = float(targets.std()), float(predictions.std())
    correlation = (float(np.corrcoef(predictions, targets)[0, 1])
                   if min(target_std, prediction_std) > 1e-12 else None)
    blocks = []
    for number, indices in enumerate(np.array_split(np.arange(len(rows)), min(4, len(rows))), start=1):
        blocks.append({"block": number, "first_frame": rows[int(indices[0])]["frame_id"],
                       "last_frame": rows[int(indices[-1])]["frame_id"],
                       "target_counts": describe(targets[indices]),
                       "model": metrics(predictions[indices], targets[indices]),
                       "train_mean_baseline": metrics(np.full(len(indices), mean_count), targets[indices]),
                       "train_median_baseline": metrics(np.full(len(indices), median_count), targets[indices])})
    history = compare_history(checkpoint.parent / "history.jsonl", state["epoch"], model_metrics)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (settings_path, manifest_path, Path(__file__).resolve())}
    summary = {
        "split": "val", "checkpoint": str(checkpoint), "epoch": state["epoch"],
        "device": str(device), "amp": amp, "use_roi": use_roi,
        "clip_frames": dataset.frames, "image_shape_hw": list(dataset.shape),
        "training_full_frame": bool(settings.get("full_frame", False)),
        "count_reference": "supplied H5 density integral, using saved ROI setting",
        "constant_fit_partition": "manifest.train only; full frames; no augmentation",
        "test_images_and_labels_opened": False,
        "train_target_counts": train_stats, "val_target_counts": describe(targets),
        "val_prediction_counts": describe(predictions), "model": model_metrics,
        "train_mean_baseline": {"constant": mean_count, **mean_metrics},
        "train_median_baseline": {"constant": median_count, **median_metrics},
        "mae_gain_over_train_mean": mean_metrics["mae"] - model_metrics["mae"],
        "mae_gain_over_train_median": median_metrics["mae"] - model_metrics["mae"],
        "pearson_r": correlation,
        "prediction_std_over_target_std": prediction_std / target_std if target_std > 1e-12 else None,
        "history_comparison": history, "chronological_blocks": blocks,
        "largest_errors": sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[:10],
        "provenance_sha256": provenance,
        "notes": ["Positive signed error means overcounting; positive MAE gain means the model beats that constant.",
                  "Correlation and variance ratio are descriptive, not proof of correct localization or temporal benefit.",
                  "A history mismatch requires checking weights, data, ROI and runtime settings before interpreting gains.",
                  "No optimizer, parameter update, test inference, MAT point audit or density-map visualization is performed."]}
    # Refuse existing output directories and files. Original run files are never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "train_counts.csv", train_rows)
    write_csv(output / "val_predictions.csv", rows)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output_dir": str(output), "model": model_metrics,
                      "train_mean_baseline": summary["train_mean_baseline"],
                      "train_median_baseline": summary["train_median_baseline"],
                      "pearson_r": correlation,
                      "prediction_std_over_target_std": summary["prediction_std_over_target_std"],
                      "history_comparison": history}, indent=2, allow_nan=False), flush=True)
    print("Saved summary.json, val_predictions.csv and train_counts.csv.", flush=True)


if __name__ == "__main__":
    main()
