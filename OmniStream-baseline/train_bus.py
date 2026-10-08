"""Bus real-data check, training, and separate checkpoint evaluation.

Run prepare_bus.py first. No action is taken until main() is called.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from crowd_counting import CrowdConfig, OmniStreamCrowd, CrowdCountingLoss
from crowd_counting.bus import BusClips, read_manifest, mask_prediction
from crowd_counting.checkpoint import save_checkpoint, load_checkpoint
from crowd_counting.engine import build_optimizer, move_batch
from crowd_counting.losses import align_target


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, batch_size, workers, shuffle, seed, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def step(model, batch, optimizer, criterion, amp, scaler):
    model.train()
    batch = move_batch(batch, next(model.parameters()).device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
        outputs = model(batch["pixel_values"])
    outputs = mask_prediction(outputs, batch["roi"])
    losses = criterion(outputs, batch["density"], batch["frame_mask"])
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Non-finite loss.")
    if amp:
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
    else:
        losses["loss"].backward()
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=not amp)
    if amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    result = {key: float(value.detach()) for key, value in losses.items()}
    result["gradient_norm"] = float(norm)
    result["optimizer_step_skipped"] = not math.isfinite(float(norm))
    return result


@torch.no_grad()
def evaluate(model, batches, amp=False, max_batches=None, diagnostics=None):
    model.eval()
    absolute = squared = 0.0
    count = 0
    for index, batch in enumerate(batches):
        batch = move_batch(batch, next(model.parameters()).device)
        if diagnostics is not None:
            diagnostics.start_batch(model, index)
        try:
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                raw_outputs = model(batch["pixel_values"])
        finally:
            if diagnostics is not None:
                diagnostics.end_batch()
        outputs = mask_prediction(raw_outputs, batch["roi"])
        target, valid = align_target(outputs["density"], batch["density"], batch["frame_mask"])
        errors = outputs["counts"][valid] - target.sum(dim=(2, 3, 4))[valid]
        if not torch.isfinite(errors).all():
            raise FloatingPointError("Non-finite evaluation counts.")
        if diagnostics is not None:
            diagnostics.add(model, batch, raw_outputs, outputs, target, valid, amp)
        absolute += errors.abs().sum().item()
        squared += errors.square().sum().item()
        count += errors.numel()
        if (index + 1) % 100 == 0:
            print(f"Evaluated {count} frames", flush=True)
        if max_batches is not None and index + 1 >= max_batches:
            break
    if count == 0:
        raise ValueError("No frames evaluated.")
    if diagnostics is not None:
        diagnostics.finish()
    return {"mae": absolute / count, "rmse": math.sqrt(squared / count), "frames": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "train", "evaluate"])
    parser.add_argument("--manifest", default="configs/bus_manifest.json")
    parser.add_argument("--config", default="configs/crowd_baseline.json")
    parser.add_argument("--run-dir", default="runs/bus_baseline")
    parser.add_argument("--checkpoint", help="Required for evaluate; uses settings and manifest saved beside checkpoint.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--full-frame", action="store_true",
                        help="Train/check at native image size without cropping or resizing; crop-size is ignored.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-roi", action="store_true", help="Explicit alternative full-image counting protocol.")
    parser.add_argument("--diagnostics", action="store_true", help="Save per-frame CSV, count chart, fixed/worst density panels.")
    parser.add_argument("--diagnostic-samples", type=int, default=6, help="Maximum fixed and worst panels per evaluation; 0 disables panels.")
    parser.add_argument("--diagnostic-features", action="store_true", help="Save selected validation feature statistics and four channels as NPY.")
    parser.add_argument("--temporal-diagnostics", action="store_true", help="Evaluate only: extra forwards with repeated current / fixed earlier donor frames.")
    parser.add_argument("--diagnostic-dir", help="Evaluate only: new, nonexisting output directory.")
    args = parser.parse_args()
    if args.diagnostic_samples < 0:
        parser.error("diagnostic-samples must be nonnegative.")
    if (args.temporal_diagnostics or args.diagnostic_dir) and args.mode != "evaluate":
        parser.error("temporal-diagnostics and diagnostic-dir are evaluate-only.")
    if args.mode == "check" and (args.diagnostics or args.diagnostic_features):
        parser.error("Diagnostics are available in train/evaluate modes.")
    args.diagnostics = args.diagnostics or args.diagnostic_features or args.temporal_diagnostics or bool(args.diagnostic_dir)
    if min(args.frames, args.batch_size, args.epochs) < 1 or args.workers < 0:
        parser.error("frames, batch-size and epochs must be positive; workers >= 0.")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires --device cuda.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected interpreter.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint.")
        checkpoint = Path(args.checkpoint)
        with (checkpoint.parent / "settings.json").open(encoding="utf-8") as handle:
            settings = json.load(handle)
        manifest = read_manifest(checkpoint.parent / "manifest.json")
        dataset = BusClips(manifest, args.split, frames=settings["frames"],
                           crop_size=settings["crop_size"], use_roi=settings["use_roi"],
                           full_frame=settings.get("full_frame", False))
        model, state = load_checkpoint(checkpoint, device)
        batches = loader(dataset, 1, args.workers, False, args.seed, device)
        diagnostic = None
        if args.diagnostics:
            from datetime import datetime
            from crowd_counting.diagnostics import BusDiagnostics
            if args.temporal_diagnostics and settings["frames"] < 2:
                parser.error("Temporal diagnostics require a checkpoint trained with at least 2 frames.")
            directory = args.diagnostic_dir or checkpoint.parent / "diagnostics" / (args.split + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            diagnostic = BusDiagnostics(directory, len(dataset), args.diagnostic_samples,
                                        args.temporal_diagnostics, args.diagnostic_features)
        metrics = evaluate(model, batches, args.amp, diagnostics=diagnostic)
        result = {"split": args.split, "epoch": state["epoch"], "checkpoint": str(checkpoint.resolve()),
                  "use_roi": settings["use_roi"], "amp": args.amp,
                  "training_full_frame": settings.get("full_frame", False),
                  "count_reference": "supplied H5 density integral", **metrics}
        if diagnostic is not None:
            result["diagnostic_directory"] = str(diagnostic.directory.resolve())
            (diagnostic.directory / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    if args.checkpoint:
        parser.error("--checkpoint is for evaluate only; this entry point does not resume training.")
    manifest = read_manifest(args.manifest)
    config = CrowdConfig.from_json(args.config)
    if config.input_is_normalized:
        raise ValueError("Bus loader produces raw uint8 RGB; input_is_normalized must be false.")
    if args.frames > config.temporal_length:
        raise ValueError("frames exceeds the model's fixed temporal_length.")
    train_data = BusClips(manifest, "train", args.frames, args.crop_size, not args.no_roi,
                          full_frame=args.full_frame)
    val_data = BusClips(manifest, "val", args.frames, args.crop_size, not args.no_roi,
                        full_frame=args.full_frame)
    input_hw = list(train_data.shape) if args.full_frame else [args.crop_size, args.crop_size]
    print(f"Training input HxW={input_hw}; full_frame={args.full_frame}; "
          f"frames={args.frames}; batch_size={args.batch_size}; ROI={not args.no_roi}", flush=True)
    train_batches = loader(train_data, args.batch_size, args.workers, True, args.seed, device)
    val_batches = loader(val_data, 1, args.workers, False, args.seed, device)
    run = Path(args.run_dir)
    if args.mode == "train" and run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run}; choose a new --run-dir.")
    model = OmniStreamCrowd.from_pretrained(config).to(device)
    optimizer = build_optimizer(model)
    criterion = CrowdCountingLoss(config.density_loss_weight, config.count_loss_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if args.mode == "check":
        batch = next(iter(train_batches))
        if list(batch["pixel_values"].shape[-2:]) != input_hw:
            raise AssertionError("Training batch spatial dimensions do not match the selected input mode.")
        # This audit should catch accidental target resizing that changes count mass.
        from crowd_counting.density import sum_preserving_downsample
        density = batch["density"]
        coarse = sum_preserving_downsample(density, (density.shape[-2] // 4, density.shape[-1] // 4))
        torch.testing.assert_close(coarse.sum((-2, -1)), density.sum((-2, -1)))
        result = step(model, batch, optimizer, criterion, args.amp, scaler)
        grads = [p.grad for p in model.head.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
            raise RuntimeError("Real-data check: density-head gradients are zero or non-finite.")
        if config.freeze_backbone and any(p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError("Frozen backbone received gradients.")
        result.update({"real_data_check": "passed", "input_shape": list(batch["pixel_values"].shape),
                       "full_frame": args.full_frame,
                       "target_counts": density.sum((2, 3, 4))[:, -1].tolist(),
                       "full_frame_val_check": evaluate(model, val_batches, args.amp, max_batches=1),
                       "notice": "One training batch and one validation frame only; no checkpoint saved or meaningful accuracy measured."})
        print(json.dumps(result, indent=2))
        return

    run.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, run / "manifest.json")
    settings = {**vars(args), "use_roi": not args.no_roi, "model_config": config.to_dict(),
                "training_input_hw": input_hw,
                "torch": torch.__version__, "count_reference": "supplied H5 density integral",
                "supervision": "last frame only; validation evaluates each frame once"}
    (run / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    best = float("inf")
    print(f"Training targets={len(train_data)}, validation targets={len(val_data)}; test is not evaluated.", flush=True)
    for epoch in range(1, args.epochs + 1):
        totals, samples, skipped = {}, 0, 0
        for index, batch in enumerate(train_batches):
            values = step(model, batch, optimizer, criterion, args.amp, scaler)
            size = batch["pixel_values"].shape[0]
            samples += size
            skipped += int(values["optimizer_step_skipped"])
            for key in ("loss", "density_mse", "count_l1"):
                totals[key] = totals.get(key, 0.0) + values[key] * size
            if index == 0 or (index + 1) % 50 == 0:
                print(f"epoch={epoch} batch={index+1}/{len(train_batches)} loss={values['loss']:.6g}", flush=True)
                if args.diagnostics:
                    status = {"epoch": epoch, "batch": index+1, **values,
                              "learning_rates": [g["lr"] for g in optimizer.param_groups]}
                    with (run / "training_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(status) + "\n")
        diagnostic = None
        if args.diagnostics:
            from crowd_counting.diagnostics import BusDiagnostics
            diagnostic = BusDiagnostics(run / "diagnostics" / f"val_epoch_{epoch:03d}",
                                        len(val_data), args.diagnostic_samples, features=args.diagnostic_features)
        validation = evaluate(model, val_batches, args.amp, diagnostics=diagnostic)
        improved = validation["mae"] < best
        if improved:
            best = validation["mae"]
            save_checkpoint(run / "best.pt", model, optimizer, epoch, scaler)
        save_checkpoint(run / "last.pt", model, optimizer, epoch, scaler)
        record = {"epoch": epoch, "train": {k: v / samples for k, v in totals.items()},
                  "skipped_optimizer_steps": skipped, "val": validation, "best_val_mae": best}
        with (run / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
