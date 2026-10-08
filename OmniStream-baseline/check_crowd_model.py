"""Manual forward/backward check. Not a crowd-counting experiment.

Nothing is run on import. Run this file yourself after installing dependencies
and downloading the official OmniStream weights. The density head starts random.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/crowd_baseline.json"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--backward", action="store_true", help="Also perform one synthetic training step.")
    parser.add_argument("--save", help="Optional path to save and reload a complete model checkpoint.")
    args = parser.parse_args()

    import torch
    from crowd_counting import CrowdConfig, OmniStreamCrowd, CrowdCountingLoss
    from crowd_counting.engine import build_optimizer, train_step
    from crowd_counting.checkpoint import save_checkpoint, load_checkpoint

    if min(args.batch_size, args.frames, args.height, args.width) <= 0:
        parser.error("All dimensions must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected interpreter.")
    torch.manual_seed(42)
    cfg = CrowdConfig.from_json(args.config)
    model = OmniStreamCrowd.from_pretrained(cfg).to(args.device)
    model.eval()
    images = torch.rand(args.batch_size, args.frames, 3, args.height, args.width, device=args.device)
    with torch.no_grad():
        outputs = model(images)
    expected_shape = (args.batch_size, args.frames, 1, args.height // 4, args.width // 4)
    if tuple(outputs["density"].shape) != expected_shape:
        raise AssertionError(f"Wrong density shape: {outputs['density'].shape}")
    if not torch.isfinite(outputs["density"]).all() or (outputs["density"] < 0).any():
        raise AssertionError("Invalid density values.")
    torch.testing.assert_close(outputs["counts"], outputs["density"].float().sum(dim=(2, 3, 4)))

    # Causal inference: changing the last frame must not alter earlier predictions.
    if args.frames > 1:
        changed = images.clone()
        changed[:, -1] = torch.rand_like(changed[:, -1])
        with torch.no_grad():
            changed_output = model(changed)
        torch.testing.assert_close(outputs["density"][:, :-1], changed_output["density"][:, :-1], rtol=1e-4, atol=1e-5)

    report = {
        "notice": "Synthetic interface check only; random density head, not real counting accuracy.",
        "input_shape": list(images.shape),
        "density_shape": list(outputs["density"].shape),
        "counts_shape": list(outputs["counts"].shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_frozen": cfg.freeze_backbone,
    }
    optimizer = None
    if args.backward:
        target = torch.zeros(args.batch_size, args.frames, 1, args.height, args.width, device=args.device)
        target[..., args.height // 2, args.width // 2] = 1.0
        criterion = CrowdCountingLoss(cfg.density_loss_weight, cfg.count_loss_weight)
        optimizer = build_optimizer(model)
        report["synthetic_step"] = train_step(model, {"pixel_values": images, "density": target}, optimizer, criterion)
        grads = [p.grad for p in model.head.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
            raise AssertionError("No finite, nonzero gradient reached the density head.")
        if cfg.freeze_backbone and any(p.grad is not None for p in model.backbone.parameters()):
            raise AssertionError("Frozen backbone received gradients.")
        if not cfg.freeze_backbone and not any(p.grad is not None for p in model.backbone.parameters()):
            raise AssertionError("Unfrozen backbone received no gradients.")
        report["backward_check"] = "passed"
    if args.save:
        model.eval()
        with torch.no_grad():
            expected = model(images)["density"].cpu()
        save_checkpoint(args.save, model, optimizer=optimizer)
        del outputs, optimizer, model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        restored, _ = load_checkpoint(args.save, args.device)
        with torch.no_grad():
            actual = restored(images)["density"].cpu()
        torch.testing.assert_close(expected, actual)
        report["checkpoint_roundtrip"] = "passed"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
