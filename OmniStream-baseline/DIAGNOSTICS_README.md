# Bus baseline diagnostics

These additions were statically reviewed only. No Python, training, inference, or tests were executed during implementation.

## Evaluate an existing checkpoint first

Run in the project directory with OST1 activated. Replace the checkpoint path with an existing checkpoint. Settings and manifest are restored from its run directory.

```powershell
python train_bus.py evaluate --device cuda --checkpoint runs/YOUR_RUN/best.pt --split val --diagnostics
```

The default output directory is a new timestamped `diagnostics/val_...` directory beside the checkpoint. An optional `--diagnostic-dir` must point to a directory that does not already exist. Existing evaluations are not overwritten.

Optional intermediate features and temporal probes:

```powershell
python train_bus.py evaluate --device cuda --checkpoint runs/YOUR_RUN/best.pt --split val --diagnostics --diagnostic-features --temporal-diagnostics
```

Temporal probes cost up to three total forwards per validation frame. They require at least two input frames. They use the same weights and current frame with (1) real context, (2) current frame repeated, and (3) the first frame of the selected partition repeated as a fixed earlier donor. The first sample has no earlier donor and is excluded from that variant. Context near the partition start is inherently duplicated. No frames outside the selected partition are used; this is not an online cache or a random-shuffle experiment. CSV includes donor IDs. `paired_real_mae` compares the same subset for each variant. These probes do not establish causal explanations or replace separately trained single-frame baselines.

## New training run

```powershell
python train_bus.py train --device cuda --epochs 20 --frames 2 --full-frame --run-dir runs/bus_fullframe_diag --diagnostics
```

This starts a new run; it does not resume existing training. Choose an empty/new run directory. Validation diagnostics are written every epoch. No test evaluation is performed during training. Optional `--diagnostic-features` captures only the fixed validation samples. `--diagnostic-samples 6` controls both fixed and worst sample limits; zero disables images and feature capture. Without diagnostic flags, no diagnostic files or hooks are created.

## Output files

- `frames.csv`: one row per supervised frame: count, signed/absolute error, density MSE, raw density maximum/zero fraction, raw/ROI counts, excluded mass, target downsampling mass difference.
- `counts.png`: chronological target and prediction curves; x-axis is sample index, with IDs in CSV.
- `fixed_*.png`, `worst_*.png`: RGB / ROI / GT / prediction / signed density error. GT and prediction share a color range within the panel. Range is printed and may differ between epochs or samples. Red error means overprediction, blue means underprediction. Enlargement is display-only and must not be integrated for counts.
- Matching `.npz`: true and predicted ROI densities at native model output resolution, retaining count mass.
- `summary.json`: aggregate errors, density MSE, mass consistency, selected fixed indices.
- `features.json`, `features_*.npy`: optional input-to-head and two head-stage feature statistics, plus the first four channels of the current frame. The first four channels are arbitrary, not semantic labels. All saved arrays are detached CPU data. Nonfinite feature counts are reported. No full attention matrices are stored.
- `temporal.csv`, `temporal_summary.json`: optional sensitivity probes and matched-subset errors (empty when disabled).
- `evaluation.json`: standalone evaluation metadata and original MAE/RMSE.
- Run-level `training_diagnostics.jsonl`: first/every 50th batch losses, learning rates, gradient norm before clipping, skipped-step flag. Full epoch losses remain in `history.jsonl`.

Diagnostics use the existing model, ROI mask, last-frame supervision and count-preserving target alignment. Hooks are removed in a `finally` block. They do not change weights, losses, random augmentation, or optimizer configuration. Fixed/worst retained samples are bounded, while per-frame tables grow with partition length. PIL and NumPy are already used by the data loader; no new plotting dependency is needed.

Use validation for development. If explicitly evaluating `--split test`, keep that output separate and do not use it for repeated model selection.
