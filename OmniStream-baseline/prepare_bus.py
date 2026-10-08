"""Read-only annotation audit and reproducible split manifest; no model is loaded."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from crowd_counting.bus import make_manifest, load_roi, load_density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the Bus dataset directory.")
    parser.add_argument("--output", default="configs/bus_manifest.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="Validate every image and H5, rather than 3 per source split.")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite split manifest: {output}; choose another --output.")
    manifest = make_manifest(args.root, args.val_fraction, args.gap)
    root = Path(args.root)
    report = {"partitions": {k: len(manifest[k]) for k in ("train", "gap", "val", "test")},
              "full_audit": args.full, "count_reference": "sum of supplied H5 density inside binary ROI; not verified point counts"}
    shape = None
    roi = None
    for split in ("train", "test"):
        names = (manifest["train"] + manifest["gap"] + manifest["val"]) if split == "train" else manifest["test"]
        selected = names if args.full else [names[i] for i in sorted({0, len(names)//2, len(names)-1})]
        counts, roi_counts = [], []
        for i, name in enumerate(selected):
            with Image.open(root / split / "images" / (name + ".jpg")) as image:
                current = (image.height, image.width)
                image.verify()
            if shape is None:
                shape = current
                if any(value % 16 for value in shape):
                    raise ValueError("Expected image dimensions divisible by 16.")
                roi = load_roi(root, shape)
            if current != shape:
                raise ValueError(f"Image shape changed at {split}/{name}.")
            density = load_density(root / split / "ground_truth" / (name + ".h5"), shape)
            counts.append(float(density.sum(dtype=np.float64)))
            roi_counts.append(float((density * roi).sum(dtype=np.float64)))
            if args.full and (i + 1) % 200 == 0:
                print(f"Audited {split}: {i+1}/{len(selected)}", flush=True)
        report[split] = {"checked": len(selected), "full_density_count_range": [min(counts), max(counts)],
                         "roi_density_count_range": [min(roi_counts), max(roi_counts)]}
    report["image_shape_hw"] = shape
    report["roi_fraction"] = float(roi.mean())
    manifest["audit"] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Manifest saved: {output.resolve()}")


if __name__ == "__main__":
    main()
