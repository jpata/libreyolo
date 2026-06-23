#!/usr/bin/env python3
"""
Prepare Offsed dataset for LibreSegDet training.

Converts raw Offsed directory structure into libreYOLO-compatible format:
  raw/offsed/
    left/*.png                        → processed/offsed_segdet/images/*.png
    devkit_offsed/SegmentationClass/  → processed/offsed_segdet/masks/*.png (decoded RGB→class IDs)
    devkit_offsed/SegmentationObject/ → processed/offsed_segdet/labels/*.txt (derived YOLO bboxes)
    devkit_offsed/labelmap.txt        → processed/offsed_segdet/class_config.json

Usage:
  python3 prepare_offsed_data.py --raw data/raw/offsed --output data/processed/offsed_segdet
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops


def parse_offsed_labelmap(path):
    """Parse Offsed labelmap.txt. Returns (thing_names, stuff_names, rgb_to_cid)."""
    thing_class_names = []
    stuff_class_names = []
    rgb_to_name = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            if not name or name == "background":
                continue
            try:
                rgb = tuple(int(x) for x in parts[1].strip().split(","))
            except (ValueError, IndexError):
                continue
            thing_names = {"obstacle", "person", "held object", "car", "truck", "camper", "excavator"}
            if name in thing_names:
                thing_class_names.append(name)
            else:
                stuff_class_names.append(name)
            rgb_to_name[rgb] = name

    # Build sorted class list: background(0), things(sorted), stuff(sorted)
    thing_class_names = sorted(thing_class_names)
    stuff_class_names = sorted(stuff_class_names)
    all_names = ["background"] + thing_class_names + stuff_class_names
    name_to_id = {n: i for i, n in enumerate(all_names)}

    rgb_lookup = np.zeros((256, 256, 256), dtype=np.int64)
    for rgb, name in rgb_to_name.items():
        rgb_lookup[rgb[0], rgb[1], rgb[2]] = name_to_id.get(name, 0)

    return thing_class_names, stuff_class_names, name_to_id, rgb_lookup


def decode_offsed_instance_mask(mask_array):
    """Decode quantized RGB instance mask: each channel in {0,64,128,192}."""
    r = (mask_array[:, :, 0].astype(np.int32) // 64) * 16
    g = (mask_array[:, :, 1].astype(np.int32) // 64) * 4
    b = mask_array[:, :, 2].astype(np.int32) // 64
    return (r + g + b).astype(np.int32)


def prepare_offsed(raw_root, output_root):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "images").mkdir(exist_ok=True)
    (output_root / "labels").mkdir(exist_ok=True)
    (output_root / "masks").mkdir(exist_ok=True)

    labelmap_path = raw_root / "devkit_offsed" / "labelmap.txt"
    if not labelmap_path.exists():
        print(f"Error: labelmap.txt not found at {labelmap_path}")
        sys.exit(1)

    thing_names, stuff_names, name_to_id, rgb_lookup = parse_offsed_labelmap(labelmap_path)
    thing_name_to_yolo = {n: i for i, n in enumerate(thing_names)}
    print(f"Found {len(thing_names)} thing classes: {thing_names}")
    print(f"Found {len(stuff_names)} stuff classes")
    print(f"Total classes (incl background): {len(name_to_id)}")

    num_thing_classes = len(thing_names)
    class_config = {
        "thing_class_names": thing_names,
        "stuff_class_names": stuff_names,
        "num_classes": len(name_to_id),
        "num_thing_classes": num_thing_classes,
        "class_to_id": {n: i for i, n in enumerate(name_to_id.keys())},
        "thing_class_indices": [i for i, n in enumerate(name_to_id.keys()) if n in thing_names],
    }
    with open(output_root / "class_config.json", "w") as f:
        json.dump(class_config, f, indent=2)

    sem_dir = raw_root / "devkit_offsed" / "SegmentationClass"
    inst_dir = raw_root / "devkit_offsed" / "SegmentationObject"
    img_dir = raw_root / "left"

    if not sem_dir.exists() or not inst_dir.exists():
        print("Error: SegmentationClass or SegmentationObject directory not found")
        sys.exit(1)

    sem_paths = {p.stem: p for p in sem_dir.glob("*.png")}
    inst_paths = {p.stem: p for p in inst_dir.glob("*.png")}
    matched = sorted(set(sem_paths.keys()) & set(inst_paths.keys()))

    if not matched:
        print("Error: No labeled images found (intersection of SegmentationClass and SegmentationObject)")
        sys.exit(1)

    print(f"Processing {len(matched)} labeled images...")
    for i, stem in enumerate(matched):
        # Copy image
        src_img = img_dir / f"{stem}.png"
        if src_img.exists():
            shutil.copy2(src_img, output_root / "images" / f"{stem}.png")
        else:
            continue

        # Decode semantic mask
        sem_rgb = cv2.imread(str(sem_paths[stem]))
        sem_rgb = cv2.cvtColor(sem_rgb, cv2.COLOR_BGR2RGB)
        sem_ids = rgb_lookup[sem_rgb[:, :, 0], sem_rgb[:, :, 1], sem_rgb[:, :, 2]]
        cv2.imwrite(str(output_root / "masks" / f"{stem}.png"), sem_ids.astype(np.uint8))

        # Decode instance mask → bounding boxes
        inst_rgb = cv2.imread(str(inst_paths[stem]))
        inst_rgb = cv2.cvtColor(inst_rgb, cv2.COLOR_BGR2RGB)
        inst_ids = decode_offsed_instance_mask(inst_rgb)

        labels = []
        labeled, num_features = label(inst_ids, connectivity=2, return_num=True)
        for region in regionprops(labeled):
            if region.label == 0:
                continue
            minr, minc, maxr, maxc = region.bbox
            cy = (minr + maxr) / 2.0
            cx = (minc + maxc) / 2.0
            bh = maxr - minr
            bw = maxc - minc
            if bh < 2 or bw < 2:
                continue

            sample_c = int(cx)
            sample_r = int(cy)
            sample_r = min(sample_r, sem_ids.shape[0] - 1)
            sample_c = min(sample_c, sem_ids.shape[1] - 1)
            class_id = int(sem_ids[sample_r, sample_c])

            if class_id in class_config["thing_class_indices"]:
                yolo_cls = class_id - 1  # thing classes start at idx 1 in class_id space
                if 0 <= yolo_cls < num_thing_classes:
                    h_img, w_img = inst_rgb.shape[:2]
                    cx_norm = cx / w_img
                    cy_norm = cy / h_img
                    w_norm = bw / w_img
                    h_norm = bh / h_img
                    labels.append(f"{yolo_cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(output_root / "labels" / f"{stem}.txt", "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(matched)}")

    print(f"\nDone! Processed {len(matched)} images.")
    print(f"Output: {output_root}")
    print(f"  images/   -> {len(matched)} images")
    print(f"  masks/    -> {len(matched)} semantic masks")
    print(f"  labels/   -> {len(matched)} YOLO label files")
    print(f"  class_config.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare Offsed dataset for LibreSegDet")
    parser.add_argument("--raw", default="data/raw/offsed", help="Raw Offsed dataset path")
    parser.add_argument("--output", default="data/processed/offsed_segdet", help="Output path")
    args = parser.parse_args()
    prepare_offsed(args.raw, args.output)


if __name__ == "__main__":
    main()