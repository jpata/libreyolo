#!/usr/bin/env python3
"""
Convert standalone-segdet Lightning checkpoint to LibreYOLO v1.0 format.

Usage:
  python3 convert_segdet.py path/to/best_segdet_model.ckpt --size s
  # Produces: path/to/best_segdet_model_libreyolo.pt
"""

import argparse
import sys
from pathlib import Path

import torch

from libreyolo.models.segdet.model import LibreSegDet


def convert_checkpoint(ckpt_path, size, output_path=None):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = loaded.get("state_dict", loaded)
    if "model." in next(iter(state_dict.keys()), ""):
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

    model = LibreSegDet(model_path=None, size=size, nb_classes=2, num_thing_classes=1, task="detect")
    strict = model._strict_loading()
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    hyperparams = loaded.get("hyper_parameters", {})
    nc = hyperparams.get("nc", 2)
    ntc = hyperparams.get("num_thing_classes", 1)

    import json
    names = loaded.get("names", {i: f"class_{i}" for i in range(nc)})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except json.JSONDecodeError:
            names = {i: f"class_{i}" for i in range(nc)}

    libre_ckpt = {
        "model": model.model.state_dict(),
        "model_family": "segdet",
        "size": size,
        "nc": nc,
        "num_thing_classes": ntc,
        "task": "detect",
        "names": names,
        "schema_version": "1.0.0",
        "libreyolo_version": __import__("libreyolo").__version__,
    }

    if output_path is None:
        output_path = ckpt_path.with_name(ckpt_path.stem + "_libreyolo.pt")

    torch.save(libre_ckpt, output_path)
    print(f"Converted checkpoint saved to: {output_path}")
    print(f"  model_family: segdet")
    print(f"  size: {size}")
    print(f"  nc: {nc}")
    print(f"  num_thing_classes: {ntc}")
    print(f"  task: detect")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert SegDet checkpoint to LibreYOLO format")
    parser.add_argument("checkpoint", help="Path to Lightning .ckpt file")
    parser.add_argument("--size", default="s", choices=["n", "s", "m", "l"], help="Model size variant")
    parser.add_argument("--output", default=None, help="Output path (optional)")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.size, args.output)


if __name__ == "__main__":
    main()