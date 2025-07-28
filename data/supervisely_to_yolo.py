#!/usr/bin/env python3
"""
supervisely_to_yolo_dataset_merge.py

Convert *multiple* **Supervisely** export splits – for example the paired
``ds1_simplex-*`` **and** ``ds2_complex_*`` directories – into a single consolidated
**YOLOv8** dataset directory **at native image resolution**. Any number of
training/validation splits can be supplied via the CLI.

Directory layout created:

<output_dir>/
 ├── train/
 │    ├── images/*.jpg|png         (original files, unmodified)
 │    └── labels/*.txt             (YOLO‑DET 5‑value format)
 ├── val/
 │    ├── images/*.jpg|png
 │    └── labels/*.txt
 ├── sanity_check/
 │    ├── *.jpg                    (overlayed previews)
 │    └── heatmap.jpg              (viridis density heat‑map)
 └── data.yaml

**What's new in this revision**
--------------------------------
* **Multiple‑split merging** – ``--train-splits`` and ``--val-splits`` accept
  *comma‑separated* lists so you can merge, e.g. ``ds1_simplex`` and
  ``ds2_complex`` in a single call.
* **Progress counters** – the final summary now shows *total* images coming
  from *all* supplied splits.

Usage example
-------------
```bash
python3 supervisely_to_yolo_dataset_merge.py \
  --input_root /datasets/potholedatasetninja \
  --output_dir  /tmp/yolo_potholes \
  --train-splits "ds1_simplex-train,ds2_complex_train" \
  --val-splits   "ds1_simplex-test,ds2_complex_test" \
  --sanity-samples 12
```
Add ``--task seg`` for polygon labels instead of boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml  # noqa: F401 (kept for potential downstream use)
import numpy as np  # for heat‑map
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # type: ignore

try:
    import matplotlib.pyplot as plt  # heavy import – only used for sanity check
    import matplotlib.patches as patches
except ImportError:  # pragma: no cover – headless / non‑sanity‑mode environments
    plt = None  # type: ignore


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert Supervisely dataset(s) to YOLO format.")

    p.add_argument("--input_root", required=True,
                   help="Root of Supervisely export (contains meta.json and split dirs)")
    p.add_argument("--output_dir", required=True,
                   help="Destination root for YOLO dataset")

    p.add_argument("--train-splits", type=str,
                   default="ds1_simplex-train,ds2_complex-train",
                   help="Comma‑separated list of *training* split directories to merge")
    p.add_argument("--val-splits", type=str,
                   default="ds1_simplex-test,ds2_complex-test",
                   help="Comma‑separated list of *validation* split directories to merge")

    p.add_argument("--task", choices=["det", "seg"], default="det",
                   help="det = 5‑value boxes, seg = polygon coords (YOLO‑Seg)")
    p.add_argument("--sanity-samples", type=int, default=9,
                   help="Generate N random overlay images for quick visual QC")
    p.add_argument("--sanity-prefix", type=str, default="positive_",
                   help="Only pick sanity‑check samples whose filename starts with this prefix")
    p.add_argument("--heatmap-grid", type=int, default=200,
                   help="Resolution (pixels) of the pothole density heat‑map")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_meta(meta_path: Path) -> Tuple[Dict[int, int], str]:
    meta = json.loads(Path(meta_path).read_text())
    classes = meta.get("classes", [])
    if len(classes) != 1:
        raise ValueError(f"Expected exactly one class in meta.json, found {len(classes)}")
    super_id = classes[0]["id"]
    class_name = classes[0]["title"]
    return {super_id: 0}, class_name  # map Supervisely id -> YOLO index 0


def copy_images(split: str, input_root: Path, dest_images_dir: Path) -> List[str]:
    """Copy all image files from *split* into ``dest_images_dir``.

    Returns a list of filenames actually copied (used later for label generation).
    """
    src_img_dir = input_root / split / "img"
    if not src_img_dir.exists():
        raise FileNotFoundError(f"Expected directory {src_img_dir} – check split name")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    copied: List[str] = []
    for fname in tqdm(sorted(os.listdir(src_img_dir)), unit="img", desc=f"{split}: copy"):
        if Path(fname).suffix.lower() not in exts:
            continue
        shutil.copy2(src_img_dir / fname, dest_images_dir / fname)
        copied.append(fname)
    return copied


def extract_points_from_obj(obj: dict) -> Tuple[str | None, List[List[float]] | None]:
    # Handles both old and new Supervisely formats
    if "geometryType" in obj and "points" in obj:
        return obj["geometryType"], obj["points"].get("exterior", [])
    if "geometry" in obj:
        geom = obj["geometry"]
        pts_data = geom.get("points")
        if isinstance(pts_data, dict):
            return geom.get("type"), pts_data.get("exterior", [])
        if isinstance(pts_data, list):  # already a list of points
            return geom.get("type"), pts_data
    return None, None


def bbox_from_points(pts: List[List[float]]) -> Tuple[float, float, float, float]:
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def create_labels(split: str, input_root: Path, dest_labels_dir: Path,
                  class_map: Dict[int, int], image_list: List[str],
                  task: str) -> None:
    ann_dir = input_root / split / "ann"
    for fname in tqdm(image_list, unit="ann", desc=f"{split}: labels"):
        ann_path = ann_dir / f"{fname}.json"
        label_path = dest_labels_dir / f"{Path(fname).stem}.txt"

        if not ann_path.exists():  # negative image
            label_path.touch()
            continue

        sup_ann = json.loads(ann_path.read_text())
        width = sup_ann.get("size", {}).get("width")
        height = sup_ann.get("size", {}).get("height")
        if not (width and height):
            with Image.open(input_root / split / "img" / fname) as im:
                width, height = im.size

        lines: List[str] = []
        for obj in sup_ann.get("objects", []):
            if obj.get("classId") not in class_map:
                continue
            gtype, pts = extract_points_from_obj(obj)
            if not pts:
                continue

            if gtype and gtype.lower() == "rectangle" and len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[:2]
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            # normalise polygon points to [0,1]
            norm_pts: List[str] = []
            for x, y in pts:
                norm_pts.extend([f"{x / width:.6f}", f"{y / height:.6f}"])

            if task == "seg":
                line = " ".join([str(class_map[obj["classId"]])] + norm_pts)
            else:  # det: need 5 values
                x_min, y_min, x_max, y_max = bbox_from_points(pts)
                x_c = (x_min + x_max) / 2 / width
                y_c = (y_min + y_max) / 2 / height
                bw = (x_max - x_min) / width
                bh = (y_max - y_min) / height
                line = f"{class_map[obj['classId']]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
            lines.append(line)

        label_path.write_text("\n".join(lines))


def write_data_yaml(out_path: Path, output_dir: Path) -> None:
    train_img = output_dir / "train/images"
    val_img = output_dir / "val/images"
    content = (
        f"train: {train_img}\n"
        f"val: {val_img}\n"
        "nc: 1\n"
        "names: ['pothole']\n"
    )
    out_path.write_text(content)


def aggregate_images_and_labels(splits, input_root):
    """
    Aggregate all images from all splits, and determine if each is positive (has at least one annotation)
    or negative (no annotation or empty annotation).
    Returns: dict with keys 'positive' and 'negative', each a list of (split, fname)
    """
    positives = []
    negatives = []
    for split in splits:
        src_img_dir = input_root / split / "img"
        ann_dir = input_root / split / "ann"
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for fname in sorted(os.listdir(src_img_dir)):
            if Path(fname).suffix.lower() not in exts:
                continue
            ann_path = ann_dir / f"{fname}.json"
            if not ann_path.exists():
                negatives.append((split, fname))
                continue
            ann_data = json.loads(ann_path.read_text())
            objs = ann_data.get("objects", [])
            if objs:
                positives.append((split, fname))
            else:
                negatives.append((split, fname))
    return {"positive": positives, "negative": negatives}


def copy_and_create_labels(items, input_root, dest_img_dir, dest_lbl_dir, class_map, task):
    """
    items: list of (split, fname)
    """
    ensure_dir(dest_img_dir)
    ensure_dir(dest_lbl_dir)
    for split, fname in tqdm(items, unit="img", desc=f"Copying and labeling"):
        # Copy image
        src_img = input_root / split / "img" / fname
        dst_img = dest_img_dir / fname
        shutil.copy2(src_img, dst_img)
        # Create label
        ann_dir = input_root / split / "ann"
        ann_path = ann_dir / f"{fname}.json"
        label_path = dest_lbl_dir / f"{Path(fname).stem}.txt"
        if not ann_path.exists():
            label_path.touch()
            continue
        sup_ann = json.loads(ann_path.read_text())
        width = sup_ann.get("size", {}).get("width")
        height = sup_ann.get("size", {}).get("height")
        if not (width and height):
            with Image.open(src_img) as im:
                width, height = im.size
        lines: List[str] = []
        for obj in sup_ann.get("objects", []):
            if obj.get("classId") not in class_map:
                continue
            gtype, pts = extract_points_from_obj(obj)
            if not pts:
                continue
            if gtype and gtype.lower() == "rectangle" and len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[:2]
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            norm_pts: List[str] = []
            for x, y in pts:
                norm_pts.extend([f"{x / width:.6f}", f"{y / height:.6f}"])
            if task == "seg":
                line = " ".join([str(class_map[obj["classId"]])] + norm_pts)
            else:  # det: need 5 values
                x_min, y_min, x_max, y_max = bbox_from_points(pts)
                x_c = (x_min + x_max) / 2 / width
                y_c = (y_min + y_max) / 2 / height
                bw = (x_max - x_min) / width
                bh = (y_max - y_min) / height
                line = f"{class_map[obj['classId']]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
            lines.append(line)
        label_path.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Sanity‑check visualisation & heat‑map (optional)
# -----------------------------------------------------------------------------


def load_labels_det(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            boxes.append((int(cls), x, y, w, h))
    return boxes


def build_heatmap(label_dir: Path, grid: int) -> np.ndarray:
    heat = np.zeros((grid, grid), dtype=np.float32)
    lbl_files = list(label_dir.glob("*.txt"))
    for lbl in tqdm(lbl_files, unit="lbl", desc="heat‑map accumulation"):
        for cls, x, y, _, _ in load_labels_det(lbl):
            # map centre to grid cell
            ix = min(grid - 1, max(0, int(x * grid)))
            iy = min(grid - 1, max(0, int(y * grid)))
            heat[iy, ix] += 1.0
    # normalise to 0‑1 for pretty colours
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def sanity_check(train_img_dir: Path, train_lbl_dir: Path, out_dir: Path,
                 n_samples: int, prefix: str, heatmap_grid: int) -> None:
    if plt is None:
        print("matplotlib not available – skipping sanity check & heat‑map")
        return

    ensure_dir(out_dir)

    # 1) Overlay previews
    candidates = [p for p in train_img_dir.iterdir() if p.name.startswith(prefix)]
    random.shuffle(candidates)
    samples = candidates[: n_samples]

    for img_path in tqdm(samples, unit="img", desc="sanity overlays"):
        label_path = train_lbl_dir / f"{img_path.stem}.txt"
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for cls, x, y, bw, bh in load_labels_det(label_path):
            x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
            rect = patches.Rectangle((x1, y1), bw * w, bh * h,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(out_dir / f"{img_path.stem}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # 2) Heat‑map across **all** training labels (negatives contribute nothing)
    heat = build_heatmap(train_lbl_dir, heatmap_grid)
    fig, ax = plt.subplots(1)
    ax.imshow(heat, cmap='viridis', origin='lower')
    ax.set_title('Pothole spatial density')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(out_dir / "heatmap.jpg", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Expand split lists from CLI
    all_splits = [s.strip() for s in (args.train_splits + "," + args.val_splits).split(',') if s.strip()]

    # Create output subdirs
    train_img_out = output_dir / "train/images"
    train_lbl_out = output_dir / "train/labels"
    val_img_out = output_dir / "val/images"
    val_lbl_out = output_dir / "val/labels"
    for p in (train_img_out, train_lbl_out, val_img_out, val_lbl_out):
        ensure_dir(p)

    # Load class map once (assume all splits share the same meta.json)
    class_map, _ = load_meta(input_root / "meta.json")

    # Aggregate all images and their annotation status
    agg = aggregate_images_and_labels(all_splits, input_root)
    positives = agg["positive"]
    negatives = agg["negative"]

    # Stratified split: 80% train, 20% val for each group
    pos_train, pos_val = train_test_split(positives, test_size=0.2, random_state=args.seed)
    neg_train, neg_val = train_test_split(negatives, test_size=0.2, random_state=args.seed)

    # Copy and create labels for train and val sets
    copy_and_create_labels(pos_train + neg_train, input_root, train_img_out, train_lbl_out, class_map, args.task)
    copy_and_create_labels(pos_val + neg_val, input_root, val_img_out, val_lbl_out, class_map, args.task)

    train_total = len(pos_train) + len(neg_train)
    val_total = len(pos_val) + len(neg_val)

    # data.yaml
    write_data_yaml(output_dir / "data.yaml", output_dir)

    # Optional sanity check (+ heatmap)
    if args.sanity_samples > 0:
        sanity_dir = output_dir / "sanity_check"
        sanity_check(train_img_out, train_lbl_out, sanity_dir,
                     args.sanity_samples, args.sanity_prefix, args.heatmap_grid)

    # Summary
    print("\nDone! YOLO dataset created under:", output_dir)
    print(f"  • {train_total} train   images →", train_img_out)
    print(f"  • {val_total}   val     images →", val_img_out)
    print(f"  • positives: {len(pos_train)} train, {len(pos_val)} val")
    print(f"  • negatives: {len(neg_train)} train, {len(neg_val)} val")
    print("  • data.yaml          →", output_dir / "data.yaml")
    if args.sanity_samples:
        print("  • sanity_check/*.jpg →", output_dir / "sanity_check")


if __name__ == "__main__":
    main()
