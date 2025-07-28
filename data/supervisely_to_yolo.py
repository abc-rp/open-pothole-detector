# Open Pothole Detector
# Copyright (C) 2025 xRI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Supervisely dataset(s) to YOLO format."
    )
    p.add_argument(
        "--input_root",
        required=True,
        help="Root of Supervisely export (contains meta.json and split dirs)",
    )
    p.add_argument(
        "--output_dir", required=True, help="Destination root for YOLO dataset"
    )
    p.add_argument(
        "--train-splits",
        type=str,
        default="ds1_simplex-train,ds2_complex-train",
        help="Comma-separated list of training split directories to merge",
    )
    p.add_argument(
        "--val-splits",
        type=str,
        default="ds1_simplex-test,ds2_complex-test",
        help="Comma-separated list of validation split directories to merge",
    )
    p.add_argument(
        "--task",
        choices=["det", "seg"],
        default="det",
        help="det for boxes, seg for polygons",
    )
    p.add_argument(
        "--sanity-samples",
        type=int,
        default=9,
        help="Number of random overlay images for visual QC",
    )
    p.add_argument(
        "--sanity-prefix",
        type=str,
        default="positive_",
        help="Prefix for sanity-check sample filenames",
    )
    p.add_argument(
        "--heatmap-grid",
        type=int,
        default=200,
        help="Resolution of the density heat-map",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_meta(meta_path: Path) -> tuple[dict[int, int], str]:
    meta = json.loads(Path(meta_path).read_text())
    classes = meta.get("classes", [])
    if len(classes) != 1:
        raise ValueError(
            f"Expected exactly one class in meta.json, found {len(classes)}"
        )
    super_id = classes[0]["id"]
    class_name = classes[0]["title"]
    return {super_id: 0}, class_name


def extract_points_from_obj(obj: dict) -> tuple[str | None, list[list[float]] | None]:
    if "geometryType" in obj and "points" in obj:
        return obj["geometryType"], obj["points"].get("exterior", [])
    if "geometry" in obj:
        geom = obj["geometry"]
        pts_data = geom.get("points")
        if isinstance(pts_data, dict):
            return geom.get("type"), pts_data.get("exterior", [])
        if isinstance(pts_data, list):
            return geom.get("type"), pts_data
    return None, None


def bbox_from_points(pts: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def write_data_yaml(out_path: Path, output_dir: Path) -> None:
    content = (
        f"train: {output_dir / 'train/images'}\n"
        f"val: {output_dir / 'val/images'}\n"
        "nc: 1\n"
        "names: ['pothole']\n"
    )
    out_path.write_text(content)


def aggregate_images_and_labels(
    splits: list[str], input_root: Path
) -> dict[str, list[tuple[str, str]]]:
    positives = []
    negatives = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in splits:
        src_img_dir = input_root / split / "img"
        ann_dir = input_root / split / "ann"
        for fname in sorted(os.listdir(src_img_dir)):
            if Path(fname).suffix.lower() not in exts:
                continue
            ann_path = ann_dir / f"{fname}.json"
            if not ann_path.exists():
                negatives.append((split, fname))
                continue
            ann_data = json.loads(ann_path.read_text())
            if ann_data.get("objects"):
                positives.append((split, fname))
            else:
                negatives.append((split, fname))
    return {"positive": positives, "negative": negatives}


def copy_and_create_labels(
    items: list[tuple[str, str]],
    input_root: Path,
    dest_img_dir: Path,
    dest_lbl_dir: Path,
    class_map: dict[int, int],
    task: str,
) -> None:
    ensure_dir(dest_img_dir)
    ensure_dir(dest_lbl_dir)
    for split, fname in tqdm(items, unit="img", desc="Copying and labeling"):
        src_img = input_root / split / "img" / fname
        dst_img = dest_img_dir / fname
        shutil.copy2(src_img, dst_img)

        ann_path = input_root / split / "ann" / f"{fname}.json"
        label_path = dest_lbl_dir / f"{Path(fname).stem}.txt"

        if not ann_path.exists():
            label_path.touch()
            continue

        sup_ann = json.loads(ann_path.read_text())
        width, height = sup_ann.get("size", {}).get("width"), sup_ann.get(
            "size", {}
        ).get("height")
        if not (width and height):
            with Image.open(src_img) as im:
                width, height = im.size

        lines: list[str] = []
        for obj in sup_ann.get("objects", []):
            if obj.get("classId") not in class_map:
                continue
            gtype, pts = extract_points_from_obj(obj)
            if not pts:
                continue
            if gtype and gtype.lower() == "rectangle" and len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[:2]
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            norm_pts = [
                f"{p}/{dim:.6f}" for x, y in pts for p, dim in [(x, width), (y, height)]
            ]

            if task == "seg":
                line = f"{class_map[obj['classId']]} {' '.join(norm_pts)}"
            else:
                x_min, y_min, x_max, y_max = bbox_from_points(pts)
                x_c = (x_min + x_max) / 2 / width
                y_c = (y_min + y_max) / 2 / height
                bw = (x_max - x_min) / width
                bh = (y_max - y_min) / height
                line = (
                    f"{class_map[obj['classId']]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
                )
            lines.append(line)
        label_path.write_text("\n".join(lines))


def load_labels_det(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            boxes.append((int(cls), x, y, w, h))
    return boxes


def build_heatmap(label_dir: Path, grid: int) -> np.ndarray:
    heat = np.zeros((grid, grid), dtype=np.float32)
    for lbl_file in tqdm(
        list(label_dir.glob("*.txt")), unit="lbl", desc="Heatmap accumulation"
    ):
        for _, x, y, _, _ in load_labels_det(lbl_file):
            ix = min(grid - 1, max(0, int(x * grid)))
            iy = min(grid - 1, max(0, int(y * grid)))
            heat[iy, ix] += 1.0
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def sanity_check(
    train_img_dir: Path,
    train_lbl_dir: Path,
    out_dir: Path,
    n_samples: int,
    prefix: str,
    heatmap_grid: int,
) -> None:
    if plt is None:
        print("Matplotlib not available, skipping sanity check.")
        return

    ensure_dir(out_dir)

    candidates = [p for p in train_img_dir.iterdir() if p.name.startswith(prefix)]
    random.shuffle(candidates)

    for img_path in tqdm(candidates[:n_samples], unit="img", desc="Sanity overlays"):
        label_path = train_lbl_dir / f"{img_path.stem}.txt"
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for _, x, y, bw, bh in load_labels_det(label_path):
            x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
            rect = patches.Rectangle(
                (x1, y1),
                bw * w,
                bh * h,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(out_dir / f"{img_path.stem}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    heat = build_heatmap(train_lbl_dir, heatmap_grid)
    fig, ax = plt.subplots(1)
    ax.imshow(heat, cmap="viridis", origin="lower")
    ax.set_title("Pothole spatial density")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_dir / "heatmap.jpg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    all_splits = [
        s.strip()
        for s in (args.train_splits + "," + args.val_splits).split(",")
        if s.strip()
    ]

    train_img_out = output_dir / "train/images"
    train_lbl_out = output_dir / "train/labels"
    val_img_out = output_dir / "val/images"
    val_lbl_out = output_dir / "val/labels"
    for p in (train_img_out, train_lbl_out, val_img_out, val_lbl_out):
        ensure_dir(p)

    class_map, _ = load_meta(input_root / "meta.json")
    agg = aggregate_images_and_labels(all_splits, input_root)

    pos_train, pos_val = train_test_split(
        agg["positive"], test_size=0.2, random_state=args.seed
    )
    neg_train, neg_val = train_test_split(
        agg["negative"], test_size=0.2, random_state=args.seed
    )

    copy_and_create_labels(
        pos_train + neg_train,
        input_root,
        train_img_out,
        train_lbl_out,
        class_map,
        args.task,
    )
    copy_and_create_labels(
        pos_val + neg_val, input_root, val_img_out, val_lbl_out, class_map, args.task
    )

    write_data_yaml(output_dir / "data.yaml", output_dir)

    if args.sanity_samples > 0:
        sanity_dir = output_dir / "sanity_check"
        sanity_check(
            train_img_out,
            train_lbl_out,
            sanity_dir,
            args.sanity_samples,
            args.sanity_prefix,
            args.heatmap_grid,
        )

    print(f"\nDone! YOLO dataset created under: {output_dir}")
    print(f"  • {len(pos_train) + len(neg_train)} train images -> {train_img_out}")
    print(f"  • {len(pos_val) + len(neg_val)} val images -> {val_img_out}")


if __name__ == "__main__":
    main()
