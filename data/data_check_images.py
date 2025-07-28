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

import argparse
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Plot sample images with bounding boxes to check data.")
    parser.add_argument("--dataset-path", type=str, default="./yolo_potholes", help="Path to the YOLO dataset directory.")
    parser.add_argument("--output-dir", type=str, default="./data_check", help="Directory to save output images.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random images to check.")
    parser.add_argument("--image-prefix", type=str, default="positive_", help="Only check images with this prefix.")
    return parser.parse_args()

def load_labels(label_path):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                boxes.append((int(cls), x, y, w, h))
    return boxes

def plot_image_with_boxes(img_path, label_path, class_names, out_path, resize=None):
    img = Image.open(img_path).convert('RGB')
    if resize:
        img = img.resize(resize)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    boxes = load_labels(label_path)
    w, h = img.size
    for _, x, y, bw, bh in boxes:
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        box_w = bw * w
        box_h = bh * h
        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    args = parse_args()
    
    dataset_path = Path(args.dataset_path).resolve()
    img_dir = dataset_path / "train/images"
    label_dir = dataset_path / "train/labels"
    yaml_path = dataset_path / "data.yaml"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        class_names = data.get('names', [str(i) for i in range(data.get('nc', 1))])

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    if args.image_prefix:
        img_files = [f for f in img_files if f.name.startswith(args.image_prefix)]
        
    if not img_files:
        print(f"No images found in {img_dir} with prefix '{args.image_prefix}'")
        return
        
    random.shuffle(img_files)
    sample_imgs = img_files[:args.num_samples]
    
    sizes = [None, (640, 640)]

    for img_path in sample_imgs:
        label_path = label_dir / (img_path.stem + ".txt")
        for sz in sizes:
            size_str = "native" if sz is None else f"{sz[0]}x{sz[1]}"
            out_path = out_dir / f"{img_path.stem}_{size_str}.jpg"
            plot_image_with_boxes(str(img_path), str(label_path), class_names, str(out_path), resize=sz)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
