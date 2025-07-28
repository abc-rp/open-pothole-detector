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
import shutil
from glob import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Upsample positive images to balance the dataset.")
    parser.add_argument("--images-dir", type=str, default="./yolo_potholes/train/images", help="Directory containing training images.")
    parser.add_argument("--labels-dir", type=str, default="./yolo_potholes/train/labels", help="Directory containing training labels.")
    return parser.parse_args()

def main():
    args = parse_args()
    images_dir = args.images_dir
    labels_dir = args.labels_dir

    image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    positive_images = []
    negative_images = []

    label_files = glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        print(f"No label files found in {labels_dir}.")
        return

    for label_path in label_files:
        with open(label_path) as f:
            lines = f.readlines()
        
        base_name = Path(label_path).stem
        img_path = None
        for ext in image_exts:
            candidate = Path(images_dir) / (base_name + ext)
            if candidate.is_file():
                img_path = candidate
                break
        
        if not img_path:
            continue

        if any(line.strip() for line in lines):
            positive_images.append(str(img_path))
        else:
            negative_images.append(str(img_path))

    n_pos = len(positive_images)
    n_neg = len(negative_images)

    if n_pos == 0:
        print("No positive images found to upsample.")
        return
    
    if n_neg == 0:
        print("No negative images found to determine upsampling target.")
        return

    print(f"Found {n_pos} positive and {n_neg} negative images.")

    copies_needed = n_neg - n_pos
    if copies_needed <= 0:
        print("No upsampling needed (positive samples >= negative samples).")
        return

    for i in range(copies_needed):
        src_img_path = random.choice(positive_images)
        src_img_name = Path(src_img_path).name
        src_label_name = Path(src_img_path).stem + ".txt"
        src_label_path = os.path.join(labels_dir, src_label_name)

        new_img_name = f"upsampled_{i}_{src_img_name}"
        new_label_name = f"upsampled_{i}_{src_label_name}"

        dest_img_path = os.path.join(images_dir, new_img_name)
        dest_label_path = os.path.join(labels_dir, new_label_name)

        shutil.copy(src_img_path, dest_img_path)
        shutil.copy(src_label_path, dest_label_path)

    print(f"Upsampled {copies_needed} positive images. Now you have approximately {n_neg} positives and {n_neg} negatives.")

if __name__ == "__main__":
    main()
