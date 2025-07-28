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
import shutil
import subprocess
import tarfile
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from Supervisely export.")
    parser.add_argument("--archive-path", type=str, default="data/potholedatasetninja.tar", help="Path to the Supervisely tar archive.")
    parser.add_argument("--dataset-name", type=str, default="potholedatasetninja", help="Name of the dataset directory after extraction.")
    parser.add_argument("--yolo-dir", type=str, default="yolo_potholes", help="Output directory for the YOLO dataset.")
    parser.add_argument("--train-splits", type=str, default="ds1_simplex-train,ds2_complex-train", help="Comma-separated list of training splits.")
    parser.add_argument("--val-splits", type=str, default="ds1_simplex-test,ds2_complex-test", help="Comma-separated list of validation splits.")
    return parser.parse_args()

def main():
    args = parse_args()

    print("--- Step 1: Extracting dataset ---")
    if not Path(args.archive_path).exists():
        print(f"Archive not found at {args.archive_path}")
        return
    with tarfile.open(args.archive_path, "r") as tar:
        tar.extractall(path="data")
    print(f"Extracted to data/{args.dataset_name}")

    print("\n--- Step 2: Converting to YOLO format ---")
    input_root = f"data/{args.dataset_name}"
    output_dir = args.yolo_dir
    cmd = [
        "python", "data/supervisely_to_yolo.py",
        "--input_root", input_root,
        "--output_dir", output_dir,
        "--train-splits", args.train_splits,
        "--val-splits", args.val_splits
    ]
    subprocess.run(cmd, check=True)

    print("\n--- Step 3: Upsampling positives ---")
    images_dir = f"{output_dir}/train/images"
    labels_dir = f"{output_dir}/train/labels"
    cmd = [
        "python", "data/upsample_positives.py",
        "--images-dir", images_dir,
        "--labels-dir", labels_dir
    ]
    subprocess.run(cmd, check=True)

    print("\n--- Step 4: Checking data ---")
    cmd = [
        "python", "data/data_check_images.py",
        "--dataset-path", output_dir
    ]
    subprocess.run(cmd, check=True)

    print("\n--- Dataset build complete! ---")
    print(f"YOLO dataset is in: {output_dir}")
    print(f"Data check images are in: data_check/")

if __name__ == "__main__":
    main()