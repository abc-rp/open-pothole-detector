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
import json
import os
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model.")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt", help="Path to trained .pt weights.")
    parser.add_argument("--data", default="yolo_potholes/data.yaml", help="Path to dataset YAML.")
    parser.add_argument("--output", default="evaluation_metrics.json", help="Path to save metrics JSON.")
    parser.add_argument("--device", default=0, help="CUDA device index or 'cpu'.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data YAML not found: {args.data}")

    model = YOLO(args.weights)
    results = model.val(data=args.data, device=args.device)
    metrics = results.results_dict

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics written to '{args.output}'")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
