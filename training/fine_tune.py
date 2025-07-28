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
import ray
from ray import tune
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model using Ray Tune.")
    parser.add_argument("--data", type=str, default="./yolo_potholes/data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", type=str, default="yolo12m.pt", help="Base YOLO model weights.")
    parser.add_argument("--output-file", type=str, default="best_hyps.json", help="File to save the best hyperparameters.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of tuning trials.")
    parser.add_argument("--gpu-per-trial", type=int, default=1, help="Number of GPUs to allocate per trial.")
    return parser.parse_args()

def main():
    args = parse_args()
    ray.init()

    model = YOLO(args.model)

    search_space = {
        "lr0": tune.loguniform(1e-4, 1e-2),
        "lrf": tune.loguniform(1e-3, 1e-1),
        "epochs": tune.choice([20, 30]),
        "batch": tune.choice([8, 16]),
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "warmup_epochs": tune.choice([2, 3, 5, 10]),
        "warmup_momentum": tune.uniform(0.6, 0.9),
        "optimizer": tune.choice(["AdamW"]),
    }

    result_grid = model.tune(
        data=args.data,
        space=search_space,
        use_ray=True,
        gpu_per_trial=args.gpu_per_trial,
        grace_period=5,
        iterations=args.iterations,
        imgsz=640,
        augment=True,
    )
    
    best_result = result_grid.get_best_result(metric="metrics/mAP50(B)", mode="max")
    best_config = best_result.config
    
    print("\n=== Best hyperparameters found ===")
    print(best_config)

    with open(args.output_file, "w") as f:
        json.dump(best_config, f, indent=2)
        
    print(f"\nSaved best hyperparameters to '{args.output_file}'")
    print("Next, run 'python train_save.py' to train a final model with these hyperparams.")

if __name__ == "__main__":
    main()
