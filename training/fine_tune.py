import json
import os

import ray
from ray import tune
from ultralytics import YOLO


def main():
    # 1. Initialize Ray
    ray.init()

    # 2. Path to the merged pothole dataset
    dataset_path = os.path.abspath("./yolo_potholes")

    # 3. Base YOLO model weights
    model = YOLO("yolo12m.pt")

    # 4. Define hyperparameter search space
    search_space = {
        "lr0": tune.loguniform(1e-4, 1e-2),
        "lrf": tune.loguniform(1e-3, 1e-1),
        "epochs": tune.choice([20, 30]),
        "batch": tune.choice([8, 16]),  # Adjust batch size for single GPU
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "warmup_epochs": tune.choice([2, 3, 5, 10]),
        "warmup_momentum": tune.uniform(0.6, 0.9),
        "optimizer": tune.choice(
            [
                "AdamW",
            ]
        ),
    }

    # 5. Run hyperparameter tuning (YOLO will store trial logs in runs/detect/tune/ by default)
    result_grid = model.tune(
        data=os.path.join(dataset_path, "data.yaml"),
        space=search_space,
        use_ray=True,
        gpu_per_trial=1,
        grace_period=5,
        iterations=100,  # Limit to 100 trials
        imgsz=640,
        augment=True,
    )
    
    # 6. Identify best trial (by mAP50)
    best_result = result_grid.get_best_result(metric="metrics/mAP50(B)", mode="max")
    best_config = best_result.config
    print("\n=== Best hyperparameters found ===")
    print(best_config)

    # 7. Save the best hyperparameters to a JSON file in the project root
    out_path = "best_hyps.json"
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best hyperparameters to '{out_path}'")

    print(
        "Next, run 'python evaluate.py' to train a final model with these hyperparams."
    )


if __name__ == "__main__":
    main()
