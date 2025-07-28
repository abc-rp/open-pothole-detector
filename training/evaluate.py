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