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
import random
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with best hyperparameters and save results.")
    parser.add_argument("--hyps-file", type=str, default="best_hyps.json", help="JSON file with hyperparameters.")
    parser.add_argument("--data", type=str, default="./yolo_potholes/data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", type=str, default="yolo12m.pt", help="Base YOLO model weights.")
    parser.add_argument("--output-dir", type=str, default="./training_curves", help="Directory to save plots.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.hyps_file):
        print(f"ERROR: '{args.hyps_file}' not found. Run 'python fine_tune.py' first.")
        sys.exit(1)

    with open(args.hyps_file) as f:
        best_hyps = json.load(f)

    print("=== Loaded best hyperparameters ===")
    print(best_hyps)

    model = YOLO(args.model)

    print("\n=== Training with best hyperparameters ===")
    results = model.train(
        data=args.data,
        epochs=best_hyps.get("epochs", 30),
        batch=best_hyps.get("batch", 16),
        lr0=best_hyps.get("lr0", 0.01),
        lrf=best_hyps.get("lrf", 0.01),
        momentum=best_hyps.get("momentum", 0.937),
        weight_decay=best_hyps.get("weight_decay", 0.0005),
        warmup_epochs=best_hyps.get("warmup_epochs", 3.0),
        warmup_momentum=best_hyps.get("warmup_momentum", 0.8),
        optimizer=best_hyps.get("optimizer", "AdamW"),
        imgsz=640,
        patience=best_hyps.get("patience", 20),
        augment=True,
    )

    train_dir = Path(results.save_dir)
    best_weights = train_dir / "weights/best.pt"

    if not best_weights.is_file():
        raise FileNotFoundError(f"Could not find best.pt at: {best_weights}")

    print(f"\nTraining complete! Best model weights are here: {best_weights}")

    results_csv = train_dir / "results.csv"
    if results_csv.is_file():
        print(f"\nPlotting training curves from: {results_csv}")
        df = pd.read_csv(results_csv)

        sns.set_theme(style="darkgrid", context="talk")
        mpl.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "0.8",
            "grid.color": "0.5",
            "text.color": "0.9",
            "axes.labelcolor": "0.9",
            "xtick.color": "0.9",
            "ytick.color": "0.9",
            "legend.facecolor": "#0d1117",
            "legend.edgecolor": "0.9",
        })

        output_plot_dir = Path(args.output_dir)
        output_plot_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fontsize = 30

        sns.lineplot(data=df, x="epoch", y="train/box_loss", label="box_loss", ax=axes[0], color="C0")
        sns.lineplot(data=df, x="epoch", y="train/cls_loss", label="cls_loss", ax=axes[0], color="C1")
        if "train/dfl_loss" in df.columns:
            sns.lineplot(data=df, x="epoch", y="train/dfl_loss", label="dfl_loss", ax=axes[0], color="C2")
        axes[0].set_title("Training Curves", fontsize=fontsize, color="0.9")
        axes[0].set_ylabel("Loss", color="0.9")
        axes[0].legend()
        axes[0].grid(True)

        sns.lineplot(data=df, x="epoch", y="val/box_loss", label="box_loss", ax=axes[1], color="C0")
        sns.lineplot(data=df, x="epoch", y="val/cls_loss", label="cls_loss", ax=axes[1], color="C1")
        if "val/dfl_loss" in df.columns:
            sns.lineplot(data=df, x="epoch", y="val/dfl_loss", label="dfl_loss", ax=axes[1], color="C2")
        axes[1].set_title("Validation Curves", fontsize=fontsize, color="0.9")
        axes[1].set_ylabel("Loss", color="0.9")
        axes[1].legend()
        axes[1].grid(True)

        metric_cols = {
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "mAP50",
            "metrics/mAP50-95(B)": "mAP50-95",
        }
        for i, (col, label) in enumerate(metric_cols.items()):
            if col in df.columns:
                sns.lineplot(data=df, x="epoch", y=col, label=label, ax=axes[2], color=f"C{i+3}")
        axes[2].set_title("Validation Metrics", fontsize=fontsize, color="0.9")
        axes[2].set_ylabel("Metric", color="0.9")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        out_plot = output_plot_dir / "training_curves.png"
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Training curves saved to '{out_plot}'")
    else:
        print(f"No results.csv found at: {results_csv}. Cannot plot training curves.")

    best_model = YOLO(best_weights)
    
    dataset_path = Path(args.data).parent
    val_images = dataset_path / "val/images"
    print(f"\n=== Running inference on validation images: {val_images} ===")
    best_model.predict(source=val_images, save=True, save_conf=True, name="val_inference")
    print("\nInference complete! Annotated images saved in: runs/detect/val_inference/")

    predict_root = Path("runs/detect")
    val_inference_folders = sorted(predict_root.glob("val_inference*"), key=lambda p: p.stat().st_mtime)

    if val_inference_folders:
        latest_val_inference_folder = val_inference_folders[-1]
        image_files = list(latest_val_inference_folder.glob("*.*"))
        if image_files:
            random.shuffle(image_files)
            sampled_images = random.sample(image_files, k=min(4, len(image_files)))

            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            fig.patch.set_facecolor("#0d1117")
            axes = axes.ravel()

            for i, img_path in enumerate(sampled_images):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis("off")

            plt.tight_layout()
            out_examples = output_plot_dir / "inference_examples.jpg"
            plt.savefig(out_examples, dpi=150)
            plt.close()
            print(f"Saved 2x2 sample of predicted images to '{out_examples}'")

if __name__ == "__main__":
    main()
