import json
import os
import random
import sys
from pathlib import Path

# Ensure these are imported at the very top, outside of any if-block:
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from ultralytics import YOLO


def main():
    """
    Usage:
        python evaluate.py
    Steps:
      - Reads best hyperparameters from 'best_hyps.json'.
      - Trains YOLO with those hyperparams, saving best.pt in runs/detect/train/.
      - Plots the training curves (losses, metrics) using Seaborn and saves to ./training_curves/.
      - Runs inference on validation images, saving bounding-box-labeled images to runs/detect/val_inference/.
      - Randomly samples 4 predicted images and plots them in a 2x2 grid saved in ./training_curves/.
      - Data augmentations (crops, rotations, flips) can be enabled via YOLO's built-in augmentations.
      - To balance classes, upsample positive (pothole) images before training.
    """

    # 1. Load best hyperparameters from the JSON file
    hyps_file = "best_hyps.json"
    if not os.path.isfile(hyps_file):
        print(
            f"ERROR: '{hyps_file}' not found. Run 'python tune.py' first to generate it."
        )
        sys.exit(1)

    with open(hyps_file) as f:
        best_hyps = json.load(f)

    print("=== Loaded best hyperparameters ===")
    print(best_hyps)

    # 2. Path to dataset
    dataset_path = os.path.abspath("./yolo_potholes")

    # 3. Create a fresh YOLO model from base weights
    model = YOLO("yolo12m.pt")

    # 4. Re-train YOLO with the best hyperparameters
    print("\n=== Training with best hyperparameters ===")
    results = model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=best_hyps["epochs"],
        batch=best_hyps["batch"],
        lr0=best_hyps["lr0"],
        lrf=best_hyps["lrf"],
        momentum=best_hyps["momentum"],
        weight_decay=best_hyps["weight_decay"],
        warmup_epochs=best_hyps["warmup_epochs"],
        warmup_momentum=best_hyps["warmup_momentum"],
        optimizer=best_hyps["optimizer"],
        imgsz=640,
        patience=best_hyps.get("patience", 20),  # Early stopping: default 20 if not in best_hyps
        augment=True,  # Enable built-in augmentations (flips, rotations, crops, etc.)
        # etc. if more hyperparams exist in best_hyps
    )

    # YOLO typically saves the final model in runs/detect/train/weights/best.pt
    train_dir = results.save_dir  # e.g. 'runs/detect/train'
    best_weights = os.path.join(train_dir, "weights", "best.pt")

    if not os.path.isfile(best_weights):
        raise FileNotFoundError(
            f"Could not find best.pt at: {best_weights}\nCheck if YOLO training succeeded."
        )

    print(f"\nTraining complete! Best model weights are here: {best_weights}")

    # 5. Plot training curves using the 'results.csv' file YOLO generates
    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.isfile(results_csv):
        print(f"\nPlotting training curves from: {results_csv}")
        df = pd.read_csv(results_csv)

        # Step 1: Use Seaborn's "darkgrid" style and a talk-sized context
        sns.set_theme(style="darkgrid", context="talk")

        # Step 2: Update Matplotlib rcParams for a dark background and lighter text
        mpl.rcParams.update(
            {
                "figure.facecolor": "#0d1117",  # typical GitHub dark background
                "axes.facecolor": "#0d1117",
                "axes.edgecolor": "0.8",
                "grid.color": "0.5",
                "text.color": "0.9",
                "axes.labelcolor": "0.9",
                "xtick.color": "0.9",
                "ytick.color": "0.9",
                "legend.facecolor": "#0d1117",
                "legend.edgecolor": "0.9",
            }
        )

        # Create output directory for training plots
        os.makedirs("./training_curves", exist_ok=True)

        # Make a larger figure
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fontsize = 30

        #
        # Panel 1: Training losses
        #
        sns.lineplot(
            data=df,
            x="epoch",
            y="train/box_loss",
            label="box_loss",
            ax=axes[0],
            color="C0",  # use the default color cycle "C0" for distinction
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="train/cls_loss",
            label="cls_loss",
            ax=axes[0],
            color="C1",
        )
        if "train/dfl_loss" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="train/dfl_loss",
                label="dfl_loss",
                ax=axes[0],
                color="C2",
            )
        axes[0].set_title("Training Curves", fontsize=fontsize, color="0.9")
        axes[0].set_ylabel("Loss", color="0.9")
        axes[0].legend()
        axes[0].grid(True)

        #
        # Panel 2: Validation losses
        #
        sns.lineplot(
            data=df,
            x="epoch",
            y="val/box_loss",
            label="box_loss",
            ax=axes[1],
            color="C0",
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val/cls_loss",
            label="cls_loss",
            ax=axes[1],
            color="C1",
        )
        if "val/dfl_loss" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="val/dfl_loss",
                label="dfl_loss",
                ax=axes[1],
                color="C2",
            )
        axes[1].set_title("Validation Curves", fontsize=fontsize, color="0.9")
        axes[1].set_ylabel("Loss", color="0.9")
        axes[1].legend()
        axes[1].grid(True)

        #
        # Panel 3: Validation metrics
        #
        metric_colors = ["C3", "C4", "C5", "C6"]
        color_idx = 0

        if "metrics/precision(B)" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="metrics/precision(B)",
                label="precision",
                ax=axes[2],
                color=metric_colors[color_idx],
            )
            color_idx += 1
        if "metrics/recall(B)" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="metrics/recall(B)",
                label="recall",
                ax=axes[2],
                color=metric_colors[color_idx],
            )
            color_idx += 1
        if "metrics/mAP50(B)" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="metrics/mAP50(B)",
                label="mAP50",
                ax=axes[2],
                color=metric_colors[color_idx],
            )
            color_idx += 1
        if "metrics/mAP50-95(B)" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="metrics/mAP50-95(B)",
                label="mAP50-95",
                ax=axes[2],
                color=metric_colors[color_idx],
            )
            color_idx += 1

        axes[2].set_title("Validation Metrics", fontsize=fontsize, color="0.9")
        axes[2].set_ylabel("Metric", color="0.9")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        out_plot = "./training_curves/training_curves.png"
        plt.savefig(out_plot, dpi=150)  # higher dpi for clarity if desired
        plt.close()
        print(f"Training curves saved to '{out_plot}'")
    else:
        print(f"No results.csv found at: {results_csv}. Cannot plot training curves.")

    # 6. Load the best weights for inference
    best_model = YOLO(best_weights)

    # 7. Run inference on the validation images
    val_images = os.path.join(dataset_path, "val", "images")
    print(f"\n=== Running inference on validation images: {val_images} ===")
    predictions = best_model.predict(
        source=val_images,
        save=True,  # Overlays bounding boxes onto images
        save_conf=True,  # Renders confidence scores
        name="val_inference",  # subfolder name under runs/detect/
    )

    print("\nInference complete!")
    print("Annotated images saved in: runs/detect/val_inference/")

    # 8. Randomly sample 4 predicted images and plot them in a 2x2 grid
    predict_root = Path("runs/detect")
    val_inference_folders = sorted(
        predict_root.glob("val_inference*"), key=lambda p: p.stat().st_mtime
    )

    if not val_inference_folders:
        print("No val_inference* folders found in runs/detect/. Skipping 2x2 subplot.")
    else:
        latest_val_inference_folder = val_inference_folders[-1]
        print(f"Detected latest inference folder: {latest_val_inference_folder}")

        image_files = list(latest_val_inference_folder.glob("*.*"))
        if len(image_files) == 0:
            print(
                f"No images found in {latest_val_inference_folder}. Skipping 2x2 subplot."
            )
        else:
            random.shuffle(image_files)
            k = min(4, len(image_files))
            sampled_images = random.sample(image_files, k=k)

            # Keep the same dark style for the sample images
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            fig.patch.set_facecolor("#0d1117")  # background for the figure
            axes = axes.ravel()  # Flatten 2x2 to a list

            for i, img_path in enumerate(sampled_images):
                img = Image.open(img_path)
                axes[i].imshow(img)
                # No axis lines/ticks for a clean look
                axes[i].axis("off")

            plt.tight_layout()
            os.makedirs("./training_curves", exist_ok=True)
            out_examples = "./training_curves/inference_examples.jpg"
            plt.savefig(out_examples, dpi=150)
            plt.close()
            print(f"Saved 2x2 sample of predicted images to '{out_examples}'")

    print("\nAll done! You can open:")
    print("  - training_curves/training_curves.png (dark-themed training metrics plot)")
    print(
        "  - training_curves/inference_examples.jpg (random 2×2 sample of predictions)"
    )


if __name__ == "__main__":
    main()
