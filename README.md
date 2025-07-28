# Open Pothole Detector

This project provides a complete pipeline for training a YOLO12-based object detection model to identify potholes in road images. It includes scripts for data preparation, hyperparameter tuning, training, and evaluation.

## Features

- **End-to-End Workflow:** Scripts to manage the entire process from raw data to a trained model.
- **Data Processing:** Automatically converts Supervisely-formatted annotations to the YOLO format, and balances the dataset by upsampling positive samples.
- **Hyperparameter Tuning:** Uses Ray Tune to automatically find the best hyperparameters for training.
- **Training & Evaluation:** Trains a YOLO12 model and provides detailed evaluation metrics and visualizations.
- **Pre-trained Model:** Includes a YOLO12-Medium model fine-tuned on the pothole dataset.

## Project Structure

```
/open-pothole-detector
├── data/                  # Scripts and data for dataset preparation
│   ├── README.md
│   └── potholedatasetninja.tar
├── model/                 # Pre-trained model weights
│   ├── README.md
│   └── pothd-m.pt
├── training/              # Scripts for model training and evaluation
│   └── README.md
└── .gitignore
```

- **`data/`**: Contains scripts to process the raw dataset. See the `data/README.md` for detailed instructions on preparing the data.
- **`model/`**: Holds the pre-trained model weights. See `model/README.md` for model details and licensing.
- **`training/`**: Contains scripts for fine-tuning, training, and evaluating the model. See `training/README.md` for the full workflow.

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed. You will also need to install the required Python packages.

### 2. Installation

Clone the repository and install the dependencies:

```bash
# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install ultralytics pandas seaborn matplotlib scikit-learn tqdm pyyaml "ray[tune]"
```

### 3. Download the Dataset

The dataset used is the **Road Pothole Images** dataset from DatasetNinja, which is licensed under CC0.

1.  Download the `potholedatasetninja.tar` file from [this link](https://datasetninja.com/road-pothole-images#download).
2.  Place the downloaded `potholedatasetninja.tar` file into the `data/` directory.

### 4. Build the Dataset

Run the main data preparation script. This will extract the archive, convert annotations to YOLO format, upsample positive images, and create sanity-check images.

```bash
python data/build_dataset.py
```

This will create a `yolo_potholes/` directory containing the final dataset ready for training.

### 5. Train the Model

The training process involves two main steps:

**Step 1: Find the best hyperparameters (optional but recommended)**

Run the fine-tuning script to find the optimal hyperparameters for your setup. This uses Ray Tune and may take a significant amount of time.

```bash
python training/fine_tune.py
```

This will generate a `best_hyps.json` file in the project root.

**Step 2: Train the final model**

Train the model using the generated hyperparameters. If you skipped the tuning step, it will use default values.

```bash
python training/train_save.py
```

This will train the model and save the best weights to `runs/detect/train/weights/best.pt`, along with training plots in the `training_curves/` directory.

### 6. Evaluate the Model

Evaluate the performance of your newly trained model on the validation set.

```bash
python training/evaluate.py
```

This will print the metrics to the console and save them to `evaluation_metrics.json`.

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. The YOLO12 model architecture from Ultralytics is also licensed under AGPL-3.0. Any derivative works or services that use this project must also be open-sourced under the same license.
