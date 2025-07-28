# Pothole Dataset Preparation

This directory contains the scripts to download, process, and prepare the pothole dataset for training.

## Dataset

The dataset used is the "Road Pothole Images" dataset from DatasetNinja.

- **Source:** [https://datasetninja.com/road-pothole-images#download](https://datasetninja.com/road-pothole-images#download)
- **License:** CC0 (Public Domain)

Before running the build script, download the `potholedatasetninja.tar` file from the link above and place it in this `data/` directory.

## Build Process

To prepare the dataset for YOLOv8 training, run the main build script:

```bash
python3 build_dataset.py
```

This single command will orchestrate the following steps:

1.  **Extract Archive:** The `potholedatasetninja.tar` archive will be extracted into a `potholedatasetninja/` directory.
2.  **Convert to YOLO:** The script `supervisely_to_yolo.py` will be called to convert the Supervisely-formatted data into the YOLOv8 format. This will create a `yolo_potholes/` directory containing `train/` and `val/` splits with corresponding `images/` and `labels/` subdirectories.
3.  **Upsample Positives:** The `upsample_positives.py` script will balance the training set by creating copies of images containing potholes until the number of positive samples matches the number of negative samples.
4.  **Check Data:** Finally, `data_check_images.py` will run to generate a `data_check/` directory containing a few sample images with their bounding boxes drawn. This provides a quick visual confirmation that the labels have been processed correctly.

## Scripts

- **`build_dataset.py`**: The main orchestration script. Run this to perform all data preparation steps.
- **`supervisely_to_yolo.py`**: Converts Supervisely JSON annotations to YOLO `.txt` label files.
- **`upsample_positives.py`**: Balances the dataset by oversampling images with potholes.
- **`data_check_images.py`**: Creates visual samples of the final labeled data for verification.
