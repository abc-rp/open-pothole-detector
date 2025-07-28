# Model Training

This directory contains the scripts for fine-tuning, training, and evaluating the YOLOv8 pothole detection model.

## Workflow

The recommended workflow is as follows:

1.  **Fine-Tune:** Run `fine_tune.py` to find the optimal hyperparameters for the dataset. This script uses Ray Tune to automate the search process.
2.  **Train:** Once the best hyperparameters are found and saved, run `train_save.py` to train a new model from scratch using that configuration. This script also saves training plots and sample inference images.
3.  **Evaluate:** After the model is trained, run `evaluate.py` to get the final performance metrics on the validation set.

---

## Scripts

### 1. `fine_tune.py`

This script uses Ray Tune to perform a hyperparameter search for the YOLO model. It explores a predefined search space to find the configuration that yields the best Mean Average Precision (mAP50) on the validation data.

**Usage:**

```bash
python training/fine_tune.py
```

This command will start the tuning process and, upon completion, create a `best_hyps.json` file in the project root containing the optimal hyperparameters.

### 2. `train_save.py`

This script loads the hyperparameters from `best_hyps.json` and trains the YOLO model. After training is complete, it saves:

- The best model weights to `runs/detect/train/weights/best.pt`.
- Plots of training and validation curves (losses and metrics) to the `training_curves/` directory.
- A 2x2 grid of sample predictions on validation images to the `training_curves/` directory.

**Usage:**

```bash
python training/train_save.py
```

### 3. `evaluate.py`

This script takes the final trained model (`best.pt`) and evaluates its performance on the validation dataset. It saves the key performance metrics (mAP50, precision, recall, etc.) to a JSON file.

**Usage:**

```bash
python training/evaluate.py
```

This will create an `evaluation_metrics.json` file in the project root, allowing you to inspect the model's final performance.
