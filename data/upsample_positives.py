import os
import shutil
import random
from glob import glob

# Set your dataset paths
images_dir = "./yolo_potholes/train/images"
labels_dir = "./yolo_potholes/train/labels"

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Find positive and negative images
positive_images = []
negative_images = []

for label_path in glob(os.path.join(labels_dir, "*.txt")):
    with open(label_path) as f:
        lines = f.readlines()
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    img_path = None
    for ext in image_exts:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.isfile(candidate):
            img_path = candidate
            break
    if not img_path:
        continue  # skip if no image file exists for this label
    if any(line.strip() for line in lines):
        positive_images.append(img_path)
    else:
        negative_images.append(img_path)

n_pos = len(positive_images)
n_neg = len(negative_images)

if n_pos == 0 or n_neg == 0:
    print("No positive or negative images found.")
    exit(1)

print(f"Found {n_pos} positive and {n_neg} negative images.")

# Upsample positives to match negatives
copies_needed = n_neg - n_pos
if copies_needed <= 0:
    print("No upsampling needed.")
    exit(0)

for i in range(copies_needed):
    src_img = random.choice(positive_images)
    src_label = os.path.join(labels_dir, os.path.splitext(os.path.basename(src_img))[0] + ".txt")
    new_img = os.path.join(images_dir, f"upsampled_{i}_{os.path.basename(src_img)}")
    new_label = os.path.join(labels_dir, f"upsampled_{i}_{os.path.basename(src_label)}")
    shutil.copy(src_img, new_img)
    shutil.copy(src_label, new_label)

print(f"Upsampled positive images to match negatives. Now you have {n_neg} positives and {n_neg} negatives.")
