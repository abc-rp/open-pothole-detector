import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml

def load_labels(label_path):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                boxes.append((int(cls), x, y, w, h))
    return boxes

def plot_image_with_boxes(img_path, label_path, class_names, out_path, resize=None):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size
    if resize:
        img = img.resize(resize)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    boxes = load_labels(label_path)
    w, h = img.size
    for cls, x, y, bw, bh in boxes:
        # Convert YOLO format to pixel coordinates
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        box_w = bw * w
        box_h = bh * h
        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        # ax.text(x1, y1, class_names[cls] if cls < len(class_names) else str(cls), color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    dataset_path = os.path.abspath("./yolo_potholes")
    img_dir = os.path.join(dataset_path, "train/images")
    label_dir = os.path.join(dataset_path, "train/labels")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    out_dir = os.path.join("data_check")
    os.makedirs(out_dir, exist_ok=True)

    # Load class names from data.yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        class_names = data.get('names', [str(i) for i in range(data.get('nc', 1))])

    img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.JPG")) + list(Path(img_dir).glob("*.png"))
    # Only use images with prefix positive_
    img_files = [f for f in img_files if f.name.startswith("positive_")]
    if not img_files:
        print(f"No images found in {img_dir} with prefix 'positive_'")
        return
    random.shuffle(img_files)
    sample_imgs = img_files[:5]
    sizes = [None, (640, 640), (512, 768)]  # None = native, others are resized
    for img_path in sample_imgs:
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        for sz in sizes:
            size_str = f"native" if sz is None else f"{sz[0]}x{sz[1]}"
            out_path = os.path.join(out_dir, f"{img_path.stem}_{size_str}.jpg")
            plot_image_with_boxes(str(img_path), str(label_path), class_names, out_path, resize=sz)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
