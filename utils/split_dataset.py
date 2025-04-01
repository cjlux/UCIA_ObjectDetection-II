import os
import shutil
import random
from PIL import Image

# 📂 Directories
dataset_dir = "./dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
output_dir = "./yolo_split_dataset"

# ⚙️ Extensions and split ratios
image_ext = ".jpg"  # Change if needed
label_ext = ".txt"
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

# 📂 Create YOLO split folder structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

# 🔄 Gather image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(image_ext)]
random.shuffle(image_files)

# 🔀 Compute split sizes
total = len(image_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

# ✂️ Perform the split
splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

for split, files in splits.items():
    for img_file in files:
        label_file = os.path.splitext(img_file)[0] + label_ext

        img_src = os.path.join(images_dir, img_file)
        label_src = os.path.join(labels_dir, label_file)

        # convert image to gret scale
        img = Image.open(img_src).convert('L')
        img.save(img_src)
		
        img_dst = os.path.join(output_dir, split, "images", img_file)
        label_dst = os.path.join(output_dir, split, "labels", label_file)

        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"⚠️ Warning: Missing label for {img_file}")

print("✅ Dataset successfully split into train/val/test folders.")

# 🔎 Load class names from classes.txt
classes_txt = os.path.join(dataset_dir, "classes.txt")
if not os.path.exists(classes_txt):
    print("❌ Error: classes.txt not found in dataset folder!")
    exit()

with open(classes_txt, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# 📝 Auto-generate data.yaml for YOLOv8
data_yaml_path = os.path.join(output_dir, "data.yaml")
with open(data_yaml_path, "w") as yaml_file:
    yaml_file.write(f"path: {output_dir}\n\n")
    yaml_file.write(f"train: train/images\n")
    yaml_file.write(f"val: val/images\n")
    yaml_file.write(f"test: test/images\n\n")
    yaml_file.write(f"nc: {len(class_names)}\n")
    yaml_file.write("names:\n")
    for cls in class_names:
        yaml_file.write(f"  - {cls}\n")

print(f"✅ data.yaml created at {data_yaml_path}")
print(f"✔ Classes: {class_names}")
