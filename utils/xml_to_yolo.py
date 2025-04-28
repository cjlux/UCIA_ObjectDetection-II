import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# 📂 Input and output directories
input_dir = "./images/annotations_xml"
output_dir = "./images/labels_yolo"
os.makedirs(output_dir, exist_ok=True)

# Optional: Auto-collect classes
class_set = set()

# Helper: Convert VOC bbox to YOLO format
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    return x_center * dw, y_center * dh, width * dw, height * dh

def convert_xml(xml_file, auto_classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_w = int(root.find('size/width').text)
    image_h = int(root.find('size/height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        auto_classes.add(class_name)  # Collect classes dynamically

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymin = int(float(bndbox.find('ymin').text))
        ymax = int(float(bndbox.find('ymax').text))

        bbox = (xmin, xmax, ymin, ymax)
        yolo_bbox = convert_bbox((image_w, image_h), bbox)
        yolo_line = f"{class_name} " + " ".join([f"{coord:.6f}" for coord in yolo_bbox])
        yolo_lines.append(yolo_line)

    return yolo_lines

# 🔄 Process all XML files recursively
annotations = defaultdict(list)
for root_dir, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".xml"):
            xml_path = os.path.join(root_dir, file)
            yolo_annotations = convert_xml(xml_path, class_set)

            image_id = os.path.splitext(file)[0]
            annotations[image_id] = yolo_annotations

# ✅ Generate classes.txt
class_list = sorted(class_set)
class_to_id = {name: idx for idx, name in enumerate(class_list)}

with open(os.path.join(output_dir, "classes.txt"), "w") as class_file:
    for class_name in class_list:
        class_file.write(f"{class_name}\n")

# ✅ Write YOLO .txt files with class IDs
for image_id, yolo_annots in annotations.items():
    txt_path = os.path.join(output_dir, f"{image_id}.txt")
    with open(txt_path, "w") as out_file:
        for annot in yolo_annots:
            class_name, *coords = annot.split()
            class_id = class_to_id[class_name]
            out_file.write(f"{class_id} {' '.join(coords)}\n")

print("✅ Conversion completed with class mapping!")
print(f"✔ Classes found: {class_list}")
