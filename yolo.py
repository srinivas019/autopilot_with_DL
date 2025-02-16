import os
import xml.etree.ElementTree as ET
import shutil
import random

def convert_voc_to_yolo(annotations_dir, images_dir, labels_dir, classes):
    os.makedirs(labels_dir, exist_ok=True)
    
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()

        image_name = root.find("filename").text
        image_w = int(root.find("size/width").text)
        image_h = int(root.find("size/height").text)

        label_file = os.path.join(labels_dir, os.path.splitext(xml_file)[0] + ".txt")
        
        with open(label_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue

                class_id = classes.index(class_name)
                bbox = obj.find("bndbox")
                xmin, ymin, xmax, ymax = [int(bbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]

                x_center = (xmin + xmax) / 2 / image_w
                y_center = (ymin + ymax) / 2 / image_h
                width = (xmax - xmin) / image_w
                height = (ymax - ymin) / image_h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def split_dataset(images_dir, labels_dir, train_dir, val_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir.replace("images", "labels"), exist_ok=True)
    os.makedirs(val_dir.replace("images", "labels"), exist_ok=True)
    
    images = [img for img in os.listdir(images_dir) if img.endswith(".jpg")]
    random.shuffle(images)
    
    split_index = int(train_ratio * len(images))
    train_images, val_images = images[:split_index], images[split_index:]
    
    for img in train_images:
        shutil.move(os.path.join(images_dir, img), train_dir)
        shutil.move(os.path.join(labels_dir, img.replace(".jpg", ".txt")), train_dir.replace("images", "labels"))
    
    for img in val_images:
        shutil.move(os.path.join(images_dir, img), val_dir)
        shutil.move(os.path.join(labels_dir, img.replace(".jpg", ".txt")), val_dir.replace("images", "labels"))

def create_yaml_file(dataset_path, classes):
    yaml_content = f"""
train: {dataset_path}/images/train
val: {dataset_path}/images/val
nc: {len(classes)}
names: {classes}
"""
    
    with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
        f.write(yaml_content)

def main():
    dataset_path = "datasets/custom_dataset"
    annotations_dir = f"{dataset_path}/annotations"
    images_dir = f"{dataset_path}/images"
    labels_dir = f"{dataset_path}/labels"
    train_dir = f"{images_dir}/train"
    val_dir = f"{images_dir}/val"
    classes = ["dog", "cat"]  # Change this to your classes
    
    convert_voc_to_yolo(annotations_dir, images_dir, labels_dir, classes)
    split_dataset(images_dir, labels_dir, train_dir, val_dir)
    create_yaml_file(dataset_path, classes)
    print("Dataset conversion and preparation completed!")

if __name__ == "__main__":
    main()
