import os
from PIL import Image
import numpy as np

# Create class folder
def create_class_folder(class_name):
    path = os.path.join("dataset", class_name)
    os.makedirs(path, exist_ok=True)
    return path

# Save uploaded images
def save_uploaded_images(uploaded_files, class_name):
    path = create_class_folder(class_name)

    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        image = image.convert("RGB")
        image.save(os.path.join(path, f"{class_name}_{i}.jpg"))

# Load dataset for training
def load_dataset(img_size=(128, 128)):
    X = []
    y = []
    labels = {}

    base_dir = "dataset"
    class_names = os.listdir(base_dir)

    for idx, class_name in enumerate(class_names):
        labels[class_name] = idx
        class_path = os.path.join(base_dir, class_name)

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            image = Image.open(img_path)
            image = image.resize(img_size)
            image = np.array(image) / 255.0

            X.append(image)
            y.append(idx)

    return np.array(X), np.array(y), labels
