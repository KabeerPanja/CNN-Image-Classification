import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

IMAGE_SIZE = 128


def create_class_folder(class_name):
    path = os.path.join("dataset", class_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_images(uploaded_files, class_name):
    folder = create_class_folder(class_name)

    for file in uploaded_files:
        file_path = os.path.join(folder, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def load_dataset():
    data = []
    labels = []

    dataset_path = "dataset"

    class_names = sorted(os.listdir(dataset_path))

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            try:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = img_to_array(image)
                image = image / 255.0

                data.append(image)
                labels.append(class_name)

            except:
                pass

    return np.array(data), np.array(labels), class_names