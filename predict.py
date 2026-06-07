import numpy as np
from PIL import Image

from keras.models import load_model
from keras.preprocessing.image import img_to_array

IMAGE_SIZE = 128

model = None

def get_model():
    global model
    if model is None:
        model = load_model("models/cnn_model.h5")
    return model

def predict_image(image, labels):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    class_name = list(labels.keys())[list(labels.values()).index(class_index)]

    return class_name, confidence
