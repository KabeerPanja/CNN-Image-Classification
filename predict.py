import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import img_to_array

IMAGE_SIZE = 128

model = None

def get_model():
    global model
    if model is None:
        model = load_model("models/cnn_model.h5")
    return model

def predict_image(image, class_names):

    image = np.array(image)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype("float") / 255.0

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    model = get_model()
    prediction = model.predict(image)[0]

    predicted_index = np.argmax(prediction)

    confidence = prediction[predicted_index] * 100

    predicted_label = class_names[predicted_index]

    return predicted_label, confidence