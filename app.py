import streamlit as st
from PIL import Image
from utils import create_class_folder, save_uploaded_images
from train import train_model
from predict import predict_image

st.set_page_config(page_title="CNN Image Classifier", layout="wide")

st.title("Custom CNN Image Classification App")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Create Dataset",
        "Train Model",
        "Predict"
    ]
)

# ==================================================
# CREATE DATASET
# ==================================================

if menu == "Create Dataset":

    st.header("Create Classes and Upload Images")

    class_name = st.text_input("Enter Class Name")

    if st.button("Create Class"):

        if class_name.strip() != "":
            create_class_folder(class_name)
            st.success(f"Class '{class_name}' created successfully")

    st.subheader("Upload Images")

    existing_classes = []

    if os.path.exists("dataset"):
        existing_classes = os.listdir("dataset")

    if len(existing_classes) > 0:

        selected_class = st.selectbox(
            "Select Class",
            existing_classes
        )

        uploaded_files = st.file_uploader(
            "Upload Images",
            type=['jpg', 'jfif', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if st.button("Save Images"):

            if uploaded_files:
                save_uploaded_images(uploaded_files, selected_class)
                st.success("Images uploaded successfully")

    st.subheader("Optional Webcam Capture")

    camera_image = st.camera_input("Capture Image")

    if camera_image:

        if len(existing_classes) > 0:

            webcam_class = st.selectbox(
                "Select Webcam Class",
                existing_classes,
                key="webcam"
            )

            if st.button("Save Webcam Image"):

                folder = os.path.join("dataset", webcam_class)

                file_path = os.path.join(
                    folder,
                    f"webcam_{len(os.listdir(folder))}.jpg"
                )

                with open(file_path, "wb") as f:
                    f.write(camera_image.getbuffer())

                st.success("Webcam image saved")

# ==================================================
# TRAIN MODEL
# ==================================================

elif menu == "Train Model":

    st.header("Train CNN Model")

    if st.button("Start Training"):

        with st.spinner("Training model..."):

            accuracy, class_names = train_model()

            st.success("Training Completed")

            st.write(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

            st.subheader("Accuracy Graph")
            st.image("charts/accuracy.png")

            st.subheader("Loss Graph")
            st.image("charts/loss.png")

            st.subheader("Confusion Matrix")
            st.image("charts/confusion_matrix.png")

# ==================================================
# PREDICTION
# ==================================================

elif menu == "Predict":

    st.header("Predict New Image")

    class_names = []

    if os.path.exists("dataset"):
        class_names = sorted(os.listdir("dataset"))

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict"):

            label, confidence = predict_image(image, class_names)

            st.success(f"Prediction: {label}")

            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("Optional Webcam Prediction")

    webcam_predict = st.camera_input("Capture for Prediction")

    if webcam_predict:

        webcam_image = Image.open(webcam_predict)

        st.image(webcam_image, width=300)

        if st.button("Predict Webcam Image"):

            label, confidence = predict_image(webcam_image, class_names)

            st.success(f"Prediction: {label}")

            st.info(f"Confidence: {confidence:.2f}%")
