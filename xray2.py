
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Chest X-Ray Disease Detection")
st.write("AI-powered chest X-ray image classification")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "model/xray.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Update these labels to match your training classes
CLASS_NAMES = ["Normal", "Pneumonia"]

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-Ray")
        st.image(image, use_container_width=True)

    # -----------------------------
    # IMAGE PREPROCESSING
    # -----------------------------
    IMG_SIZE = (224, 224)

    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if st.button("Analyze X-Ray", type="primary"):

        with st.spinner("Analyzing image..."):

            predictions = model.predict(img_array)

            probabilities = predictions[0]

            predicted_index = np.argmax(probabilities)

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = probabilities[predicted_index] * 100

        with col2:
            st.subheader("Prediction Results")

            st.success(f"Prediction: {predicted_class}")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Class Probabilities")

            for i, class_name in enumerate(CLASS_NAMES):
                st.write(
                    f"{class_name}: "
                    f"{probabilities[i] * 100:.2f}%"
                )

            st.bar_chart({
                CLASS_NAMES[i]: float(probabilities[i] * 100)
                for i in range(len(CLASS_NAMES))
            })

st.divider()

st.caption(
    "Educational AI demonstration only. "
    "This model is not a substitute for professional medical diagnosis."
)
