import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("pneumonia.keras")
CLASSES_PATH = Path("pneumonia.json")
IMG_SIZE = (64, 64)  # must match the size used in training

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


st.title("🫁 Chest X-ray Classifier")
st.write("Upload a chest X-ray image and the model will predict its class.")
st.caption("Capstone project demo. Not a medical diagnosis tool.")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        f"Could not find `{MODEL_PATH}` and/or `{CLASSES_PATH}`. "
        "Run train.py first and keep them in the same folder as app.py."
    )
    st.stop()

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")  # training used 3-channel images
    st.image(img, caption="Uploaded image", use_container_width=True)

    # The model has a Rescaling layer inside, so pass raw 0-255 pixel values
    x = np.array(img.resize(IMG_SIZE), dtype="float32")
    x = np.expand_dims(x, axis=0)

    with st.spinner("Analyzing..."):
        probs = model.predict(x, verbose=0)[0]

    best = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[best]}")
    st.write(f"Confidence: {probs[best]:.1%}")

    st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})
