
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
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
# MODEL CONFIGURATION
# -----------------------------
MODEL_PATH = "model/chest_xray_model.keras"

CLASS_NAMES = ["Normal", "Pneumonia Bacterial","Pneumonia Viral"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('xray.keras')

try:
    model = load_model()
    st.success("CNN Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Automatically detect input shape
input_shape = model.inputs[0].shape

IMG_HEIGHT = int(input_shape[1] or 224)
IMG_WIDTH = int(input_shape[2] or 224)
CHANNELS = int(input_shape[3] or 3)

st.sidebar.header("Model Information")
st.sidebar.write("Input shape:", str(input_shape))
st.sidebar.write("Image size:", f"{IMG_WIDTH} x {IMG_HEIGHT}")
st.sidebar.write("Channels:", CHANNELS)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    original_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-Ray")
        st.image(original_image, use_container_width=True)

    if st.button("Analyze X-Ray", type="primary"):

        try:
            with st.spinner("Analyzing X-ray..."):

                # -----------------------------
                # IMAGE PREPROCESSING
                # -----------------------------
                if CHANNELS == 1:
                    image = original_image.convert("L")
                elif CHANNELS == 3:
                    image = original_image.convert("RGB")
                else:
                    raise ValueError(
                        f"Unsupported channel count: {CHANNELS}"
                    )

                image = image.resize(
                    (IMG_WIDTH, IMG_HEIGHT)
                )

                img_array = np.array(
                    image, dtype=np.float32
                )

                if CHANNELS == 1:
                    img_array = np.expand_dims(
                        img_array, axis=-1
                    )

                # Normalize pixels
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array, axis=0
                )

                # -----------------------------
                # MODEL PREDICTION
                # -----------------------------
                predictions = model(
                    img_array, training=False
                )

                predictions = np.asarray(predictions)

                probabilities = predictions[0].astype(
                    np.float32
                ).flatten()

                # -----------------------------
                # HANDLE MODEL OUTPUT
                # -----------------------------
                if len(probabilities) == 1:

                    # Binary sigmoid output
                    p = float(probabilities[0])

                    if p < 0 or p > 1:
                        p = float(tf.math.sigmoid(p).numpy())

                    probabilities = np.array(
                        [1 - p, p]
                    )

                else:

                    # Multi-class output
                    if (
                        np.any(probabilities < 0)
                        or np.any(probabilities > 1)
                        or not np.isclose(
                            np.sum(probabilities), 1.0,
                            atol=0.01
                        )
                    ):
                        probabilities = tf.nn.softmax(
                            probabilities
                        ).numpy()

                if len(probabilities) != len(CLASS_NAMES):
                    raise ValueError(
                        f"Model returns {len(probabilities)} "
                        f"classes, but CLASS_NAMES contains "
                        f"{len(CLASS_NAMES)} labels."
                    )

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = (
                    float(probabilities[predicted_index]) * 100
                )

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            with col2:

                st.subheader("Prediction Results")

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Class Probabilities")

                for i, class_name in enumerate(CLASS_NAMES):

                    score = float(probabilities[i]) * 100

                    st.write(
                        f"**{class_name}:** {score:.2f}%"
                    )

                    st.progress(
                        min(max(float(probabilities[i]), 0), 1)
                    )

                st.subheader("Prediction Chart")

                chart_data = {
                    CLASS_NAMES[i]: float(
                        probabilities[i] * 100
                    )
                    for i in range(len(CLASS_NAMES))
                }

                st.bar_chart(chart_data)

        except Exception as e:

            st.error("Error while analyzing X-ray!")

            st.exception(e)

st.divider()

st.caption(
    "Educational AI demonstration only. "
    "This model is not a substitute for professional "
    "medical diagnosis."
)
