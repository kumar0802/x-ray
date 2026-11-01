import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import json
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🩻",
    layout="wide"
)

OUT = "model"
MODEL_PATH = os.path.join(OUT, "xray_model.keras")
METRICS_PATH = os.path.join(OUT, "metrics.json")
CLASSES_PATH = os.path.join(OUT, "class_names.json")
SIZE = 224

@st.cache_resource
def load_artifacts():
    """Load the trained model, class names, and evaluation metrics."""
    model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    classes = json.load(open(CLASSES_PATH)) if os.path.exists(CLASSES_PATH) else ["Normal", "Bacterial pneumonia", "Viral pneumonia"]
    metrics = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else None
    return model, classes, metrics

model, classes, metrics = load_artifacts()

# App Header
st.title("🩻 Chest X-Ray Pneumonia Classification")
st.markdown("Classify chest X-ray scans into **Normal**, **Bacterial Pneumonia**, or **Viral Pneumonia** using a fine-tuned MobileNetV2 model.")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Navigation", ["Image Prediction", "Model Performance & Metrics"])

---

if app_mode == "Image Prediction":
    st.header("Chest X-Ray Inference")
    
    if model is None:
        st.warning(f"⚠️ Model not found at `{MODEL_PATH}`. Please run your training script first to generate the saved model artifacts.")
    else:
        uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                with st.spinner("Analyzing image..."):
                    # Preprocess image to match training pipeline
                    img_resized = image.resize((SIZE, SIZE))
                    img_array = np.array(img_resized)
                    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
                    img_tensor = tf.expand_dims(img_tensor, 0)
                    
                    # Model Inference
                    preds = model.predict(img_tensor)[0]
                    pred_idx = np.argmax(preds)
                    pred_class = classes[pred_idx]
                    confidence = preds[pred_idx] * 100
                    
                    # Display Results
                    st.success(f"**Prediction:** {pred_class}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                    
                    st.markdown("### Class Probabilities")
                    for c, p in zip(classes, preds):
                        st.progress(float(p), text=f"{c}: {p*100:.2f}%")

---

elif app_mode == "Model Performance & Metrics":
    st.header("📊 Evaluation & Test Metrics")
    
    if metrics is None:
        st.warning(f"⚠️ Metrics file not found at `{METRICS_PATH}`. Please run training to generate evaluation metrics.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC (OvR)", f"{metrics.get('roc_auc_ovr', 0):.4f}")
        with col2:
            st.metric("Total Classes", len(classes))
            
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        st.subheader("Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        st.dataframe(cm_df, use_container_width=True)
