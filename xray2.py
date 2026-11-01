"""
Streamlit Web App – Chest X-Ray Pneumonia Detection
Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch as tf
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Make sure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import CHECKPOINT_DIR, CLASS_NAMES, IMG_SIZE, BACKBONE
from model import create_model

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #0D9488;
    }
    .pred-normal {
        color: #059669;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .pred-pneumonia {
        color: #DC2626;
        font-size: 1.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- Model Loading --------------------
@st.cache_resource
def load_model():
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        return None, device, "No trained model found. Please run training first."

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", BACKBONE)

    model = create_model(backbone=backbone, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, None


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def predict(model, image: Image.Image, device):
    transform = get_transform()
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# -------------------- Grad-CAM (simplified) --------------------
def generate_gradcam(model, image: Image.Image, device, target_class=None):
    """
    Simple Grad-CAM for ResNet-style models.
    Returns a heatmap overlay as a PIL Image.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None  # library not installed

    # Find last conv layer
    target_layers = []
    if hasattr(model, "layer4"):
        target_layers = [model.layer4[-1]]
    elif hasattr(model, "features"):
        # EfficientNet / DenseNet-ish
        target_layers = [model.features[-1]]

    if not target_layers:
        return None

    transform = get_transform()
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb_img = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)


# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.title("About")
    st.markdown("""
    This app uses a **Convolutional Neural Network** (transfer learning)
    trained on chest X-ray images to detect signs of **Pneumonia**.

    **How to use**
    1. Upload a chest X-ray image (JPEG / PNG)
    2. Click **Predict**
    3. View the prediction, confidence, and optional Grad-CAM heatmap

    **Model**: Transfer Learning (ResNet / EfficientNet / DenseNet)
    **Classes**: Normal · Pneumonia
    """)
    st.markdown("---")
    st.caption("Capstone Project – Chest X-Ray Classification")


# -------------------- Main UI --------------------
st.markdown('<p class="main-header">🫁 Chest X-Ray Pneumonia Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image for AI-assisted screening</p>', unsafe_allow_html=True)

model, device, error_msg = load_model()

if error_msg:
    st.warning(error_msg)
    st.info("""
    **To train the model:**
    1. Place your images in the correct folder structure (see README)
    2. Run: `python src/prepare_data.py --source /path/to/images`
    3. Run: `python src/train.py`
    4. Refresh this app
    """)
else:
    st.success(f"Model loaded successfully on **{device}**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    use_gradcam = st.checkbox("Show Grad-CAM heatmap (requires pytorch-grad-cam)", value=True)

    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

with col2:
    st.subheader("Results")
    result_placeholder = st.empty()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if predict_btn and model is not None:
        with st.spinner("Analyzing image..."):
            pred_idx, probs = predict(model, image, device)
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            with result_placeholder.container():
                if pred_label == "Normal":
                    st.markdown(f'<p class="pred-normal">Prediction: {pred_label}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="pred-pneumonia">Prediction: {pred_label}</p>', unsafe_allow_html=True)

                st.progress(float(probs[pred_idx]))
                st.write(f"**Confidence:** {confidence:.1f}%")

                # Probability bars
                st.markdown("#### Class Probabilities")
                for i, name in enumerate(CLASS_NAMES):
                    st.write(f"{name}")
                    st.progress(float(probs[i]))
                    st.caption(f"{probs[i]*100:.1f}%")

                # Grad-CAM
                if use_gradcam:
                    st.markdown("#### Grad-CAM Explanation")
                    cam_img = generate_gradcam(model, image, device, target_class=pred_idx)
                    if cam_img is not None:
                        st.image(cam_img, caption="Regions the model focused on", use_container_width=True)
                    else:
                        st.info("Install `pytorch-grad-cam` for heatmap visualization:\n```\npip install grad-cam\n```")

                st.markdown("---")
                st.caption("⚠️ This tool is for educational / research purposes only and is **not** a substitute for professional medical diagnosis.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748B; font-size:0.85rem;'>"
    "Capstone Project · Chest X-Ray Classification using Deep Learning · "
    "Not for clinical use"
    "</div>",
    unsafe_allow_html=True
)
