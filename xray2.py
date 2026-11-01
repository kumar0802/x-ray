import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for medical dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #0284C7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .class-normal {
        color: #166534;
        background-color: #DCFCE7;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }
    .class-bacterial {
        color: #9A3412;
        background-color: #FFEDD5;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }
    .class-viral {
        color: #991B1B;
        background-color: #FEE2E2;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL & CONSTANTS
# ==========================================
IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Pneumonia Bacterial", "Pneumonia Viral"]

@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model('xray.keras')
            return model, None
        except Exception as e:
            return None, str(e)
    return None, f"Model file not found at path: `{path}`"

def preprocess_image(img_pil):
    """Preprocess PIL Image for TensorFlow prediction."""
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    
    # Handle Grayscale images by converting to 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # Handle RGBA
        img_array = img_array[:, :, :3]
        
    img_scaled = img_array.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_scaled, axis=0)
    return img_batch

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/x-ray--v1.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Image Diagnosis", "Batch Metadata Explorer", "Model Diagnostics"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Info")
st.sidebar.text("Model Target: 224x224 RGB")
st.sidebar.text("Classes: 3 (Normal, Bact, Vir)")

# ==========================================
# PAGE 1: SINGLE IMAGE DIAGNOSIS
# ==========================================
if page == "Single Image Diagnosis":
    st.markdown('<p class="main-header">🩺 Chest X-Ray AI Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a chest radiograph to classify between Normal, Bacterial Pneumonia, and Viral Pneumonia.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Radiograph")
        uploaded_file = st.file_uploader("Choose an X-Ray image file...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

    with col2:
        st.subheader("2. AI Analysis & Classification")
        
        if uploaded_file is not None:
            model, err = load_trained_model(MODEL_PATH)
            
            if err:
                st.warning(f"⚠️ Demo Mode: Using Simulated Inference ({err})")
                # Fallback dummy predictions for demo purposes if model file not found local
                probs = [0.08, 0.76, 0.16]
            else:
                processed_img = preprocess_image(image)
                probs = model.predict(processed_img)[0]

            pred_class_idx = int(np.argmax(probs))
            pred_class_name = CLASS_NAMES[pred_class_idx]
            confidence = probs[pred_class_idx] * 100

            # Result Header Badge
            if pred_class_idx == 0:
                badge_html = f'<span class="class-normal">Normal</span>'
            elif pred_class_idx == 1:
                badge_html = f'<span class="class-bacterial">Pneumonia Bacterial</span>'
            else:
                badge_html = f'<span class="class-viral">Pneumonia Viral</span>'

            st.markdown(f"### Diagnosis Result: {badge_html}", unsafe_allow_html=True)
            st.metric("Top Class Confidence", f"{confidence:.2f}%")
            
            st.markdown("---")
            st.write("**Probability Distribution across Classes:**")
            
            for idx, c_name in enumerate(CLASS_NAMES):
                score = float(probs[idx])
                st.write(f"**{c_name}** ({score*100:.1f}%)")
                st.progress(score)

        else:
            st.info("👈 Please upload a Chest X-ray image in the left panel to begin classification.")

# ==========================================
# PAGE 2: BATCH METADATA EXPLORER
# ==========================================
elif page == "Batch Metadata Explorer":
    st.markdown('<p class="main-header">📊 Dataset & Metadata Filter</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Load and inspect the CoronaHack metadata catalog.</p>', unsafe_allow_html=True)

    csv_file = st.file_uploader("Upload Metadata CSV (Chest_xray_Corona_Metadata.csv)", type=["csv"])

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write(f"**Loaded Records:** {len(df)}")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            label_filter = st.multiselect("Filter by Label", options=df["Label"].unique(), default=df["Label"].unique())
        with filter_col2:
            dataset_type = st.multiselect("Filter by Dataset Type", options=df["Dataset_type"].unique(), default=df["Dataset_type"].unique())

        filtered_df = df[(df["Label"].isin(label_filter)) & (df["Dataset_type"].isin(dataset_type))]
        
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button("Download Filtered CSV", filtered_df.to_csv(index=False), "filtered_metadata.csv")
    else:
        st.info("Upload `Chest_xray_Corona_Metadata.csv` to explore category distributions.")

# ==========================================
# PAGE 3: MODEL DIAGNOSTICS
# ==========================================
elif page == "Model Diagnostics":
    st.markdown('<p class="main-header">📈 Model Training Architecture & Metrics</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Input Target Resolution", "224 x 224 x 3")
    col2.metric("Batch Size", "32")
    col3.metric("Optimizer", "Adam (lr=0.0001)")

    st.markdown("### Model Structure Overview")
    st.code("""
    Layer (type)                 Output Shape              Param #   
    =================================================================
    InputLayer                   (None, 224, 224, 3)       0         
    DataAugmentation             (None, 224, 224, 3)       0         
    Conv2D + BatchNorm + MaxPool (None, 112, 112, 32)      960       
    Conv2D + BatchNorm + MaxPool (None, 56, 56, 64)        18,560    
    Conv2D + BatchNorm + MaxPool (None, 28, 28, 128)       73,856    
    Conv2D + BatchNorm + MaxPool (None, 14, 14, 256)       295,168   
    GlobalAveragePooling2D       (None, 256)               0         
    Dense + Dropout (0.5)        (None, 256)               65,792    
    Dense (Softmax Output)       (None, 3)                 771       
    =================================================================
    """, language="text")
