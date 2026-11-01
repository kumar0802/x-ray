"""Launch with:  streamlit run app.py"""
import json, os
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
from matplotlib import cm

st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🩻", layout="wide")
CSV, MODEL_DIR, SIZE = "Chest_xray_Corona_Metadata.csv", "model", 224

@st.cache_data
def load_meta():
    d = pd.read_csv(CSV, index_col=0).rename(columns={"Label_1_Virus_category": "Type", "Label_2_Virus_category": "Sub-type"})
    d["Label"] = d.Label.replace({"Pnemonia": "Pneumonia"})
    d["Type"] = d.Type.fillna("None (normal)"); return d

@st.cache_resource
def load_model():
    import tensorflow as tf
    return (tf.keras.models.load_model(f"{MODEL_DIR}/xray_model.keras"),
            json.load(open(f"{MODEL_DIR}/class_names.json")))

def gradcam(model, arr):
    import tensorflow as tf
    bb, gap, head = model.get_layer("backbone"), model.get_layer("gap"), model.get_layer("head")
    x = tf.convert_to_tensor(arr[None] / 127.5 - 1.0, tf.float32)
    with tf.GradientTape() as tape:
        f = bb(x, training=False); tape.watch(f)
        p = head(gap(f)); loss = p[:, tf.argmax(p[0])]
    w = tf.reduce_mean(tape.gradient(loss, f), (1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * f, -1))[0].numpy()
    return cam / (cam.max() + 1e-8)

page = st.sidebar.radio("Navigate", ["Overview", "Data Explorer", "Predict", "Model Performance"])
st.sidebar.warning("Educational project. Not a medical device and not for diagnosis.")
meta = load_meta()

if page == "Overview":
    st.title("🩻 Chest X-Ray Pneumonia Classifier")
    st.write("Deep-learning capstone: MobileNetV2 transfer learning classifies chest X-rays as "
             "**Normal**, **Bacterial pneumonia** or **Viral pneumonia**, with Grad-CAM heatmaps showing what the model looked at.")
    c = st.columns(4)
    c[0].metric("Images", f"{len(meta):,}"); c[1].metric("Train", f"{(meta.Dataset_type=='TRAIN').sum():,}")
    c[2].metric("Test", f"{(meta.Dataset_type=='TEST').sum():,}"); c[3].metric("Pneumonia share", f"{(meta.Label=='Pneumonia').mean():.0%}")

elif page == "Data Explorer":
    st.header("Metadata explorer")
    split = st.radio("Split", ["All", "TRAIN", "TEST"], horizontal=True)
    d = meta if split == "All" else meta[meta.Dataset_type == split]
    a, b = st.columns(2)
    a.subheader("Label"); a.bar_chart(d.Label.value_counts())
    b.subheader("Pneumonia type"); b.bar_chart(d[d.Label == "Pneumonia"].Type.value_counts())
    st.subheader("Rare sub-labels"); st.dataframe(d["Sub-type"].value_counts().rename("images"))
    st.dataframe(d.head(200), use_container_width=True)

elif page == "Predict":
    st.header("Upload a chest X-ray")
    if not os.path.exists(f"{MODEL_DIR}/xray_model.keras"):
        st.error("No trained model found. Run `python train.py <image_folder>` first."); st.stop()
    up = st.file_uploader("X-ray image", type=["jpg", "jpeg", "png"])
    if up:
        model, classes = load_model()
        img = Image.open(up).convert("RGB").resize((SIZE, SIZE)); arr = np.array(img, dtype="float32")
        probs = model.predict(arr[None], verbose=0)[0]
        cam = np.array(Image.fromarray((gradcam(model, arr) * 255).astype("uint8")).resize((SIZE, SIZE)))
        heat = (cm.jet(cam / 255.0)[..., :3] * 255).astype("uint8")
        over = (0.6 * arr + 0.4 * heat).astype("uint8")
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Input"); c2.image(over, caption="Grad-CAM")
        c3.subheader(classes[int(probs.argmax())]); c3.write(f"Confidence: {probs.max():.1%}")
        c3.bar_chart(pd.Series(probs, index=classes))

else:
    st.header("Held-out test performance")
    path = f"{MODEL_DIR}/metrics.json"
    if not os.path.exists(path):
        st.info("Train the model first (`python train.py <image_folder>`)."); st.stop()
    m = json.load(open(path)); st.metric("ROC-AUC (one-vs-rest)", f"{m['roc_auc_ovr']:.3f}")
    rep = pd.DataFrame(m["classification_report"]).T
    st.dataframe(rep.round(3), use_container_width=True)
    st.subheader("Confusion matrix (rows = true, columns = predicted)")
    st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["classes"], columns=m["classes"]))
