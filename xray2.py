"""Streamlit version of train.py.   Run:  streamlit run train_app.py"""
import json, os
import numpy as np, pandas as pd, streamlit as st, tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Train X-Ray Model", page_icon="🩻", layout="wide")
st.title("🩻 Train the chest X-ray classifier")
SIZE = 224

# ---------------- sidebar settings (defaults = your paths from train.py)
with st.sidebar:
    st.header("Settings")
    img_dir = st.text_input("Image folder", "D:/rahul/Coronahack-Chest-XRay-Dataset")
    csv_path = st.text_input("Metadata CSV", "D:/rahul/Chest_xray_Corona_Metadata.csv")
    out = st.text_input("Output folder", "model")
    bs = st.select_slider("Batch size", [8, 16, 32, 64], 32)
    e1 = st.slider("Epochs: train new head", 1, 15, 8)
    e2 = st.slider("Epochs: fine-tune top 30 layers", 0, 10, 5)
    val_frac = st.slider("Validation share of TRAIN", 0.05, 0.30, 0.15)

# ---------------- 1. metadata -> clean targets
@st.cache_data(show_spinner="Indexing images...")
def build_df(csv_path, img_dir):
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df.Label_1_Virus_category != "Stress-Smoking"]          # only 2 rows
    df["target"] = np.where(df.Label == "Normal", "Normal",
                    np.where(df.Label_1_Virus_category == "Virus", "Viral pneumonia", "Bacterial pneumonia"))
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk(img_dir) for f in fs}
    df["path"] = df.X_ray_image_name.map(paths)
    return df

if not os.path.exists(csv_path): st.error(f"CSV not found: {csv_path}"); st.stop()
if not os.path.isdir(img_dir): st.error(f"Image folder not found: {img_dir}"); st.stop()
df = build_df(csv_path, img_dir)
missing = int(df.path.isna().sum()); df = df.dropna(subset=["path"])
if df.empty: st.error("No images from the CSV were found in the image folder."); st.stop()
classes = sorted(df.target.unique()); idx = {c: i for i, c in enumerate(classes)}

c = st.columns(4)
c[0].metric("Images found", f"{len(df):,}"); c[1].metric("Missing on disk", missing)
c[2].metric("TRAIN", f"{(df.Dataset_type == 'TRAIN').sum():,}"); c[3].metric("TEST", f"{(df.Dataset_type == 'TEST').sum():,}")
st.bar_chart(pd.crosstab(df.target, df.Dataset_type))

# ---------------- helpers
def make_ds(d, shuffle=False):
    def load(p, y):
        im = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.image.resize(im, (SIZE, SIZE)), y
    ds = tf.data.Dataset.from_tensor_slices((d.path.values, d.target.map(idx).values))
    if shuffle: ds = ds.shuffle(len(d), seed=42)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)

class Live(tf.keras.callbacks.Callback):
    """Pushes every finished epoch to the page."""
    def __init__(self, hist, chart, status, bar, total):
        super().__init__(); self.hist, self.chart, self.status, self.bar, self.total = hist, chart, status, bar, total
    def on_epoch_end(self, epoch, logs=None):
        self.hist.append({k: float(logs[k]) for k in ["accuracy", "val_accuracy", "loss", "val_loss"]})
        h = pd.DataFrame(self.hist)
        self.chart.line_chart(h[["accuracy", "val_accuracy"]])
        self.bar.progress(min(len(h) / self.total, 1.0))
        self.status.write(f"Epoch {len(h)}/{self.total} | loss {logs['loss']:.3f} | val_acc {logs['val_accuracy']:.3f}")

# ---------------- 2. train
if st.button("🚀 Start training", type="primary"):
    os.makedirs(out, exist_ok=True)
    train_df = df[df.Dataset_type == "TRAIN"]; test_df = df[df.Dataset_type == "TEST"]
    train_df, val_df = train_test_split(train_df, test_size=val_frac, stratify=train_df.target, random_state=42)

    aug = tf.keras.Sequential([layers.RandomRotation(0.05), layers.RandomZoom(0.1), layers.RandomContrast(0.1)])
    base = tf.keras.applications.MobileNetV2((SIZE, SIZE, 3), include_top=False, weights="imagenet", name="backbone")
    base.trainable = False
    inp = layers.Input((SIZE, SIZE, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(aug(inp))
    x = base(x, training=False)
    x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D(name="gap")(x))
    model = tf.keras.Model(inp, layers.Dense(len(classes), activation="softmax", name="head")(x))

    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=train_df.target.map(idx).values)))
    tr, va, te = make_ds(train_df, True), make_ds(val_df), make_ds(test_df)

    st.subheader("Training progress")
    bar, status, chart = st.progress(0.0), st.empty(), st.empty()
    hist = []; live = Live(hist, chart, status, bar, e1 + e2)

    def fit(lr, epochs):
        model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(tr, validation_data=va, epochs=epochs, class_weight=cw, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), live])

    fit(1e-3, e1)                                   # phase 1: train the new head
    if e2 > 0:
        base.trainable = True
        for l in base.layers[:-30]: l.trainable = False
        fit(1e-5, e2)                               # phase 2: fine-tune the top layers

    # ---------------- 3. evaluate on held-out TEST set and save
    with st.spinner("Evaluating on the TEST set..."):
        probs = model.predict(te, verbose=0)
    y_true = test_df.target.map(idx).values; y_pred = probs.argmax(1)
    metrics = {"classification_report": classification_report(y_true, y_pred, target_names=classes, output_dict=True),
               "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
               "roc_auc_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr")), "classes": classes}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    json.dump(classes, open(f"{out}/class_names.json", "w"))
    model.save(f"{out}/xray_model.keras")

    st.success(f"Done. Model and metrics saved in '{out}/'. Now run: streamlit run app.py")
    st.metric("Test ROC-AUC (one-vs-rest)", f"{metrics['roc_auc_ovr']:.3f}")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).T.round(3), use_container_width=True)
    st.subheader("Confusion matrix (rows = true, columns = predicted)")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes))
