
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path("dataset/Coronahack-Chest-XRay-Dataset")

# Change these paths according to your dataset
IMAGE_DIR = BASE_DIR / "D:/rahul/Coronahack-Chest-XRay-Dataset"
CSV_PATH = BASE_DIR /"D:/rahul/Chest_xray_Corona_Metadata.csv"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "chest_xray_model.keras"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

CLASS_NAMES = [
    "Normal",
    "Pneumonia Bacterial",
    "Pneumonia Viral"
]

NUM_CLASSES = 3

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. CHECK DATASET
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

df = pd.read_csv(CSV_PATH)

print("\nDataset columns:")
print(df.columns.tolist())

required_columns = [
    "X_ray_image_name",
    "Label",
    "Label_1_Virus_category"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing CSV column: {column}")

# ==========================================
# 3. CREATE THREE CLASS LABELS
# ==========================================

def get_class(row):

    label = str(row["Label"]).strip().lower()

    category = str(
        row["Label_1_Virus_category"]
    ).strip().lower()

    if label == "normal":
        return 0

    if category == "bacteria":
        return 1

    if category == "virus":
        return 2

    # Exclude unknown or unsupported categories
    return None


df["class_id"] = df.apply(get_class, axis=1)

df = df.dropna(subset=["class_id"]).copy()

df["class_id"] = df["class_id"].astype(int)

print("\nClass distribution:")
print(
    df["class_id"].value_counts().sort_index()
)

# ==========================================
# 4. FIND IMAGE FILES
# ==========================================

print("\nScanning image directory...")

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif"
}

image_map = {}

for path in IMAGE_DIR.rglob("*"):

    if path.is_file() and path.suffix.lower() in image_extensions:

        image_map[path.name] = str(path)

df["image_path"] = df["X_ray_image_name"].map(image_map)

df = df.dropna(subset=["image_path"]).copy()

if df.empty:
    raise ValueError(
        "No matching images found. Check IMAGE_DIR and CSV_PATH."
    )

print("\nAvailable images:")
print(len(df))

# ==========================================
# 5. REMOVE DUPLICATE IMAGE RECORDS
# ==========================================

df = df.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)

# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["class_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["class_id"]
)

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# ==========================================
# 7. IMAGE PREPROCESSING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(dataframe, training=False):

    paths = dataframe["image_path"].values

    labels = dataframe["class_id"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(dataframe),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df, training=True)

val_ds = create_dataset(val_df)

test_ds = create_dataset(test_df)

# ==========================================
# 8. CLASS WEIGHTS
# ==========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_id"].values
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")
print(class_weights)

# ==========================================
# 9. CNN MODEL ARCHITECTURE
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.10),

])

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    data_augmentation,

    tf.keras.layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        256, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])

# ==========================================
# 10. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ==========================================
# 11. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

]

# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\nStarting CNN training...")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)

# Save final best model
model.save(MODEL_PATH)

print("\nModel saved:", MODEL_PATH)

# ==========================================
# 13. MODEL EVALUATION
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

y_pred = np.argmax(
    model.predict(test_ds),
    axis=1
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )
)

print("\nTraining completed successfully!")
