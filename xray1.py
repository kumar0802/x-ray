import tensorflow as tf
from pathlib import Path

# Find a directory whose immediate subfolders contain images (one folder per class)
base = Path(r"C:\Users\RAHUL KUMAR\Downloads\archive\Coronahack-Chest-XRay-Dataset\Coronahack-Chest-XRay-Dataset")
image_exts = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}

if not base.exists():
    raise FileNotFoundError(f"Dataset path does not exist: {base}")

data_dir = None
for folder in [base, *base.rglob("*")]:
    if folder.is_dir() and any(
        subfolder.is_dir()
        and any(file.suffix.lower() in image_exts for file in subfolder.iterdir() if file.is_file())
        for subfolder in folder.iterdir()
    ):
        data_dir = folder
        break

if data_dir is None:
    raise ValueError(
        "No class folders containing images were found. Organize images into "
        "subfolders (one per class), then set base to their parent directory."
    )

data = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    image_size=(64, 64),
    batch_size=2
)
# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax")
])
# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Train
model.fit(data, epochs=10)
model.summary()
# Prediction
test_dir = base / "test"
img_path = next(
    (p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts),
    None
)
if img_path is None:
    raise FileNotFoundError(f"No image files found in {test_dir}")

img = tf.keras.utils.load_img(str(img_path), target_size=(64, 64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)

p = model.predict(x)
print(data.class_names[p.argmax()])
model.save('covid.keras')

import json

# Save class names to a JSON file
with open("xray.json", "w") as f:
    json.dump(data.class_names, f)
    import json

# Read class names back from the JSON file
with open("xray.json", "r") as f:
    class_names = json.load(f)

import tensorflow as tf
m=tf.keras.models.load_model("covid.keras")
img = tf.keras.utils.load_img(str(img_path), target_size=(64, 64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)
p=m.predict(x)
print(p)
print(class_names[p.argmax()])
