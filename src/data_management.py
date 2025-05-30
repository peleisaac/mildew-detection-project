import json
import joblib
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image


def load_pkl_file(file_path):
    return joblib.load(file_path)


def load_class_indices(json_path="../outputs/class_indices.json"):
    with open(json_path, "r") as f:
        return json.load(f)


def load_and_prepare_image(uploaded_file, target_size=(256, 256)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array
