# # USING BINARY
# import streamlit as st
# import numpy as np
# import plotly.express as px
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from pathlib import Path

# # base_dir = Path(__file__).resolve().parent.parent.parent
# mildew_model_path = "outputs/mildew_model_softmax.h5"


# @st.cache_resource
# def get_model():
#     model = load_model(str(mildew_model_path))
#     return model


# def resize_input_image(img, target_size=(256, 256)):
#     image_array = img_to_array(img.resize(target_size))
#     image_array = np.expand_dims(image_array, axis=0) / 255.0
#     return image_array


# def load_model_and_predict(image_array, model_path=mildew_model_path):
#     model = load_model(str(model_path))
#     print("🔍 Using model path:", model_path)
#     prediction_prob = model.predict(image_array)[0][0]
#     predicted_class = "powdery_mildew" if prediction_prob > 0.5 else "healthy"
#     confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob

#     return predicted_class, round(confidence * 100, 2), prediction_prob


# def plot_predictions_probabilities(prediction_prob):
#     import pandas as pd

#     labels = ["healthy", "powdery_mildew"]
#     values = [1 - prediction_prob, prediction_prob]

#     df = pd.DataFrame({"Class": labels, "Probability": values})

#     fig = px.bar(
#         df,
#         x="Class",
#         y="Probability",
#         color="Class",
#         color_discrete_map={"healthy": "green", "powdery_mildew": "red"},
#         range_y=[0, 1],
#         height=300,
#     )

#     st.plotly_chart(fig)





# THIS IS THE CODES FOR THAT WHICH USES SOFTMAX
import streamlit as st
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path
import pandas as pd

# === Model path ===
mildew_model_path = "outputs/mildew_model_softmax.h5"

# === Preprocess image ===
def resize_input_image(img, target_size=(256, 256)):
    image_array = img_to_array(img.resize(target_size))
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    return image_array

# === Predict using softmax model ===
def load_model_and_predict(image_array, model_path=mildew_model_path):
    model = load_model(str(model_path))
    print("🔍 Using model path:", model_path)

    predictions = model.predict(image_array)[0]  # returns an array like [0.1, 0.9]
    class_index = np.argmax(predictions)
    labels = ["healthy", "powdery_mildew"]
    predicted_class = labels[class_index]
    confidence = round(predictions[class_index] * 100, 2)

    return predicted_class, confidence, predictions

# === Plot bar chart ===
def plot_predictions_probabilities(predictions):
    
    labels = ["healthy", "powdery_mildew"]

    df = pd.DataFrame({
        "Class": labels,
        "Probability": predictions
    })

    fig = px.bar(
        df,
        x="Class",
        y="Probability",
        color="Class",
        color_discrete_map={"healthy": "green", "powdery_mildew": "red"},
        range_y=[0, 1],
        height=300,
        title="Probability Breakdown"
    )

    st.plotly_chart(fig)
