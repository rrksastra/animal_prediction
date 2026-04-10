import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(page_title="Animal", page_icon="🐾")

# Title
st.title("🐾 Animal Detection")
st.caption("Upload an animal image")

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)
classes = ["kutya", "lepke", "tyuk"]

# Upload
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")   # ✅ ensure 3 channels
    st.image(image, use_column_width=True)

    # ✅ Preprocess (FIXED)
    img = image.resize((128, 128))  
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Debug (optional)
    # st.write(img.shape)  # should be (1, 128, 128, 3)

    # Predict
    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = np.max(pred) * 100

    st.subheader("Prediction")

    # Result
    st.success(f"Prediction: {classes[idx]} ({conf:.2f}%)")

    # Confidence
    st.subheader("Confidence")
    for i, p in enumerate(pred[0]):
        st.progress(float(p))
        st.write(f"{classes[i]}: {p*100:.2f}%")
