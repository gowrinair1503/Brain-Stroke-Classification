import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Ensure 'model.h5' is in your repo
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((100, 100))  # Resize to match training input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit App UI
st.title("Brain Stroke Classification")
st.write("Upload a brain CT scan to check for stroke.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess & predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    
    # Display result
    result = "Stroke Detected" if prediction >= 0.5 else "No Stroke Detected"
    st.write(f"### Prediction: {result}")
