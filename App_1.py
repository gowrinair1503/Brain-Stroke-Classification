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
    image = image.resize((100, 100)) 
    image = image.convert('RGB')# Resize to match training input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Custom CSS for pastel purple theme
st.markdown(
    """
    <style>
    body {
        background-color: #f3e5f5;
        color: #4a148c;
    }
    .stApp {
        background-color: #f8f0ff;
    }
    .stButton > button {
        background-color: #b39ddb;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #9575cd;
    }
    .stFileUploader label {
        color: #6a1b9a;
        font-weight: bold;
    }
    .result-box {
        background-color: #e1bee7;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #4a148c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App UI
st.markdown("<h1 style='text-align: center; color: #6a1b9a;'>🧠 Brain Stroke Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7b1fa2;'>Upload a brain CT scan to check for stroke.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess & predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    # Display result
    result = "🔴 Stroke Detected" if prediction >= 0.5 else "🟢 No Stroke Detected"
    
    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
