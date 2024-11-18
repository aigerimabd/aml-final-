import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
import numpy as np
from PIL import Image
import os

# Define class labels
class_indices = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

# Function to download and load the model
@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1oIjbZPyBovtG8xN752Otc5KQpDIZgO9z"
    output = "improved_model.h5"
    
    # Check if the model file already exists
    if not os.path.exists(output):
        st.write("Downloading model from Google Drive...")
        response = requests.get(url, stream=True)
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.write("Download completed.")

    # Load the model
    return load_model(output)

# Load the model
model = download_model()

# Preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image = img_to_array(image)      # Convert to NumPy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0           # Normalize pixel values
    return image

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the following categories:")
st.write(", ".join(class_indices.values()))

# File uploader for user input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess and classify the image
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # Display the results
    st.write(f"**Predicted Class:** {class_indices[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
