import gdown
from tensorflow.keras.models import load_model
import streamlit as st
import os

@st.cache_resource
def download_model():
    # Correct public link
    url = "https://drive.google.com/uc?id=1oIjbZPyBovtG8xN752Otc5KQpDIZgO9z"
    output = "improved_model.h5"
    
    # Check if the model already exists locally
    if not os.path.exists(output):
        st.write("Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)
        st.write("Download completed.")
    
    # Load the model
    return load_model(output)

# Call the function to load the model
model = download_model()

# Rest of your app code
st.title("Image Classification App")
st.write("Upload an image to classify it into categories.")
