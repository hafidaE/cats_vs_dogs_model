"""
Dogs vs Cats Classification Web App
Streamlit application for deploying the trained model
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

# Page configuration
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .dog-prediction {
        background-color: #E3F2FD;
        border: 3px solid #1E88E5;
    }
    .cat-prediction {
        background-color: #FCE4EC;
        border: 3px solid #E91E63;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model directly from Hugging Face Hub"""
    try:
        with st.spinner("Downloading model weights from Hugging Face... (This happens only once)"):
            # This downloads the file securely and caches it on the Streamlit server
            model_file = hf_hub_download(
                repo_id="hafidaEr/dogs-vs-cats-vgg16", # e.g., "hafidat/dogs-vs-cats-vgg16"
                filename="dogs_vs_cats_transfer_model.keras" # Change to .h5 if that's your extension
            )
            model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Resize to model input size
    img = image.resize((350, 350))

    # Convert to array
    img_array = np.array(img)

    # Ensure RGB (in case of RGBA or grayscale)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict(model, image):
    """Make prediction on the image"""
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)
    probability = prediction[0][0]

    # 0 = Cat, 1 = Dog (based on alphabetical folder ordering)
    if probability > 0.5:
        label = "Dog"
        confidence = probability
        emoji = "🐕"
    else:
        label = "Cat"
        confidence = 1 - probability
        emoji = "🐈"

    return label, confidence, emoji

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🐾 Dogs vs Cats Classifier</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload an image of a dog or cat, and the AI model will predict which one it is!

    **Model Details:**
    - Architecture: Convolutional Neural Network (CNN) / Transfer Learning with VGG16
    - Training Dataset: 2000 images from Kaggle Dogs vs Cats competition
    - Input Size: 350x350 pixels
    """)

    # Load model
    model = load_model()

    if model is None:
        st.stop()

    st.success("Model loaded successfully!")

    # File uploader
    st.markdown("---")
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a dog or cat image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a dog or cat for best results"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        # Make prediction
        with st.spinner('Analyzing image...'):
            label, confidence, emoji = predict(model, image)

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        prediction_class = "dog-prediction" if label == "Dog" else "cat-prediction"

        st.markdown(f"""
        <div class="prediction-box {prediction_class}">
            <h1>{emoji} {label}!</h1>
            <h2>Confidence: {confidence*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar 
        st.progress(float(confidence))

        # Additional info
        with st.expander("See confidence breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dog Probability", f"{(confidence if label == 'Dog' else 1-confidence)*100:.2f}%")
            with col2:
                st.metric("Cat Probability", f"{(confidence if label == 'Cat' else 1-confidence)*100:.2f}%")

    else:
        # Show example
        st.info("👆 Upload an image to get started!")

        st.markdown("---")
        st.subheader("How it works")
        st.markdown("""
        1. **Upload** a JPG or PNG image of a dog or cat
        2. The image is **preprocessed** and resized to 350x350 pixels
        3. The CNN model **analyzes** the image features
        4. A **prediction** is made with confidence score
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and TensorFlow</p>
        <p>Project: MSDE Dogs vs Cats Classification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
