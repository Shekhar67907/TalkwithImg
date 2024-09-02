import os
import subprocess
import sys

# Install the openai package if not installed
try:
    import openai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai
import streamlit as st
import openai
from PIL import Image
import tensorflow as tf
import numpy as np

# Load API key from Streamlit secrets
openai.api_key = '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'


# Caching the model to improve performance
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Get image classification results with error handling
def classify_image(image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
        return decoded_predictions[0][0][1]  # Return the most likely class name
    except Exception as e:
        st.error("Failed to classify the image. Please try again.")
        return None

# Generate a response using GPT with error handling
def generate_response(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error("Failed to generate a response. Please try again.")
        return None

# Streamlit UI
st.title("Image-Chatbot with Streamlit")
st.write("Upload an image and ask a question!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_class = classify_image(image)
    if image_class:
        st.write(f"Image Classification: **{image_class}**")

        user_question = st.text_input("Ask a question based on the image:")

        if user_question:
            if len(user_question.strip()) > 0:
                context = f"The image shows a {image_class}. {user_question}"
                chatbot_response = generate_response(context)
                if chatbot_response:
                    st.write(f"Chatbot: {chatbot_response}")
            else:
                st.warning("Please enter a valid question.")
