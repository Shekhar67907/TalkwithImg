import streamlit as st
import openai
from PIL import Image
import tensorflow as tf
import numpy as np

# Set up OpenAI API
openai.api_key = 'your-openai-api-key'

# Load a pre-trained image classification model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Get image classification results
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]  # Return the most likely class name

# Generate a response using GPT
def generate_response(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("Image-Chatbot with Streamlit")
st.write("Upload an image and ask a question!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_class = classify_image(image)
    st.write(f"Image Classification: **{image_class}**")

    user_question = st.text_input("Ask a question based on the image:")

    if user_question:
        context = f"The image shows a {image_class}. {user_question}"
        chatbot_response = generate_response(context)
        st.write(f"Chatbot: {chatbot_response}")
