import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import openai

# Load API key from Streamlit secrets
openai.api_key = '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'


# Load the pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# Preprocess the image for the model
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to 224x224 pixels as required by MobileNetV2
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalize the image
    return img_array

# Function to classify the image using the pre-trained model
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]  # Return the most likely class name

# Function to generate a response using OpenAI's GPT
def generate_response(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("Image-Chatbot with Pre-trained Model")
st.write("Upload an image and ask a question!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_class = classify_image(image)
    st.write(f"Image Classification: **{image_class}**")

    user_question = st.text_input("Ask a question based on the image:")

    if user_question:
        if len(user_question.strip()) > 0:
            context = f"The image shows a {image_class}. {user_question}"
            chatbot_response = generate_response(context)
            st.write(f"Chatbot: {chatbot_response}")
        else:
            st.warning("Please enter a valid question.")
